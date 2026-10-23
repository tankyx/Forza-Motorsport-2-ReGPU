// The native renderer's entry points, Vulkan side: the same seam the Direct3D 12 renderer
// implements (gpu/renderer.h), so the command processor, the lookahead and the presenter do
// not know which backend is running. FM2_RENDERER=vulkan selects it.
//
// This is the seam and the frame loop, not the draw translation yet: the backend opens the
// device, builds the shaders the guest loads, keeps the EDRAM surfaces and ends each frame,
// and says plainly (once) that draws are not translated. Everything a draw will need is in
// place and separately verified (device, shader service, textures, state, index expansion,
// constants, submission, surfaces) — what is missing is the ~2 000 lines of DrawImpl that read
// the guest's state per draw, and those are worth writing with frames to look at.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "gpu/constants.h"
#include "gpu/indices.h"
#include "gpu/regs.h"
#include "gpu/renderer.h"
#include "gpu/vk/vk_context.h"
#include "gpu/vk/vk_draw.h"
#include "gpu/vk/vk_shaders.h"
#include "gpu/vk/vk_state.h"
#include "gpu/vk/vk_surface.h"
#include "gpu/vk/vk_texture.h"
#include "log.h"

namespace
{
struct StateKeyHash
{
    size_t operator()(const vk::StateKey& k) const
    {
        uint64_t h = k.vs * 0x9E3779B97F4A7C15ull ^ k.ps;
        h ^= (uint64_t(k.blend) << 32 | k.depth) * 0xC2B2AE3D27D4EB4Full;
        h ^= (uint64_t(k.stencilMaskMode) << 32 | k.flags) * 0x165667B19E3779F9ull;
        return size_t(h ^ (h >> 29));
    }
};

struct State
{
    bool enabled = false, initialised = false, failed = false;
    vk::Context context;
    vk::ShaderSet shaders;
    vk::SurfaceCache surfaces;
    vk::TextureCache textures;
    vk::DrawContext draw;
    VkSampler samplers[4]{};
    VkBuffer gmem = VK_NULL_HANDLE, gphys = VK_NULL_HANDLE;
    std::unordered_map<vk::StateKey, VkPipeline, StateKeyHash> pipelines;

    uint64_t vsHash = 0, psHash = 0;
    std::vector<uint32_t> vsCode, psCode;
    uint64_t frame = 0;
    bool reportedDraws = false;
    uint32_t surfacesSeen = 0;

    // The frame: one command buffer, one render pass per surface (the D3D12 path does the same
    // — the guest's tile replays become one pass), submitted when the guest swaps.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk::Surface* openSurface = nullptr;
    bool passOpen = false;
    uint64_t drawsThisFrame = 0;
} g;

float RegFloat(uint32_t index)
{
    const uint32_t word = __builtin_bswap32(gpureg::Get(index));
    float f;
    memcpy(&f, &word, 4);
    return f;
}

vk::StateKey MakeKey(uint64_t vs, uint64_t ps, bool cullable)
{
    vk::StateKey key;
    key.vs = vs;
    key.ps = ps;
    key.blend = gpureg::Get(0x2201);                              // RB_BLENDCONTROL0
    key.depth = gpureg::Get(0x2200);                              // RB_DEPTHCONTROL
    const uint32_t mode = gpureg::Get(gpureg::PA_SU_SC_MODE_CNTL) & 7;
    const uint32_t mask = gpureg::Get(0x2104) & 0xF;              // RB_COLOR_MASK
    const uint32_t stencil = gpureg::Get(0x210D) & 0xFFFF00;      // RB_STENCILREFMASK (masks only)
    key.stencilMaskMode = stencil | (mask << 24) | (mode << 28);
    key.flags = ((gpureg::Get(0x2202) >> 4) & 1) | (uint32_t(cullable) << 1); // RB_COLORCONTROL bit 4
    return key;
}

VkPipeline PipelineFor(const vk::StateKey& key, const vk::VkShader* vs, const vk::VkShader* ps, vk::Surface& surface)
{
    if (auto it = g.pipelines.find(key); it != g.pipelines.end())
        return it->second;
    vk::TargetFormats targets;
    targets.samples = surface.samples;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (!vk::CreatePipeline(g.context, key, vs->module, ps->module, g.draw.layout, surface.pass, targets, pipeline))
        return VK_NULL_HANDLE;
    g.pipelines[key] = pipeline;
    return pipeline;
}

// Closes the pass the current surface has open, if any.
void ClosePass()
{
    if (!g.passOpen)
        return;
    vk::SurfaceEnd(g.context, *g.openSurface, g.cmd);
    g.passOpen = false;
    g.openSurface = nullptr;
}

// The constant blocks, filled the way the D3D12 path fills them (renderer.cpp: DrawImpl).
void BuildConstants(const vk::VkShader* vs, const vk::VkShader* ps, gpu::VSConstants& vc, gpu::PSConstants& pc)
{
    const uint8_t* banks = gpureg::File() + gpureg::AluConstants * 4;
    const uint32_t nv = std::min(vs->constantsUsed, 256u), np = std::min(ps->constantsUsed, 256u);
    for (uint32_t i = 0; i < nv; i++)
        for (uint32_t c = 0; c < 4; c++)
        {
            uint32_t word;
            memcpy(&word, banks + i * 16 + c * 4, 4);
            word = __builtin_bswap32(word);
            memcpy(&vc.c[i][c], &word, 4);
        }
    for (uint32_t i = 0; i < np; i++)
        for (uint32_t c = 0; c < 4; c++)
        {
            uint32_t word;
            memcpy(&word, banks + 256 * 16 + i * 16 + c * 4, 4);
            word = __builtin_bswap32(word);
            memcpy(&pc.c[i][c], &word, 4);
        }
    for (uint32_t i = 0; i < 8; i++)
        vc.b[i] = pc.b[i] = gpureg::Get(gpureg::BoolConstants + i);
    // Vertex fetch: where each buffer is (vf) and the shader's layout of it (vfl).
    for (uint32_t k : vs->vertexFetchConstants)
    {
        const uint32_t d0 = gpureg::Get(gpureg::FetchConstants + k * 2), d1 = gpureg::Get(gpureg::FetchConstants + k * 2 + 1);
        vc.vf[k][0] = (d0 & 0xFFFFFFFC) & 0x1FFFFFFF; // physical offset inside gphys
        vc.vf[k][1] = d1 & 3;                         // endian
        vc.vf[k][2] = 1;                              // read from gphys
        vc.vf[k][3] = 0;
    }
    const size_t layout = std::min(vs->vertexFetchLayout.size(), sizeof(vc.vfl) / sizeof(vc.vfl[0])) * sizeof(vc.vfl[0]) / sizeof(uint32_t);
    if (layout)
        memcpy(vc.vfl, vs->vertexFetchLayout.data(), layout * sizeof(uint32_t));
    vc.info[1] = (gpureg::Get(0x2000) & 0x3FFF) ? 0 : 0; // position mode: clip
    vc.vscale[0] = RegFloat(0x210F);
    vc.vscale[1] = RegFloat(0x2111);
    vc.vscale[2] = RegFloat(0x2113);
    vc.vscale[3] = 0;
    vc.voffset[0] = RegFloat(0x2110);
    vc.voffset[1] = RegFloat(0x2112);
    vc.voffset[2] = RegFloat(0x2114);
    vc.voffset[3] = 0;
    vc.vte[0] = gpureg::Get(0x2206); // PA_CL_VTE_CNTL
    vc.vte[1] = vc.vte[2] = vc.vte[3] = 0;
    vc.vwide[0] = vc.vwide[1] = vc.vwide[2] = vc.vwide[3] = 1.0f;
    // Pixel shader: alpha test and the texture tables. Indices point at the null texture until
    // the texture capture exists (decode + upload), so a draw samples white rather than garbage.
    pc.info[0] = 7; // alpha test always passes
    pc.info[1] = 0xFF;
    pc.info2[0] = 0;
    pc.info2[1] = 1.0f;
    pc.info2[2] = 1.0f;
    pc.info2[3] = 0;
    for (int slot = 0; slot < 32; slot++)
    {
        pc.tinfo[slot][0] = 0; // linear wrap, 2D
        pc.tinfo[slot][1] = 0;
        pc.tinfo[slot][2] = 0;
        pc.tinfo[slot][3] = 0;
        pc.tbias[slot][0] = 0.0f;
        pc.tbias[slot][1] = 0.0f; // bindless index 0
        pc.tbias[slot][2] = 0.0f;
        pc.tbias[slot][3] = 0.0f;
    }
}
}

bool VulkanRendererEnabled()
{
    static const bool enabled = getenv("FM2_RENDERER") && !strcmp(getenv("FM2_RENDERER"), "vulkan");
    return enabled;
}

namespace
{
bool Init()
{
    if (g.initialised)
        return !g.failed;
    g.initialised = true;
    if (!vk::Init(g.context, "out/vkcache"))
    {
        Log("renderer_vk: no Vulkan device");
        g.failed = true;
        return false;
    }
    Log("renderer_vk: %s, doing the frame loop; draws are not translated yet", g.context.props.deviceName);
    g.shaders.cacheDir = "out/vkcache";
    if (!vk::TextureCacheInit(g.context, g.textures, 1024))
    {
        g.failed = true;
        return false;
    }
    VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    info.magFilter = info.minFilter = VK_FILTER_LINEAR;
    info.addressModeU = info.addressModeV = info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.maxLod = 16;
    for (auto& sampler : g.samplers)
        vkCreateSampler(g.context.device, &info, nullptr, &sampler);

    // Bindless index 0 is the null texture: white, so a draw whose textures are not captured yet
    // reads white instead of an unbound descriptor.
    const uint8_t white[4] = { 255, 255, 255, 255 };
    if (vk::TextureAdd(g.context, g.textures, 1, 1, white) != 0)
    {
        Log("renderer_vk: could not create the null texture");
        g.failed = true;
        return false;
    }
    if (!vk::DrawInit(g.context, g.draw, VK_NULL_HANDLE, g.samplers, g.textures.set, VK_NULL_HANDLE, 16u << 20, 64u << 20))
    {
        Log("renderer_vk: draw context failed");
        g.failed = true;
        return false;
    }
    VkCommandBufferAllocateInfo allocate{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocate.commandPool = g.context.pool;
    allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g.context.device, &allocate, &g.cmd) != VK_SUCCESS)
    {
        g.failed = true;
        return false;
    }
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &beginInfo);
    return true;
}

void EndFrame()
{
    if (!g.cmd)
        return;
    vkEndCommandBuffer(g.cmd);
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g.cmd;
    if (vkQueueSubmit(g.context.queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS)
        vkQueueWaitIdle(g.context.queue);
    vkResetCommandBuffer(g.cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(g.cmd, &beginInfo);
}
}

// ---- Shaders ----------------------------------------------------------------------------
void RendererVkOnShaderLoad(uint32_t type, const uint32_t* microcodeBE, uint32_t dwords)
{
    if (!Init())
        return;
    std::vector<uint32_t>& code = type == 0 ? g.vsCode : g.psCode;
    code.assign(microcodeBE, microcodeBE + dwords);
    uint64_t hash = 0xCBF29CE484222325ull;
    for (uint32_t w : code)
        hash = (hash ^ w) * 0x100000001B3ull;
    hash ^= type ? 1u : 0u;
    (type == 0 ? g.vsHash : g.psHash) = hash;
}

void RendererVkSetShader(uint32_t type, const uint32_t* microcodeBE, uint32_t usedDwords, uint64_t hash)
{
    std::vector<uint32_t>& code = type == 0 ? g.vsCode : g.psCode;
    code.assign(microcodeBE, microcodeBE + usedDwords);
    (type == 0 ? g.vsHash : g.psHash) = hash;
}

// ---- Draws ------------------------------------------------------------------------------
void RendererVkOnDraw(uint32_t initiator, uint32_t dmaBase, uint32_t dmaSize)
{
    // The draw path is behind its own switch while it is being brought up: with just
    // FM2_RENDERER=vulkan the seam and the shader service run against the real game, and draws
    // are skipped rather than drawn wrongly.
    static const bool drawEnabled = getenv("FM2_VK_DRAWS") != nullptr;
    if (!Init())
        return;
    if (!g.vsHash || !g.psHash)
        return;
    const vk::VkShader* vs = vk::ShaderGet(g.context, g.shaders, g.vsHash, g.vsCode.data(), uint32_t(g.vsCode.size()), false);
    const vk::VkShader* ps = vk::ShaderGet(g.context, g.shaders, g.psHash, g.psCode.data(), uint32_t(g.psCode.size()), true);
    if (!vs || !vs->ok || !ps || !ps->ok)
        return;
    if (!drawEnabled)
    {
        if (!g.reportedDraws)
        {
            g.reportedDraws = true;
            Log("renderer_vk: shaders compile and cache (%u compiled, %u failed); draws are behind "
                "FM2_VK_DRAWS=1 while the translation is brought up", g.shaders.compiled, g.shaders.failed);
        }
        return;
    }

    // The surface the guest renders into: EDRAM base from RB_COLOR_INFO, size from RB_SURFACE_INFO.
    const uint32_t base = gpureg::Get(0x2001) & 0xFFF;
    const uint32_t width = gpureg::Get(0x2000) & 0x3FFF, height = (gpureg::Get(0x2000) >> 16) & 0x3FFF;
    vk::Surface* surface = vk::SurfaceAcquire(g.context, g.surfaces, base, width ? width : 1280, height ? height : 720,
        VK_SAMPLE_COUNT_1_BIT);
    if (!surface)
        return;

    // Expanded index list: the vertex shader fetches it out of guest memory.
    std::vector<uint32_t> list;
    if (!gpu::BuildIndexList(initiator, dmaBase, dmaSize, list) || list.empty())
        return;

    const vk::StateKey key = MakeKey(g.vsHash, g.psHash, (initiator & 0x3F) == 4);
    VkPipeline pipeline = PipelineFor(key, vs, ps, *surface);
    if (!pipeline)
        return;

    gpu::VSConstants vc{};
    gpu::PSConstants pc{};
    BuildConstants(vs, ps, vc, pc);

    if (g.openSurface != surface)
    {
        ClosePass();
        // A surface starts cleared; the guest's predicated tile replays are the same pass.
        VkClearColorValue clear{};
        clear.float32[3] = 1.0f;
        vk::SurfaceBegin(g.context, *surface, g.cmd, &clear);
        g.passOpen = true;
        g.openSurface = surface;
    }
    vk::DrawList(g.context, g.draw, g.cmd, pipeline, g.textures.set, vc, pc, list.data(), uint32_t(list.size()));
    g.drawsThisFrame++;
}

void RendererVkOnSwap(uint64_t frame, uint32_t frontBuffer)
{
    (void)frontBuffer;
    if (!Init())
        return;
    ClosePass();
    // Submit the frame and start the next one. The command buffer is reused: waiting keeps it
    // simple, and the first frames are not where the time goes.
    EndFrame();
    g.frame = frame;
    g.drawsThisFrame = 0;
    vk::DrawBeginFrame(g.draw);
    if (frame % 60 == 0)
        Log("renderer_vk: frame %llu, %u draws, %u shaders compiled, %zu surfaces", (unsigned long long)frame,
            g.drawsThisFrame, g.shaders.compiled, g.surfaces.surfaces.size());
}
