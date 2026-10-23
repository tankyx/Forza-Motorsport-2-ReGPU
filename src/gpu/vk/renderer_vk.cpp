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
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "gpu/constants.h"
#include "gpu/indices.h"
#include "gpu/texture.h"
#include "gpu/regs.h"
#include "gpu/renderer.h"
#include "gpu/vk/vk_context.h"
#include "gpu/vk/vk_draw.h"
#include "gpu/vk/vk_shader.h"
#include "gpu/vk/vk_shaders.h"
#include "gpu/vk/vk_state.h"
#include "gpu/vk/vk_surface.h"
#include "gpu/vk/vk_texture.h"
#include "memory.h"
#include "png.h"
#include "render/presenter.h"
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
    vk::TextureCache textures0; // the bindless uploader
    vk::DrawContext draw;
    VkSampler samplers[4]{};
    VkBuffer gmem = VK_NULL_HANDLE, gphys = VK_NULL_HANDLE;
    std::unordered_map<vk::StateKey, VkPipeline, StateKeyHash> pipelines;

    uint64_t vsHash = 0, psHash = 0;
    VkShaderModule flatModule = VK_NULL_HANDLE; // the FM2_VK_TESTPS shader, when used

    // Decoded textures, by fetch identity: the index the shader samples, and a hash of the
    // guest data so a rewritten texture is re-uploaded instead of being served stale.
    struct CachedTexture
    {
        uint32_t index = 0;
        uint64_t hash = 0;
    };
    std::unordered_map<uint64_t, CachedTexture> textures;
    std::vector<uint32_t> vsCode, psCode;
    uint64_t frame = 0;
    bool reportedDraws = false;
    uint32_t surfacesSeen = 0;
    uint32_t skippedShader = 0, skippedSurface = 0, skippedIndex = 0, skippedPipeline = 0, skippedDraw = 0;
    uint32_t texBound = 0, texFailed = 0, texSlots = 0;
    // Placement state per draw, histogrammed: the frame's dominant viewport is what matters, not
    // whichever draw happened to be last.
    std::unordered_map<uint64_t, uint32_t> viewports;
    uint32_t viewportTarget = 0;

    // The frame: one command buffer, one render pass per surface (the D3D12 path does the same
    // — the guest's tile replays become one pass), submitted when the guest swaps.
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vk::Surface* openSurface = nullptr;
    bool passOpen = false;
    uint64_t drawsThisFrame = 0;
} g;

float RegFloat(uint32_t index)
{
    const uint32_t word = gpureg::Get(index);
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
// Writes a surface out as a PNG (out/vk_frame.png): the first thing to look at when the draw
// path is being brought up.
void DumpSurface(vk::Surface& surface)
{
    const VkDeviceSize bytes = VkDeviceSize(surface.width) * surface.height * 8; // RGBA16F
    void* readback = nullptr;
    VkBuffer copy = vk::CreateBuffer(g.context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback);
    if (!copy)
        return;
    // Resolve through the path the surface test verified (it handles the layouts), then read.
    vk::Image resolved = vk::CreateImage(g.context, surface.width, surface.height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const bool ok = resolved.image && vk::Submit(g.context, [&](VkCommandBuffer cmd) {
        vk::SurfaceResolve(g.context, surface, cmd, resolved.image, surface.width, surface.height);
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { surface.width, surface.height, 1 };
        vkCmdCopyImageToBuffer(cmd, resolved.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy, 1, &region);
    });
    if (ok)
    {
        auto half = [](uint16_t h) {
            const uint32_t sign = uint32_t(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits = 0;
            if (exp == 0)
                bits = sign;
            else if (exp == 31)
                bits = sign | 0x7F800000u | (man << 13);
            else
                bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
            float f;
            memcpy(&f, &bits, 4);
            return f;
        };
        const uint16_t* src = static_cast<const uint16_t*>(readback);
        std::vector<uint8_t> rgba(size_t(surface.width) * surface.height * 4);
        size_t lit = 0;
        for (size_t i = 0; i < size_t(surface.width) * surface.height; i++)
        {
            for (int c = 0; c < 3; c++)
            {
                const float v = std::min(1.0f, std::max(0.0f, half(src[i * 4 + c])));
                rgba[i * 4 + c] = uint8_t(v * 255.0f);
                lit += v > 0.02f;
            }
            rgba[i * 4 + 3] = 255;
        }
        char name[64];
        snprintf(name, sizeof(name), "out/vk_frame_%llu.png", (unsigned long long)g.frame);
        WritePng(name, surface.width, surface.height, rgba);
        Log("renderer_vk: wrote %s (%ux%u, %zu lit components)", name, surface.width, surface.height, lit);
    }
    vkDeviceWaitIdle(g.context.device);
    vk::DestroyImage(g.context, resolved);
    vkDestroyBuffer(g.context.device, copy, nullptr);
}

namespace
{
const char kTestVs[] = R"(
struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};
VSOut main(uint id : SV_VertexID)
{
    float2 p = float2((id << 1) & 2, id & 2);
    VSOut o;
    o.pos = float4(p * 2 - 1, 0, 1);
    o.uv = float4(p, 0, 1);
    return o;
}
)";

const char kTestPs[] = R"(
struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};
float4 main(VSOut input) : SV_Target0
{
    return float4(0.9, 0.2, 0.6, 1); // flat magenta: nothing to get wrong
}
)";

VkShaderModule MakeTestModule(const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(g.context.device, &info, nullptr, &module) == VK_SUCCESS ? module : VK_NULL_HANDLE;
}

// The flat magenta pixel shader, compiled once (FM2_VK_TESTPS).
const vk::VkShader* FlatPixelShader()
{
    static vk::VkShader shader;
    static bool tried = false;
    if (!tried)
    {
        tried = true;
        std::vector<uint32_t> code;
        std::string error;
        if (vk::CompileHlsl(kTestPs, true, vk::HashSource(kTestPs), "out/vkcache", code, &error))
        {
            shader.module = MakeTestModule(code);
            shader.ok = shader.module != VK_NULL_HANDLE;
            shader.pixel = true;
            g.flatModule = shader.module;
        }
        else
            Log("renderer_vk: flat pixel shader: %s", error.c_str());
    }
    return shader.ok ? &shader : nullptr;
}

// FM2_VK_TESTDRAW=1: one extra pass into the surface with a magenta full-screen triangle.
void TestDraw(vk::Surface& surface)
{
    static VkPipeline pipeline = VK_NULL_HANDLE;
    Log("renderer_vk: test draw: begin (pipeline %p)", (void*)pipeline);
    if (!pipeline)
    {
        std::vector<uint32_t> vsCode, psCode;
        std::string error;
        if (!vk::CompileHlsl(kTestVs, false, vk::HashSource(kTestVs), "out/vkcache", vsCode, &error) ||
            !vk::CompileHlsl(kTestPs, true, vk::HashSource(kTestPs), "out/vkcache", psCode, &error))
        {
            Log("renderer_vk: test shaders: %s", error.c_str());
            return;
        }
        Log("renderer_vk: test draw: shaders compiled");
        VkShaderModule vs = MakeTestModule(vsCode), ps = MakeTestModule(psCode);
        vk::StateKey key;
        key.blend = 0x00010001;           // opaque
        key.stencilMaskMode = 0x0F << 24; // all channels written
        vk::TargetFormats targets;
        targets.samples = surface.samples;
        if (!vk::CreatePipeline(g.context, key, vs, ps, g.draw.layout, surface.pass, targets, pipeline))
        {
            Log("renderer_vk: test pipeline failed");
            return;
        }
        vkDestroyShaderModule(g.context.device, vs, nullptr);
        vkDestroyShaderModule(g.context.device, ps, nullptr);
        Log("renderer_vk: test draw: pipeline %p", (void*)pipeline);
    }
    VkClearColorValue clear{};
    clear.float32[0] = 0.05f;
    clear.float32[1] = 0.10f;
    clear.float32[2] = 0.25f;
    clear.float32[3] = 1.0f;
    vk::SurfaceBegin(g.context, surface, g.cmd, &clear);
    Log("renderer_vk: test draw: pass open");
    vkCmdBindPipeline(g.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkDescriptorSet sets[3] = { g.draw.set0, g.textures0.set, g.textures0.set };
    const uint32_t offsets[2] = { 0, 0 };
    
    vkCmdDraw(g.cmd, 3, 1, 0, 0);
    vk::SurfaceEnd(g.context, surface, g.cmd);
    Log("renderer_vk: test draw: recorded");
}
}

void ClosePass()
{
    if (!g.passOpen)
        return;
    vk::SurfaceEnd(g.context, *g.openSurface, g.cmd);
    g.passOpen = false;
    g.openSurface = nullptr;
}

// The constant blocks, filled the way the D3D12 path fills them (renderer.cpp: DrawImpl).
// FNV over the guest bytes a texture covers, sampled: enough to notice the game rewriting one.
uint64_t GuestTextureHash(const TextureFetch& fetch)
{
    const size_t bytes = std::min<size_t>(TextureSourceSize(fetch), 4u << 20);
    if (!bytes)
        return 0;
    const uint8_t* source = GuestPtr<uint8_t>(GpuPhysToGuest(fetch.Address()));
    uint64_t hash = 0xCBF29CE484222325ull;
    const size_t step = std::max<size_t>(4, bytes / 512 / 4 * 4);
    for (size_t i = 0; i < bytes; i += step)
    {
        hash = (hash ^ source[i]) * 0x100000001B3ull;
    }
    return hash ^ bytes;
}

// One fetch slot: decode on the CPU (every format the decoder knows) and hand the shader a
// bindless index. Cube maps and unsupported formats keep the null texture for now.
void BindTexture(gpu::PSConstants& pc, uint32_t slot)
{
    g.texSlots++;
    const TextureFetch fetch = ReadTextureFetch(slot);
    if (fetch.Type() != 2 || fetch.Dimension() == 3 || !TextureFormatSupported(fetch.Format()))
    {
        g.texFailed++;
        static std::unordered_map<uint32_t, uint32_t> byFormat;
        if (byFormat[fetch.Format()]++ < 3)
            Log("texture_vk: refused format %u (%s) type %u dimension %u", fetch.Format(), TextureFormatName(fetch.Format()), fetch.Type(),
                fetch.Dimension());
        return;
    }
    // Cache first: decoding is milliseconds (tiling, format conversion) and a draw samples the
    // same textures every frame, so only a new or rewritten one goes through the decoder.
    const uint64_t key = uint64_t(fetch.Address()) << 24 ^ uint64_t(fetch.Format()) << 16 ^ uint64_t(fetch.Width()) ^
        uint64_t(fetch.Height()) << 40 ^ uint64_t(fetch.Swizzle()) << 8;
    const uint64_t hash = GuestTextureHash(fetch);
    uint32_t index = 0;
    auto cached = g.textures.find(key);
    if (cached != g.textures.end() && cached->second.hash == hash)
    {
        index = cached->second.index;
    }
    else
    {
        std::vector<uint8_t> rgba;
        if (!DecodeTexture(fetch, rgba))
            return;
        if (cached != g.textures.end())
        {
            index = cached->second.index;
            vk::TextureReplace(g.context, g.textures0, index, fetch.Width(), fetch.Height(), rgba.data());
            cached->second.hash = hash;
        }
        else
        {
            index = vk::TextureAdd(g.context, g.textures0, fetch.Width(), fetch.Height(), rgba.data());
            if (index == UINT32_MAX)
                return;
            g.textures[key] = { index, hash };
        }
    }

    // The tables the pixel shader reads (see the D3D12 path's fill in renderer.cpp).
    const uint32_t clampX = (fetch.dwords[0] >> 10) & 7, clampY = (fetch.dwords[0] >> 13) & 7;
    const bool clamp = clampX != 0 && clampY != 0;
    const bool point = ((fetch.dwords[3] >> 19) & 3) == 0;
    pc.tinfo[slot][0] = (point ? (clamp ? 2u : 3u) : (clamp ? 1u : 0u)) | ((uint32_t(fetch.ExpAdjust()) & 0x3F) << 8);
    if (!clamp)
        pc.tinfo[slot][0] |= (clampX ? 1u << 18 : 0u) | (clampY ? 1u << 19 : 0u);
    if (clampX >= 6 && clampY >= 6)
        pc.tinfo[slot][0] |= (1u << 16) | ((fetch.dwords[5] & 3) == 1 ? 1u << 17 : 0u);
    pc.tinfo[slot][1] = fetch.Width();
    pc.tinfo[slot][2] = fetch.Height();
    pc.tinfo[slot][3] = 1; // 2D, swizzled on the CPU while decoding
    pc.tbias[slot][0] = fetch.LodBias();
    memcpy(&pc.tbias[slot][1], &index, 4);
    pc.tbias[slot][2] = pc.tbias[slot][3] = 1.0f;
    g.texBound++;
}

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
    // The target size and the point-sprite sizes, as the D3D12 path sets them: 2D draws and
    // point sprites are placed from these, not from the viewport.
    uint32_t targetWidth = 0, targetHeight = 0;
    RenderSize(targetWidth, targetHeight);
    vc.info[2] = targetWidth;
    vc.info[3] = targetHeight;
    const uint32_t pointSize = gpureg::Get(0x2085 /* PA_SU_POINT_SIZE */), pointMinMax = gpureg::Get(0x2089 /* PA_SU_POINT_MINMAX */);
    vc.point[0] = float(pointSize >> 16) / 8.0f;
    vc.point[1] = float(pointSize & 0xFFFF) / 8.0f;
    vc.point[2] = float(pointMinMax & 0xFFFF) / 8.0f;
    vc.point[3] = float(pointMinMax >> 16) / 8.0f;
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
    // Every fetch slot either shader samples: decode, upload, and put the bindless index in
    // the tables the shader reads.
    for (uint32_t slot = 0; slot < 32; slot++)
        if ((ps->usedTextureSlots | vs->usedTextureSlots) & (1u << slot))
            BindTexture(pc, slot);
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
    if (!vk::TextureCacheInit(g.context, g.textures0, 1024))
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
    if (vk::TextureAdd(g.context, g.textures0, 1, 1, white) != 0)
    {
        Log("renderer_vk: could not create the null texture");
        g.failed = true;
        return false;
    }
    // The guest's physical memory window, wrapped rather than copied: the vertex shaders read
    // vertex data straight out of it (gpu/memory.h maps the guest's 4 GB at g_base).
    g.gphys = vk::CreateHostImportedBuffer(g.context, GuestPtr<uint8_t>(0xA0000000), 0x20000000);
    if (!g.gphys)
        Log("renderer_vk: guest memory is not importable; vertex data will read as zero");
    if (!vk::DrawInit(g.context, g.draw, g.gphys, g.samplers, g.textures0.set, VK_NULL_HANDLE, 16u << 20, 64u << 20))
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
    const VkResult ended = vkEndCommandBuffer(g.cmd);
    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &g.cmd;
    const VkResult submitted = ended == VK_SUCCESS ? vkQueueSubmit(g.context.queue, 1, &submit, VK_NULL_HANDLE) : ended;
    const VkResult waited = submitted == VK_SUCCESS ? vkQueueWaitIdle(g.context.queue) : submitted;
    static int reported = 0;
    if ((submitted != VK_SUCCESS || waited != VK_SUCCESS) && reported++ < 3)
        Log("renderer_vk: frame submit end %d, submit %d, wait %d", int(ended), int(submitted), int(waited));
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
    // Resolve packets (RB_MODECONTROL 6) copy EDRAM into memory: they are not geometry and must
    // not be drawn as triangles. The D3D12 path turns them into a texture the title samples
    // later (BlitResolve / EnsureResolved); that is the next piece here.
    if ((gpureg::Get(0x2208 /* RB_MODECONTROL */) & 7) == 6)
    {
        static uint32_t resolves = 0;
        if (resolves < 6)
        {
            resolves++;
            Log("resolve_vk: base %08x pitch %u height %u destInfo %08x source %u (initiator %08x, %u vertices listed)",
                gpureg::Get(0x2319 /* RB_COPY_DEST_BASE */) & 0x1FFFFFFF, gpureg::Get(0x231A /* RB_COPY_DEST_PITCH */) & 0x3FFF,
                (gpureg::Get(0x231A) >> 16) & 0x3FFF, gpureg::Get(0x231B /* RB_COPY_DEST_INFO */),
                (gpureg::Get(0x2318 /* RB_COPY_CONTROL */) >> 4) & 7, initiator, initiator >> 16);
        }
        return;
    }
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
    {
        if (g.skippedShader++ < 4)
            Log("renderer_vk: shader not ready (vs %d/%s, ps %d/%s)", vs ? int(vs->ok) : -1, vs ? vs->error.c_str() : "?",
                ps ? int(ps->ok) : -1, ps ? ps->error.c_str() : "?");
        return;
    }
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

    // The surface the guest renders into: EDRAM base from RB_COLOR_INFO, sized like the D3D12
    // path sizes its surfaces — the render resolution from the settings, not the register's
    // pitch/height fields (those describe the tiled pass, not the host target).
    const uint32_t base = gpureg::Get(0x2001) & 0xFFF;
    uint32_t width = 0, height = 0;
    RenderSize(width, height);
    vk::Surface* surface = vk::SurfaceAcquire(g.context, g.surfaces, base, width, height, VK_SAMPLE_COUNT_1_BIT);
    if (!surface)
    {
        if (g.skippedSurface++ < 4)
            Log("renderer_vk: no surface for base %u (%ux%u)", base, width, height);
        return;
    }

    // Expanded index list: the vertex shader fetches it out of guest memory.
    std::vector<uint32_t> list;
    if (!gpu::BuildIndexList(initiator, dmaBase, dmaSize, list) || list.empty())
    {
        if (g.skippedIndex++ < 4)
            Log("renderer_vk: no index list (initiator %08x prim %u source %u count %u dma %08x/%08x, %zu entries)",
                initiator, initiator & 0x3F, (initiator >> 6) & 3, initiator >> 16, dmaBase, dmaSize, list.size());
        return;
    }

    static const bool noCull = getenv("FM2_VK_NOCULL") != nullptr;
    // FM2_VK_TESTPS=1: the guest's vertex shader with a flat magenta pixel shader. If shapes
    // appear, the vertex path (fetch, layout, constants) is right and the pixel side is not.
    if (getenv("FM2_VK_TESTPS"))
        if (const vk::VkShader* flat = FlatPixelShader())
            ps = flat;
    const vk::StateKey key = MakeKey(g.vsHash, ps->module == g.flatModule ? 0 : g.psHash, !noCull && (initiator & 0x3F) == 4);
    VkPipeline pipeline = PipelineFor(key, vs, ps, *surface);
    if (!pipeline)
    {
        if (g.skippedPipeline++ < 4)
            Log("renderer_vk: no pipeline (blend %08x depth %08x mask %08x flags %u)", key.blend, key.depth, key.stencilMaskMode,
                key.flags);
        return;
    }

    gpu::VSConstants vc{};
    gpu::PSConstants pc{};
    BuildConstants(vs, ps, vc, pc);

    if (g.openSurface != surface)
    {
        ClosePass();
        // A surface starts cleared; the guest's predicated tile replays are the same pass.
        // Not black, so a cleared frame can be told apart from one that never ran.
        VkClearColorValue clear{};
        clear.float32[0] = 0.05f;
        clear.float32[1] = 0.10f;
        clear.float32[2] = 0.25f;
        clear.float32[3] = 1.0f;
        vk::SurfaceBegin(g.context, *surface, g.cmd, &clear);
        g.passOpen = true;
        g.openSurface = surface;
    }
    // One-shot look at one draw: what the vertices and the fetch table actually say.
    static uint32_t seen = 0;
    static bool dumped = false;
    if (seen++ == (getenv("FM2_VK_SAMPLE") ? uint32_t(atoi(getenv("FM2_VK_SAMPLE"))) : 0u) && !dumped)
    {
        dumped = true;
        Log("renderer_vk: first draw: initiator %08x prim %u count %u, %zu vertices", initiator, initiator & 0x3F,
            initiator >> 16, list.size());
        Log("renderer_vk:   indices %u %u %u %u %u %u", list[0], list.size() > 1 ? list[1] : 0, list.size() > 2 ? list[2] : 0,
            list.size() > 3 ? list[3] : 0, list.size() > 4 ? list[4] : 0, list.size() > 5 ? list[5] : 0);
        for (uint32_t k : vs->vertexFetchConstants)
            Log("renderer_vk:   vf%u: offset %08X endian %u source %u", k, vc.vf[k][0], vc.vf[k][1], vc.vf[k][2]);
        Log("renderer_vk:   vfl0 %08x %08x %08x %08x", vc.vfl[0][0], vc.vfl[0][1], vc.vfl[0][2], vc.vfl[0][3]);
        Log("renderer_vk:   info %08x %08x %08x %08x, vte %08x, scale %g %g %g, offset %g %g %g, point %g %g", vc.info[1], vc.info[2],
            vc.info[3], vc.info[0], vc.vte[0], vc.vscale[0], vc.vscale[1], vc.vscale[2], vc.voffset[0], vc.voffset[1],
            vc.voffset[2], vc.point[0], vc.point[1]);
        Log("renderer_vk:   vscale %g %g %g offset %g %g %g, target %u %u, vinfo %08x %08x %08x %08x", vc.vscale[0],
            vc.vscale[1], vc.vscale[2], vc.voffset[0], vc.voffset[1], vc.voffset[2], vc.info[2], vc.info[3], vc.info[0],
            vc.info[1], vc.info[2], vc.info[3]);
        Log("renderer_vk:   surface %ux%u base %u, shaders vs %016llx ps %016llx", width, height, base,
            (unsigned long long)g.vsHash, (unsigned long long)g.psHash);
    }
    {
        // FM2_VK_LOG_DRAWS=<n>: the first n draws of a frame, with what places them.
        static const uint32_t logDraws = getenv("FM2_VK_LOG_DRAWS") ? uint32_t(atoi(getenv("FM2_VK_LOG_DRAWS"))) : 0;
        static uint32_t logged = 0;
        if (logDraws && (initiator & 0x3F) == 4 && g.frame >= 900 && g.frame < 960 && logged++ < logDraws)
            Log("draw_vk: frame %llu prim %u list %zu, scale %g %g offset %g %g vte %08x target %u %u slots %08x", (unsigned long long)g.frame,
                initiator & 0x3F, list.size(), vc.vscale[0], vc.vscale[1], vc.voffset[0], vc.voffset[1], vc.vte[0], vc.info[2],
                vc.info[3], ps->usedTextureSlots);
    }
    {
        // Which viewport the frame is actually drawn with (the dump reports the top ones).
        const uint32_t scaleX = uint32_t(int32_t(vc.vscale[0])), scaleY = uint32_t(int32_t(vc.vscale[1]));
        g.viewports[(uint64_t(scaleX) << 32) | uint64_t(scaleY)]++;
        g.viewportTarget = vc.info[2];
    }
    if (!vk::DrawList(g.context, g.draw, g.cmd, pipeline, g.textures0.set, vc, pc, list.data(), uint32_t(list.size())))
    {
        if (g.skippedDraw++ < 4)
            Log("renderer_vk: DrawList refused (%zu vertices)", list.size());
        return;
    }
    g.drawsThisFrame++;
}

void RendererVkOnSwap(uint64_t frame, uint32_t frontBuffer)
{
    (void)frontBuffer;
    if (!Init())
        return;
    const uint32_t draws = g.drawsThisFrame;
    vk::Surface* last = g.openSurface;
    ClosePass();
    // Submit the frame and start the next one. The command buffer is reused: waiting keeps it
    // simple, and the first frames are not where the time goes.
    EndFrame();
    if (last && frame % 500 == 0 && draws)
    {
        static const bool test = getenv("FM2_VK_TESTDRAW") != nullptr;
        if (test)
        {
            TestDraw(*last); // records into the same command buffer; EndFrame below submits it
        }
        EndFrame();
        {
            std::vector<std::pair<uint32_t, uint64_t>> top(g.viewports.begin(), g.viewports.end());
            std::sort(top.begin(), top.end(), [](auto& a, auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < std::min<size_t>(top.size(), 3); i++)
                Log("renderer_vk: dumped frame %llu: viewport scale %d %d, %u draws (target %u)",
                    (unsigned long long)frame, int32_t(top[i].first >> 32), int32_t(top[i].first), top[i].second, g.viewportTarget);
        }
        DumpSurface(*last);
    }
    g.frame = frame;
    vk::DrawBeginFrame(g.draw);
    if (frame % 60 == 0)
        Log("renderer_vk: frame %llu, %u draws, %u shaders compiled, %zu surfaces, %u pipelines, textures bound %u, "
            "refused %u of %u slots",
            (unsigned long long)frame, draws, g.shaders.compiled, g.surfaces.surfaces.size(), unsigned(g.pipelines.size()),
            g.texBound, g.texFailed, g.texSlots);
    g.drawsThisFrame = 0;
    g.texBound = g.texFailed = g.texSlots = 0;
    g.viewports.clear();
}
