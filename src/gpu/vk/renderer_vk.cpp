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
#include "gpu/regs.h"
#include "gpu/renderer.h"
#include "gpu/vk/vk_context.h"
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
    VkSampler samplers[4]{};
    VkBuffer gmem = VK_NULL_HANDLE, gphys = VK_NULL_HANDLE;
    std::unordered_map<vk::StateKey, VkPipeline, StateKeyHash> pipelines;

    uint64_t vsHash = 0, psHash = 0;
    std::vector<uint32_t> vsCode, psCode;
    uint64_t frame = 0;
    bool reportedDraws = false;
    uint32_t surfacesSeen = 0;
} g;
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
    return true;
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
    (void)initiator;
    (void)dmaBase;
    (void)dmaSize;
    if (!Init())
        return;
    // The shaders still go through the backend's service, so a session with draws selected
    // proves translation and compilation (and fills the SPIR-V cache) before the draw path
    // exists to use them.
    if (g.vsHash && g.psHash)
    {
        vk::ShaderGet(g.context, g.shaders, g.vsHash, g.vsCode.data(), uint32_t(g.vsCode.size()), false);
        vk::ShaderGet(g.context, g.shaders, g.psHash, g.psCode.data(), uint32_t(g.psCode.size()), true);
    }
    if (!g.reportedDraws)
    {
        g.reportedDraws = true;
        Log("renderer_vk: draws are not translated yet; shaders are compiled and cached "
            "(%u compiled, %u failed)", g.shaders.compiled, g.shaders.failed);
    }
}

void RendererVkOnSwap(uint64_t frame, uint32_t frontBuffer)
{
    (void)frontBuffer;
    if (!Init())
        return;
    // The guest finished a frame: the surfaces it rendered into are still just surfaces. Log
    // the shape of the frame once a second so the surface model can be checked against what
    // the command processor is being asked to do.
    g.frame = frame;
    if (frame % 60 == 0)
        Log("renderer_vk: frame %llu, %u shaders compiled, %zu surfaces", (unsigned long long)frame, g.shaders.compiled,
            g.surfaces.surfaces.size());
}
