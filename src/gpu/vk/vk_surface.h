#pragma once
// EDRAM surfaces: the guest renders into a 10 MB tile buffer at addresses the CPU knows
// nothing about, so the renderer maps each EDRAM base it sees to a pair of host images
// (colour and depth) and treats the tile replays as what they are — the same pass, rendered
// once at full resolution. This is the Vulkan side of that model: one surface per base, kept
// across frames like the D3D12 path does, with the image layouts tracked so nothing has to
// guess when a barrier is needed.
#include <cstdint>
#include <unordered_map>
#include <vulkan/vulkan.h>

#include "gpu/vk/vk_context.h"

namespace vk
{
struct Surface
{
    uint32_t base = 0;      // the EDRAM address this surface stands for
    uint32_t width = 0, height = 0;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
    Image color;
    Image depth;
    VkImageLayout colorLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout depthLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkRenderPass pass = VK_NULL_HANDLE; // matches format and sample count
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
};

struct SurfaceCache
{
    std::unordered_map<uint32_t, Surface> surfaces;
};

void SurfaceCacheDestroy(Context& context, SurfaceCache& cache);

// The surface for an EDRAM base, created (or resized/reformatted) on first use. nullptr if the
// images could not be made.
Surface* SurfaceAcquire(Context& context, SurfaceCache& cache, uint32_t base, uint32_t width, uint32_t height,
    VkSampleCountFlagBits samples);

// Opens a render pass on the surface: barriers the colour and depth images into attachment
// layouts. clearColor null keeps what is there (load), otherwise the colour is cleared.
void SurfaceBegin(Context& context, Surface& surface, VkCommandBuffer cmd, const VkClearColorValue* clearColor);

// Closes the render pass and moves the colour image to TRANSFER_SRC so it can be resolved or
// copied; call SurfaceShaderRead instead when a later pass samples it.
void SurfaceEnd(Context& context, Surface& surface, VkCommandBuffer cmd);
void SurfaceShaderRead(Context& context, Surface& surface, VkCommandBuffer cmd);

// Resolves the surface (MSAA to one sample) or copies it when it is already single sampled,
// recording into `cmd` and leaving `destination` in TRANSFER_SRC. `destination` must be a fresh
// image (UNDEFINED layout) of the surface's size.
bool SurfaceResolve(Context& context, Surface& surface, VkCommandBuffer cmd, VkImage destination, uint32_t width,
    uint32_t height);
}
