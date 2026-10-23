#pragma once
// EDRAM resolves: the guest copies a render target into memory (RB_MODECONTROL 6), and later
// samples the result at that address. The renderer never writes guest memory, so the copy goes
// into a texture instead — sampled directly by whatever fetch asks for that address.
//
// The source is a render target (half-float colour here), the destination the RGBA8 image the
// bindless cache hands out, so this is a small full-screen pass that also applies the copy's
// channel swap.
#include <vulkan/vulkan.h>

#include "gpu/vk/vk_context.h"

namespace vk
{
struct Resolve
{
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkRenderPass pass = VK_NULL_HANDLE;
    VkPipeline copy = VK_NULL_HANDLE;
    VkPipeline swap = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
};

bool ResolveInit(Context& context, Resolve& resolve);
void ResolveShutdown(Context& context, Resolve& resolve);

// Copies `source` (any sampled format) into `destination` (RGBA8, in UNDEFINED layout), leaving
// it in SHADER_READ_ONLY. Recorded into the caller's command buffer, inside no render pass.
bool ResolveImage(const Context& context, Resolve& resolve, VkCommandBuffer cmd, VkImageView source, VkImage destination,
    uint32_t width, uint32_t height, bool swapRedBlue);
}
