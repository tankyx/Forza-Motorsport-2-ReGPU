#pragma once
// Per-draw submission: the constant blocks, the index list and the bind chain, the Vulkan
// counterpart of what DrawImpl does on Direct3D 12.
//
// Constants go into a ring that is rewritten every draw and bound with dynamic offsets (one
// set, no descriptor writes per draw); the expanded index list goes into another ring and is
// bound as a 32-bit index buffer. Nothing here knows about EDRAM or render targets: the caller
// opens the render pass and binds the pipeline.
#include <cstdint>
#include <vulkan/vulkan.h>

#include "gpu/constants.h"
#include "gpu/vk/vk_context.h"
#include "gpu/vk/vk_state.h"

namespace vk
{
struct DrawRing
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* cpu = nullptr;
    VkDeviceSize size = 0, used = 0;
};

struct DrawContext
{
    VkPipelineLayout layout = VK_NULL_HANDLE;
    DrawRing constants;   // VSConstants then PSConstants, per draw
    DrawRing indices;     // expanded index lists, for vkCmdDrawIndexed
    DrawRing guestMemory; // the expanded index list the vertex shader reads (gmem)
    VkDescriptorSet set0 = VK_NULL_HANDLE;      // constants (dynamic), guest memory, samplers
    VkDescriptorSet bindless2D = VK_NULL_HANDLE; // set 1
    VkDescriptorSet bindlessCube = VK_NULL_HANDLE; // set 2
    VkDescriptorPool pool = VK_NULL_HANDLE;       // owns set0
    VkDescriptorBufferInfo constantsInfo{};     // both dynamic offsets point into this buffer
};

// constantsBytes should hold at least a few hundred draws' worth of both blocks (they are
// 6352 and 5184 bytes); indexBytes a frame's expanded lists (the D3D12 path uses 64 MB).
bool DrawInit(Context& context, DrawContext& draw, VkBuffer gphys, VkSampler samplers[4], VkDescriptorSet bindless2D,
    VkDescriptorSet bindlessCube, VkDeviceSize constantsBytes, VkDeviceSize listBytes);
void DrawDestroy(Context& context, DrawContext& draw);

// Rewinds both rings: called at the start of a frame.
void DrawBeginFrame(DrawContext& draw);

// Records one draw. Returns false when a ring is full (the caller can flush and retry).
// indexed false means the vertex shader pulls the index stream itself (the renderer's
// "direct" case: triangle lists and restart-free strips read straight from guest memory).
bool DrawIndexed(const Context& context, DrawContext& draw, VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet textures,
    const gpu::VSConstants& vs, const gpu::PSConstants& ps, const uint32_t* indices, uint32_t count, bool indexed);

// Records one draw the way the translated shaders expect: the expanded index list is appended
// to the guest-memory ring and its offset goes into vc.info.x, then the vertices are drawn
// non-indexed — the vertex shader fetches each index out of that list itself. This is the
// renderer's normal path (the D3D12 side does exactly this with its upload ring).
bool DrawList(const Context& context, DrawContext& draw, VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet textures,
    gpu::VSConstants vs, const gpu::PSConstants& ps, const uint32_t* list, uint32_t count);

// The blend factor the guest set (constant blend factors), applied with vkCmdSetBlendConstants.
void DrawSetBlendConstants(VkCommandBuffer cmd, const float rgba[4]);
}
