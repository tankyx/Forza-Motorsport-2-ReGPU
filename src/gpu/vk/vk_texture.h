#pragma once
// Bindless texture cache: the Vulkan counterpart of the renderer's SRV heap. Game textures are
// decoded on the CPU (texture.cpp), uploaded once, and kept in a descriptor array the shaders
// index with the fetch slot's index — the same index the D3D12 path writes into its heap.
//
// The array is one descriptor set (set 1 for 2D, set 2 for cube in the renderer's layout),
// variable-sized and partially bound, with UPDATE_AFTER_BIND so a texture can be added while
// commands that only use earlier slots are in flight.
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

#include "gpu/vk/vk_context.h"

namespace vk
{
struct TextureCache
{
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::vector<Image> slots;      // the image behind each bindless index
    std::vector<uint32_t> freeSlots;
    uint32_t capacity = 0;
    uint32_t next = 0; // the next untouched slot
};

// capacity is the number of bindless slots (the renderer's heap is kBindless entries).
bool TextureCacheInit(Context& context, TextureCache& cache, uint32_t capacity);
void TextureCacheShutdown(Context& context, TextureCache& cache);

// Uploads RGBA8 (tightly packed, top row first) and returns its bindless index, or UINT32_MAX
// if the cache is full. Reuses a freed slot when there is one.
uint32_t TextureAdd(Context& context, TextureCache& cache, uint32_t width, uint32_t height, const void* rgba,
    VkImageUsageFlags extraUsage = 0);
// Reserves an index whose image is drawn into rather than uploaded (the renderer's resolves).
// The image is created with the given usage and left in COLOR_ATTACHMENT_OPTIMAL.
uint32_t TextureReserve(Context& context, TextureCache& cache, uint32_t width, uint32_t height);
void TextureFree(Context& context, TextureCache& cache, uint32_t index);
// Replaces the image in an existing slot (the renderer reuses an index when a texture's
// contents change), so no descriptor has to be rewritten.
bool TextureReplace(Context& context, TextureCache& cache, uint32_t index, uint32_t width, uint32_t height, const void* rgba);
}
