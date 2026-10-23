#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_texture.h"

#include <cstdio>
#include <cstring>

namespace vk
{
bool TextureCacheInit(Context& context, TextureCache& cache, uint32_t capacity, bool cube)
{
    cache.cube = cube;
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0; // set 1, binding 0: the 2D array (dxc's register space 1)
    binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    binding.descriptorCount = capacity;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bindingFlags.bindingCount = 1;
    bindingFlags.pBindingFlags = &flags;
    VkDescriptorSetLayoutCreateInfo layout{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layout.pNext = &bindingFlags;
    layout.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout.bindingCount = 1;
    layout.pBindings = &binding;
    if (vkCreateDescriptorSetLayout(context.device, &layout, nullptr, &cache.layout) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize size{ VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, capacity };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &size;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &cache.pool) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = cache.pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &cache.layout;
    if (vkAllocateDescriptorSets(context.device, &allocate, &cache.set) != VK_SUCCESS)
        return false;

    cache.capacity = capacity;
    cache.slots.resize(capacity);
    return true;
}

void TextureCacheShutdown(Context& context, TextureCache& cache)
{
    for (Image& image : cache.slots)
        DestroyImage(context, image);
    if (cache.pool)
        vkDestroyDescriptorPool(context.device, cache.pool, nullptr);
    if (cache.layout)
        vkDestroyDescriptorSetLayout(context.device, cache.layout, nullptr);
    cache = {};
}

namespace
{
bool WriteSlot(Context& context, TextureCache& cache, uint32_t index, const Image& image)
{
    VkDescriptorImageInfo info{ VK_NULL_HANDLE, image.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    write.dstSet = cache.set;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    return true;
}
}

uint32_t TextureAdd(Context& context, TextureCache& cache, uint32_t width, uint32_t height, const void* rgba,
    VkImageUsageFlags extraUsage)
{
    uint32_t index;
    if (!cache.freeSlots.empty())
    {
        index = cache.freeSlots.back();
        cache.freeSlots.pop_back();
    }
    else if (cache.next < cache.capacity)
        index = cache.next++;
    else
        return UINT32_MAX;

    if (!cache.slots[index].image)
    {
        cache.slots[index] = CreateImage(context, width, height, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | extraUsage,
            cache.cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D, cache.cube ? 6 : 1);
        if (!cache.slots[index].image)
            return UINT32_MAX;
        WriteSlot(context, cache, index, cache.slots[index]);
    }
    UploadImage(context, cache.slots[index], rgba, VkDeviceSize(width) * height * 4);
    return index;
}

uint32_t TextureReserve(Context& context, TextureCache& cache, uint32_t width, uint32_t height)
{
    uint32_t index;
    if (!cache.freeSlots.empty())
    {
        index = cache.freeSlots.back();
        cache.freeSlots.pop_back();
    }
    else if (cache.next < cache.capacity)
        index = cache.next++;
    else
        return UINT32_MAX;
    if (cache.slots[index].image)
        DestroyImage(context, cache.slots[index]);
    cache.slots[index] = CreateImage(context, width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    if (!cache.slots[index].image)
        return UINT32_MAX;
    WriteSlot(context, cache, index, cache.slots[index]);
    // The descriptor points at a SHADER_READ image; the resolve moves it there when it is done.
    return index;
}

uint32_t TextureAddCube(Context& context, TextureCache& cache, uint32_t width, uint32_t height, const uint8_t* const faces[6])
{
    uint32_t index;
    if (!cache.freeSlots.empty())
    {
        index = cache.freeSlots.back();
        cache.freeSlots.pop_back();
    }
    else if (cache.next < cache.capacity)
        index = cache.next++;
    else
        return UINT32_MAX;
    if (cache.slots[index].image)
        DestroyImage(context, cache.slots[index]);
    cache.slots[index] = CreateImage(context, width, height, VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_VIEW_TYPE_CUBE, 6);
    if (!cache.slots[index].image)
        return UINT32_MAX;
    WriteSlot(context, cache, index, cache.slots[index]);
    // One upload per face: cheap enough for the handful of cube maps a scene has.
    for (uint32_t face = 0; face < 6; face++)
    {
        void* staging = nullptr;
        const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
        VkBuffer buffer = CreateBuffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &staging);
        if (!buffer)
            return UINT32_MAX;
        memcpy(staging, faces[face], size_t(bytes));
        Submit(context, [&](VkCommandBuffer cmd) {
            VkImageMemoryBarrier toCopy{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            toCopy.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            toCopy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toCopy.srcQueueFamilyIndex = toCopy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            toCopy.image = cache.slots[index].image;
            toCopy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, face, 1 };
            toCopy.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                &toCopy);
            VkBufferImageCopy region{};
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, face, 1 };
            region.imageExtent = { width, height, 1 };
            vkCmdCopyBufferToImage(cmd, buffer, cache.slots[index].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            VkImageMemoryBarrier toSampled = toCopy;
            toSampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            toSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            toSampled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            toSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
                nullptr, 1, &toSampled);
        });
        vkDeviceWaitIdle(context.device);
        vkDestroyBuffer(context.device, buffer, nullptr);
    }
    return index;
}

void TextureFree(Context& context, TextureCache& cache, uint32_t index)
{
    if (index >= cache.capacity || !cache.slots[index].image)
        return;
    // Keep the image: a freed slot is reused by the next add, and its descriptor already points
    // at it. Only the slot is returned to the free list.
    cache.freeSlots.push_back(index);
}

bool TextureReplace(Context& context, TextureCache& cache, uint32_t index, uint32_t width, uint32_t height, const void* rgba)
{
    if (index >= cache.capacity || !cache.slots[index].image)
        return false;
    const Image& image = cache.slots[index];
    if (image.width != width || image.height != height)
    {
        // A different size needs a new image; the descriptor is rewritten, so shaders that
        // already have the index keep working.
        DestroyImage(context, cache.slots[index]);
        cache.slots[index] = CreateImage(context, width, height, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (!cache.slots[index].image)
            return false;
        WriteSlot(context, cache, index, cache.slots[index]);
    }
    UploadImage(context, cache.slots[index], rgba, VkDeviceSize(width) * height * 4);
    return true;
}
}
