#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_draw.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace vk
{
namespace
{
bool MakeRing(const Context& context, DrawRing& ring, VkDeviceSize size, VkBufferUsageFlags usage)
{
    ring.size = size;
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(context.device, &bi, nullptr, &ring.buffer) != VK_SUCCESS)
        return false;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(context.device, ring.buffer, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemory(context, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (ai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(context.device, &ai, nullptr, &ring.memory) != VK_SUCCESS ||
        vkBindBufferMemory(context.device, ring.buffer, ring.memory, 0) != VK_SUCCESS)
        return false;
    return vkMapMemory(context.device, ring.memory, 0, VK_WHOLE_SIZE, 0, reinterpret_cast<void**>(&ring.cpu)) == VK_SUCCESS;
}

VkDeviceSize Align(VkDeviceSize value, VkDeviceSize alignment) { return (value + alignment - 1) & ~(alignment - 1); }
}

bool DrawInit(Context& context, DrawContext& draw, VkBuffer gmem, VkBuffer gphys, VkSampler samplers[4],
    VkDescriptorSet bindless2D, VkDescriptorSet bindlessCube, VkDeviceSize constantsBytes, VkDeviceSize indexBytes)
{
    if (!MakeRing(context, draw.constants, constantsBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ||
        !MakeRing(context, draw.indices, indexBytes, VK_BUFFER_USAGE_INDEX_BUFFER_BIT))
        return false;
    draw.constantsInfo = { draw.constants.buffer, 0, VK_WHOLE_SIZE };
    draw.bindless2D = bindless2D;
    draw.bindlessCube = bindlessCube;

    VkDescriptorSetLayout set0Layout = VK_NULL_HANDLE;
    if (!CreateRendererLayout(context, draw.layout, &set0Layout))
        return false;

    VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }, { VK_DESCRIPTOR_TYPE_SAMPLER, 4 } };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &draw.pool) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = draw.pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &set0Layout;
    if (vkAllocateDescriptorSets(context.device, &allocate, &draw.set0) != VK_SUCCESS)
        return false;

    VkDescriptorBufferInfo guestMemory{ gmem, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo guestPhysical{ gphys, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo samplerInfo[4]{};
    for (int i = 0; i < 4; i++)
        samplerInfo[i] = { samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };

    VkWriteDescriptorSet writes[8]{};
    for (int i = 0; i < 8; i++)
    {
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet = draw.set0;
        writes[i].dstBinding = i < 2 ? uint32_t(i) : i < 4 ? uint32_t(100 + i - 2) : uint32_t(200 + i - 4);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            : i < 4 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_SAMPLER;
        if (i < 2)
            writes[i].pBufferInfo = &draw.constantsInfo;
        else if (i == 2)
            writes[i].pBufferInfo = &guestMemory;
        else if (i == 3)
            writes[i].pBufferInfo = &guestPhysical;
        else
            writes[i].pImageInfo = &samplerInfo[i - 4];
    }
    vkUpdateDescriptorSets(context.device, 8, writes, 0, nullptr);
    vkDestroyDescriptorSetLayout(context.device, set0Layout, nullptr);
    return true;
}

void DrawDestroy(Context& context, DrawContext& draw)
{
    for (DrawRing* ring : { &draw.constants, &draw.indices })
    {
        if (ring->cpu)
            vkUnmapMemory(context.device, ring->memory);
        if (ring->buffer)
            vkDestroyBuffer(context.device, ring->buffer, nullptr);
        if (ring->memory)
            vkFreeMemory(context.device, ring->memory, nullptr);
    }
    if (draw.pool)
        vkDestroyDescriptorPool(context.device, draw.pool, nullptr);
    if (draw.layout)
        vkDestroyPipelineLayout(context.device, draw.layout, nullptr);
    draw = {};
}

void DrawBeginFrame(DrawContext& draw)
{
    draw.constants.used = 0;
    draw.indices.used = 0;
}

bool DrawIndexed(const Context& context, DrawContext& draw, VkCommandBuffer cmd, VkPipeline pipeline, VkDescriptorSet textures,
    const gpu::VSConstants& vs, const gpu::PSConstants& ps, const uint32_t* indices, uint32_t count, bool indexed)
{
    const VkDeviceSize alignment = 256; // the alignment dynamic uniform offsets must respect on every driver
    const VkDeviceSize vsOffset = Align(draw.constants.used, alignment);
    const VkDeviceSize psOffset = Align(vsOffset + sizeof(gpu::VSConstants), alignment);
    const VkDeviceSize next = psOffset + sizeof(gpu::PSConstants);
    VkDeviceSize indexOffset = 0;
    if (indexed)
    {
        indexOffset = Align(draw.indices.used, alignment);
        if (indexOffset + VkDeviceSize(count) * 4 > draw.indices.size)
            return false;
    }
    if (next > draw.constants.size)
        return false;

    memcpy(draw.constants.cpu + vsOffset, &vs, sizeof vs);
    memcpy(draw.constants.cpu + psOffset, &ps, sizeof ps);
    draw.constants.used = next;
    if (indexed)
    {
        memcpy(draw.indices.cpu + indexOffset, indices, size_t(count) * 4);
        draw.indices.used = indexOffset + VkDeviceSize(count) * 4;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const uint32_t offsets[2] = { uint32_t(vsOffset), uint32_t(psOffset) };
    // Set 2 is the cube array; a draw that has none still binds the 2D set there.
    VkDescriptorSet sets[3] = { draw.set0, textures, draw.bindlessCube ? draw.bindlessCube : textures };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, draw.layout, 0, 3, sets, 2, offsets);
    if (indexed)
    {
        vkCmdBindIndexBuffer(cmd, draw.indices.buffer, indexOffset, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, count, 1, 0, 0, 0);
    }
    else
        vkCmdDraw(cmd, count, 1, 0, 0);
    (void)context;
    return true;
}

void DrawSetBlendConstants(VkCommandBuffer cmd, const float rgba[4]) { vkCmdSetBlendConstants(cmd, rgba); }
}
