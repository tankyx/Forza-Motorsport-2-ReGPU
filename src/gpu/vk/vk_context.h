#pragma once
// Vulkan device plumbing for the native renderer: instance, device, queue, memory and the
// one-shot command helpers everything else uses. No window and no swap chain yet: the
// renderer renders to images and the presenter reads them back (FM2_HEADLESS), so the whole
// backend is testable without a platform layer.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include "gpu/vk/vk_loader.h"

namespace vk
{
struct Context
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queueFamily = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    // Device-wide pipeline cache, kept on disk: creating a pipeline costs ~140 ms cold on this
    // driver and ~2.6 ms from the cache, which is the difference between a 2 minute first start
    // and a 2 second one for the game's ~780 pipelines.
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    std::filesystem::path pipelineCachePath;
    // Descriptor indexing: the shaders index their texture arrays with runtime values.
    bool descriptorIndexing = false;
    // VK_EXT_external_memory_host: the guest's own memory becomes a storage buffer, so the
    // shaders read vertex and index data where the game wrote it (the D3D12 path opens the same
    // memory as a heap).
    bool externalMemoryHost = false;
};

bool Init(Context& context, const std::filesystem::path& cacheDir = {});
void Shutdown(Context& context);

uint32_t FindMemory(const Context& context, uint32_t bits, VkMemoryPropertyFlags want);

// Host-visible, coherent buffer. mapped is kept valid for the buffer's lifetime.
VkBuffer CreateBuffer(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage, void** mapped);

// Wraps a page-aligned host allocation as a storage buffer without copying it. Returns
// VK_NULL_HANDLE when the device lacks VK_EXT_external_memory_host.
VkBuffer CreateHostImportedBuffer(const Context& context, void* host, VkDeviceSize size);

// The pipeline layout the game's translated shaders expect: two constant buffers, guest
// memory as storage buffers, four samplers, and one bindless array each for 2D and cube
// textures (dxc's register mapping: b0/b1, t0->100, t1->101, s0..s3->200..203, and the
// register spaces become descriptor sets 1 and 2).
// set0Out (optional) receives the layout of set 0, for the caller to allocate a descriptor set from.
bool CreateRendererLayout(const Context& context, VkPipelineLayout& layout, VkDescriptorSetLayout* set0Out = nullptr);

struct Image
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};
// Device-local 2D image with a view covering one mip and layer.
Image CreateImage(const Context& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D, uint32_t layers = 1,
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);
void DestroyImage(const Context& context, Image& image);

// Uploads RGBA8 rows (tightly packed) into a SHADER_READ_ONLY image.
void UploadImage(const Context& context, const Image& image, const void* rgba, VkDeviceSize bytes);

// Runs `record` in a command buffer and waits for it. Returns false if submission failed.
template <typename F>
bool Submit(const Context& context, F&& record)
{
    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = context.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    if (vkAllocateCommandBuffers(context.device, &ai, &cb) != VK_SUCCESS)
        return false;
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    bool ok = vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS;
    if (ok)
    {
        record(cb);
        ok = vkEndCommandBuffer(cb) == VK_SUCCESS;
    }
    if (ok)
    {
        VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        const VkResult submitted = vkQueueSubmit(context.queue, 1, &si, VK_NULL_HANDLE);
        const VkResult waited = submitted == VK_SUCCESS ? vkQueueWaitIdle(context.queue) : submitted;
        if (submitted != VK_SUCCESS || waited != VK_SUCCESS)
            fprintf(stderr, "vk: submit %d, wait %d\n", int(submitted), int(waited));
        ok = submitted == VK_SUCCESS && waited == VK_SUCCESS;
    }
    vkFreeCommandBuffers(context.device, context.pool, 1, &cb);
    return ok;
}
}
