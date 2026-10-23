#include "gpu/vk/vk_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace vk
{
namespace
{
bool Fail(const char* what, VkResult result)
{
    fprintf(stderr, "vk: %s failed (%d)\n", what, int(result));
    return false;
}
}

bool Init(Context& context, const std::filesystem::path& cacheDir)
{
    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "fm2";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &context.instance) != VK_SUCCESS)
        return Fail("vkCreateInstance", VK_ERROR_INITIALIZATION_FAILED);

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(context.instance, &count, nullptr);
    if (!count)
        return Fail("no Vulkan device", VK_ERROR_INITIALIZATION_FAILED);
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(context.instance, &count, devices.data());
    context.physical = devices[0];
    vkGetPhysicalDeviceProperties(context.physical, &context.props);

    VkPhysicalDeviceDescriptorIndexingFeatures indexing{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES };
    indexing.runtimeDescriptorArray = VK_TRUE;
    indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    indexing.descriptorBindingPartiallyBound = VK_TRUE;
    indexing.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    VkPhysicalDeviceFeatures2 query{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    query.pNext = &indexing;
    vkGetPhysicalDeviceFeatures2(context.physical, &query);
    context.descriptorIndexing = indexing.runtimeDescriptorArray == VK_TRUE && indexing.shaderSampledImageArrayNonUniformIndexing == VK_TRUE;

    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(context.physical, &familyCount, families.data());
    context.queueFamily = familyCount;
    for (uint32_t i = 0; i < familyCount; i++)
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
        {
            context.queueFamily = i;
            break;
        }
    if (context.queueFamily == familyCount)
        return Fail("no graphics queue", VK_ERROR_INITIALIZATION_FAILED);

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = context.queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkPhysicalDeviceFeatures2 features{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    if (context.descriptorIndexing)
        features.pNext = &indexing;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &features;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (vkCreateDevice(context.physical, &dci, nullptr, &context.device) != VK_SUCCESS)
        return Fail("vkCreateDevice", VK_ERROR_INITIALIZATION_FAILED);
    vkGetDeviceQueue(context.device, context.queueFamily, 0, &context.queue);

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.queueFamilyIndex = context.queueFamily;
    if (vkCreateCommandPool(context.device, &pci, nullptr, &context.pool) != VK_SUCCESS)
        return Fail("vkCreateCommandPool", VK_ERROR_INITIALIZATION_FAILED);

    std::vector<char> cached;
    VkPipelineCacheCreateInfo pcci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    if (!cacheDir.empty())
    {
        context.pipelineCachePath = cacheDir / "pipelines.bin";
        if (FILE* f = fopen(context.pipelineCachePath.string().c_str(), "rb"))
        {
            fseek(f, 0, SEEK_END);
            cached.resize(size_t(ftell(f)));
            fseek(f, 0, SEEK_SET);
            if (fread(cached.data(), 1, cached.size(), f) != cached.size())
                cached.clear();
            fclose(f);
        }
        pcci.initialDataSize = cached.size();
        pcci.pInitialData = cached.data();
    }
    if (vkCreatePipelineCache(context.device, &pcci, nullptr, &context.pipelineCache) != VK_SUCCESS)
        return Fail("vkCreatePipelineCache", VK_ERROR_INITIALIZATION_FAILED);
    return true;
}

void Shutdown(Context& context)
{
    if (context.device)
    {
        vkDeviceWaitIdle(context.device);
        if (context.pipelineCache && !context.pipelineCachePath.empty())
        {
            size_t bytes = 0;
            if (vkGetPipelineCacheData(context.device, context.pipelineCache, &bytes, nullptr) == VK_SUCCESS && bytes)
            {
                std::vector<char> data(bytes);
                if (vkGetPipelineCacheData(context.device, context.pipelineCache, &bytes, data.data()) == VK_SUCCESS)
                {
                    std::filesystem::create_directories(context.pipelineCachePath.parent_path());
                    if (FILE* f = fopen(context.pipelineCachePath.string().c_str(), "wb"))
                    {
                        fwrite(data.data(), 1, bytes, f);
                        fclose(f);
                    }
                }
            }
        }
        if (context.pipelineCache)
            vkDestroyPipelineCache(context.device, context.pipelineCache, nullptr);
        if (context.pool)
            vkDestroyCommandPool(context.device, context.pool, nullptr);
        vkDestroyDevice(context.device, nullptr);
    }
    if (context.instance)
        vkDestroyInstance(context.instance, nullptr);
    context = {};
}

bool CreateRendererLayout(const Context& context, VkPipelineLayout& layout)
{
    const VkShaderStageFlags both = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutBinding sets0[8]{};
    auto bind = [&](int i, VkDescriptorType type, uint32_t binding, VkShaderStageFlags stages) {
        sets0[i].binding = binding;
        sets0[i].descriptorType = type;
        sets0[i].descriptorCount = 1;
        sets0[i].stageFlags = stages;
    };
    bind(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, both);       // VSConstants
    bind(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, both);       // PSConstants
    bind(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 100, both);     // gmem
    bind(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 101, both);     // gphys (guest memory)
    for (int i = 0; i < 4; i++)
        bind(4 + i, VK_DESCRIPTOR_TYPE_SAMPLER, 200 + uint32_t(i), VK_SHADER_STAGE_FRAGMENT_BIT);
    VkDescriptorSetLayoutCreateInfo i0{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    i0.bindingCount = 8;
    i0.pBindings = sets0;
    VkDescriptorSetLayout set0 = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(context.device, &i0, nullptr, &set0) != VK_SUCCESS)
        return false;

    // The bindless arrays: indexed at runtime, partially filled, appended to as textures load.
    VkDescriptorSetLayoutBinding array{};
    array.binding = 0;
    array.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    array.descriptorCount = 1024;
    array.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo bf{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
    bf.bindingCount = 1;
    bf.pBindingFlags = &flags;
    VkDescriptorSetLayoutCreateInfo ia{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ia.pNext = &bf;
    ia.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    ia.bindingCount = 1;
    ia.pBindings = &array;
    VkDescriptorSetLayout arrayLayout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(context.device, &ia, nullptr, &arrayLayout) != VK_SUCCESS)
        return false;

    VkDescriptorSetLayout sets[3] = { set0, arrayLayout, arrayLayout };
    VkPushConstantRange push{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 3;
    pli.pSetLayouts = sets;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &push;
    const bool ok = vkCreatePipelineLayout(context.device, &pli, nullptr, &layout) == VK_SUCCESS;
    vkDestroyDescriptorSetLayout(context.device, arrayLayout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, set0, nullptr);
    return ok;
}

uint32_t FindMemory(const Context& context, uint32_t bits, VkMemoryPropertyFlags want)
{
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(context.physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

VkBuffer CreateBuffer(const Context& context, VkDeviceSize size, VkBufferUsageFlags usage, void** mapped)
{
    VkBufferCreateInfo bi{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (vkCreateBuffer(context.device, &bi, nullptr, &buffer) != VK_SUCCESS)
        return VK_NULL_HANDLE;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(context.device, buffer, &req);
    const VkMemoryPropertyFlags want = mapped ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) : 0;
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemory(context, req.memoryTypeBits, want);
    VkDeviceMemory memory = VK_NULL_HANDLE;
    if (vkAllocateMemory(context.device, &ai, nullptr, &memory) != VK_SUCCESS ||
        vkBindBufferMemory(context.device, buffer, memory, 0) != VK_SUCCESS)
    {
        vkDestroyBuffer(context.device, buffer, nullptr);
        return VK_NULL_HANDLE;
    }
    if (mapped)
        vkMapMemory(context.device, memory, 0, VK_WHOLE_SIZE, 0, mapped);
    return buffer; // memory is owned by the buffer; freed with it below only in the test
}

Image CreateImage(const Context& context, uint32_t width, uint32_t height, VkFormat format, VkImageUsageFlags usage,
    VkImageViewType viewType, uint32_t layers)
{
    Image image;
    image.format = format;
    image.width = width;
    image.height = height;
    VkImageCreateInfo ci{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { width, height, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (viewType == VK_IMAGE_VIEW_TYPE_CUBE)
        ci.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    if (vkCreateImage(context.device, &ci, nullptr, &image.image) != VK_SUCCESS)
        return {};
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(context.device, image.image, &req);
    VkMemoryAllocateInfo ai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = FindMemory(context, req.memoryTypeBits, 0);
    if (vkAllocateMemory(context.device, &ai, nullptr, &image.memory) != VK_SUCCESS ||
        vkBindImageMemory(context.device, image.image, image.memory, 0) != VK_SUCCESS)
    {
        DestroyImage(context, image);
        return {};
    }
    VkImageViewCreateInfo vi{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = image.image;
    vi.viewType = viewType;
    vi.format = format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
    if (vkCreateImageView(context.device, &vi, nullptr, &image.view) != VK_SUCCESS)
    {
        DestroyImage(context, image);
        return {};
    }
    return image;
}

void DestroyImage(const Context& context, Image& image)
{
    if (image.view)
        vkDestroyImageView(context.device, image.view, nullptr);
    if (image.image)
        vkDestroyImage(context.device, image.image, nullptr);
    if (image.memory)
        vkFreeMemory(context.device, image.memory, nullptr);
    image = {};
}

void UploadImage(const Context& context, const Image& image, const void* rgba, VkDeviceSize bytes)
{
    void* mapped = nullptr;
    VkBuffer staging = CreateBuffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &mapped);
    if (!staging)
        return;
    memcpy(mapped, rgba, size_t(bytes));
    Submit(context, [&](VkCommandBuffer cb) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = image.image;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        VkBufferImageCopy copy{};
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.imageExtent = { image.width, image.height, 1 };
        vkCmdCopyBufferToImage(cb, staging, image.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    });
    vkDeviceWaitIdle(context.device);
    vkDestroyBuffer(context.device, staging, nullptr);
}
}
