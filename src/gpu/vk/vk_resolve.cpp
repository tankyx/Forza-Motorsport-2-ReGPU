#include "gpu/vk/vk_resolve.h"

#include <vector>

#include "gpu/vk/vk_shader.h"

namespace vk
{
namespace
{
const char kResolveVs[] = R"(
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
    o.uv = float4(p.x, p.y, 0, 1);
    return o;
}
)";

const char kResolvePs[] = R"(
Texture2D<float4> src : register(t0);
SamplerState sPoint : register(s0);
struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};
float4 main(VSOut input) : SV_Target0
{
    return src.Sample(sPoint, input.uv.xy);
}
)";

const char kSwapPs[] = R"(
Texture2D<float4> src : register(t0);
SamplerState sPoint : register(s0);
struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};
float4 main(VSOut input) : SV_Target0
{
    float4 c = src.Sample(sPoint, input.uv.xy);
    return float4(c.b, c.g, c.r, c.a); // the copy's swap bit: red and blue exchanged
}
)";

VkShaderModule Module(const Context& context, const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = code.size() * 4;
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(context.device, &info, nullptr, &module) == VK_SUCCESS ? module : VK_NULL_HANDLE;
}

VkPipeline MakePipeline(const Context& context, VkShaderModule vs, VkShaderModule ps, VkPipelineLayout layout, VkRenderPass pass)
{
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1] = stages[0];
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = ps;
    VkPipelineVertexInputStateCreateInfo vertexInput{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo colorBlend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;
    VkDynamicState dynamic[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynamicState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamic;
    VkGraphicsPipelineCreateInfo info{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    info.stageCount = 2;
    info.pStages = stages;
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &assembly;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multisample;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamicState;
    info.layout = layout;
    info.renderPass = pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    return vkCreateGraphicsPipelines(context.device, context.pipelineCache, 1, &info, nullptr, &pipeline) == VK_SUCCESS ? pipeline
        : VK_NULL_HANDLE;
}
}

bool ResolveInit(Context& context, Resolve& resolve)
{
    std::vector<uint32_t> vsCode, psCode, swapCode;
    std::string error;
    if (!CompileHlsl(kResolveVs, false, HashSource(kResolveVs), "out/vkcache", vsCode, &error) ||
        !CompileHlsl(kResolvePs, true, HashSource(kResolvePs), "out/vkcache", psCode, &error) ||
        !CompileHlsl(kSwapPs, true, HashSource(kSwapPs), "out/vkcache", swapCode, &error))
    {
        fprintf(stderr, "vk: resolve shaders: %s\n", error.c_str());
        return false;
    }
    VkShaderModule vs = Module(context, vsCode), ps = Module(context, psCode), swap = Module(context, swapCode);
    if (!vs || !ps || !swap)
        return false;

    // dxc's register mapping: t0 is binding 100, s0 is binding 200.
    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 100;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 200;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 2;
    layoutInfo.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &resolve.setLayout) != VK_SUCCESS)
        return false;
    VkPipelineLayoutCreateInfo pipelineLayout{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayout.setLayoutCount = 1;
    pipelineLayout.pSetLayouts = &resolve.setLayout;
    if (vkCreatePipelineLayout(context.device, &pipelineLayout, nullptr, &resolve.layout) != VK_SUCCESS)
        return false;

    VkAttachmentDescription color{};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // the barrier after moves it on
    VkAttachmentReference ref{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &ref;
    VkRenderPassCreateInfo passInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    passInfo.attachmentCount = 1;
    passInfo.pAttachments = &color;
    passInfo.subpassCount = 1;
    passInfo.pSubpasses = &subpass;
    if (vkCreateRenderPass(context.device, &passInfo, nullptr, &resolve.pass) != VK_SUCCESS)
        return false;

    resolve.copy = MakePipeline(context, vs, ps, resolve.layout, resolve.pass);
    resolve.swap = MakePipeline(context, vs, swap, resolve.layout, resolve.pass);
    vkDestroyShaderModule(context.device, vs, nullptr);
    vkDestroyShaderModule(context.device, ps, nullptr);
    vkDestroyShaderModule(context.device, swap, nullptr);
    if (!resolve.copy || !resolve.swap)
        return false;

    VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(context.device, &samplerInfo, nullptr, &resolve.sampler) != VK_SUCCESS)
        return false;

    VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64 }, { VK_DESCRIPTOR_TYPE_SAMPLER, 64 } };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 64;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = sizes;
    return vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &resolve.pool) == VK_SUCCESS;
}

void ResolveShutdown(Context& context, Resolve& resolve)
{
    if (resolve.pool)
        vkDestroyDescriptorPool(context.device, resolve.pool, nullptr);
    if (resolve.sampler)
        vkDestroySampler(context.device, resolve.sampler, nullptr);
    if (resolve.copy)
        vkDestroyPipeline(context.device, resolve.copy, nullptr);
    if (resolve.swap)
        vkDestroyPipeline(context.device, resolve.swap, nullptr);
    if (resolve.pass)
        vkDestroyRenderPass(context.device, resolve.pass, nullptr);
    if (resolve.layout)
        vkDestroyPipelineLayout(context.device, resolve.layout, nullptr);
    if (resolve.setLayout)
        vkDestroyDescriptorSetLayout(context.device, resolve.setLayout, nullptr);
    resolve = {};
}

bool ResolveImage(const Context& context, Resolve& resolve, VkCommandBuffer cmd, VkImageView source, VkImage destination,
    uint32_t width, uint32_t height, bool swapRedBlue)
{
    VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = resolve.pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &resolve.setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(context.device, &allocate, &set) != VK_SUCCESS)
        return false;
    VkDescriptorImageInfo image{ VK_NULL_HANDLE, source, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo sampler{ resolve.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    VkWriteDescriptorSet writes[2]{};
    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    writes[0].dstSet = set;
    writes[0].dstBinding = 100;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    writes[0].pImageInfo = &image;
    writes[1] = writes[0];
    writes[1].dstBinding = 200;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    writes[1].pImageInfo = &sampler;
    vkUpdateDescriptorSets(context.device, 2, writes, 0, nullptr);

    VkImageView view = VK_NULL_HANDLE;
    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = destination;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(context.device, &viewInfo, nullptr, &view) != VK_SUCCESS)
        return false;
    VkFramebufferCreateInfo framebufferInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    framebufferInfo.renderPass = resolve.pass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &view;
    framebufferInfo.width = width;
    framebufferInfo.height = height;
    framebufferInfo.layers = 1;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(context.device, &framebufferInfo, nullptr, &framebuffer) != VK_SUCCESS)
        return false;

    VkImageMemoryBarrier toColor{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toColor.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColor.srcQueueFamilyIndex = toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColor.image = destination;
    toColor.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr,
        1, &toColor);

    VkRenderPassBeginInfo begin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    begin.renderPass = resolve.pass;
    begin.framebuffer = framebuffer;
    begin.renderArea = { { 0, 0 }, { width, height } };
    vkCmdBeginRenderPass(cmd, &begin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, swapRedBlue ? resolve.swap : resolve.copy);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, resolve.layout, 0, 1, &set, 0, nullptr);
    VkViewport viewport{ 0, 0, float(width), float(height), 0, 1 };
    VkRect2D scissor{ { 0, 0 }, { width, height } };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRenderPass(cmd);

    VkImageMemoryBarrier toSampled{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    toSampled.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; // the pass's finalLayout
    toSampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    toSampled.srcQueueFamilyIndex = toSampled.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toSampled.image = destination;
    toSampled.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    toSampled.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toSampled.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0,
        nullptr, 1, &toSampled);

    vkDestroyFramebuffer(context.device, framebuffer, nullptr);
    vkDestroyImageView(context.device, view, nullptr);
    vkFreeDescriptorSets(context.device, resolve.pool, 1, &set);
    return true;
}
}
