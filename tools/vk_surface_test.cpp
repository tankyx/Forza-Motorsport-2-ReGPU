// The EDRAM surface model: two passes into one surface (cleared, then loaded), then a resolve
// to a single-sampled image and a readback — the path a frame of the game takes. It also checks
// the thing MSAA is for: that a resolve blends a triangle's edge (1x cannot, 4x must).
//
//   ./out/vk_surface_test
#include <cstdio>
#include <cstring>
#include <vector>

#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_context.h"
#include "gpu/vk/vk_shader.h"
#include "gpu/vk/vk_surface.h"

namespace
{
const char kQuadVs[] = R"(
struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};

// A strip covering the left half, its edge deliberately off a pixel boundary so a multisampled
// resolve has something to blend.
VSOut main(uint id : SV_VertexID)
{
    float2 p = float2(id & 1, (id >> 1) & 1);
    VSOut o;
    o.pos = float4(p.x * 0.5f - 1.0f + 0.0015f, p.y * 2.0f - 1.0f, 0, 1);
    o.uv = float4(p, 0, 1);
    return o;
}
)";

const char kColorPs[] = R"(
cbuffer PSConstants : register(b1)
{
    float4 pc[256];
};

struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};

float4 main(VSOut input) : SV_Target0
{
    return pc[0];
}
)";

VkShaderModule Module(vk::Context& context, const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = code.size() * 4;
    ci.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    return vkCreateShaderModule(context.device, &ci, nullptr, &module) == VK_SUCCESS ? module : VK_NULL_HANDLE;
}

VkPipeline MakePipeline(vk::Context& context, VkShaderModule vs, VkShaderModule ps, VkPipelineLayout layout, VkRenderPass pass,
    VkSampleCountFlagBits samples)
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
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo viewport{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    multisample.rasterizationSamples = samples;
    // The render pass has a depth attachment, so the pipeline must describe depth state even
    // when the test does not use it.
    VkPipelineDepthStencilStateCreateInfo depthStencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
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
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamicState;
    info.layout = layout;
    info.renderPass = pass;
    VkPipeline pipeline = VK_NULL_HANDLE;
    return vkCreateGraphicsPipelines(context.device, context.pipelineCache, 1, &info, nullptr, &pipeline) == VK_SUCCESS ? pipeline
        : VK_NULL_HANDLE;
}
}

int main()
{
    const uint32_t width = 320, height = 200;
    vk::Context context;
    if (!vk::Init(context, "out/vkcache"))
        return 1;
    printf("vk: %s\n", context.props.deviceName);

    std::vector<uint32_t> vsCode, psCode;
    std::string error;
    if (!vk::CompileHlsl(kQuadVs, false, vk::HashSource(kQuadVs), "out/vkcache", vsCode, &error) ||
        !vk::CompileHlsl(kColorPs, true, vk::HashSource(kColorPs), "out/vkcache", psCode, &error))
    {
        fprintf(stderr, "vk: shader: %s\n", error.c_str());
        return 1;
    }
    VkShaderModule vs = Module(context, vsCode), ps = Module(context, psCode);

    vk::SurfaceCache surfaces;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout set0Layout = VK_NULL_HANDLE;
    if (!vk::CreateRendererLayout(context, layout, &set0Layout))
        return 1;

    // The pass reads its colour from the pixel constants, so a descriptor set with the two
    // dynamic constant buffers has to be bound (an unbound one is not a defined read).
    void* constantsCpu = nullptr;
    VkBuffer constants = vk::CreateBuffer(context, 65536,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &constantsCpu);
    const VkDeviceSize psOffset = 32768;
    float* pc = reinterpret_cast<float*>(static_cast<uint8_t*>(constantsCpu) + psOffset);
    pc[0] = 1.0f; // the colour the pixel shader returns
    pc[1] = 0.0f;
    pc[2] = 0.0f;
    pc[3] = 1.0f;
    VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler);
    VkDescriptorPoolSize poolSizes[3] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 }, { VK_DESCRIPTOR_TYPE_SAMPLER, 4 } };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &pool);
    VkDescriptorSetAllocateInfo allocate{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocate.descriptorPool = pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &set0Layout;
    VkDescriptorSet set0 = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(context.device, &allocate, &set0);
    VkDescriptorBufferInfo bufferInfo{ constants, 0, VK_WHOLE_SIZE };
    VkDescriptorImageInfo samplerDesc{ sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    VkWriteDescriptorSet writes[8]{};
    for (int i = 0; i < 8; i++)
    {
        writes[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        writes[i].dstSet = set0;
        writes[i].dstBinding = i < 2 ? uint32_t(i) : i < 4 ? uint32_t(100 + i - 2) : uint32_t(200 + i - 4);
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
            : i < 4 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_SAMPLER;
        if (i < 4)
            writes[i].pBufferInfo = &bufferInfo;
        else
            writes[i].pImageInfo = &samplerDesc;
    }
    vkUpdateDescriptorSets(context.device, 8, writes, 0, nullptr);

    int failures = 0;
    for (VkSampleCountFlagBits samples : { VK_SAMPLE_COUNT_1_BIT, VK_SAMPLE_COUNT_4_BIT })
    {
        vk::Surface* surface = vk::SurfaceAcquire(context, surfaces, 0xD0000000, width, height, samples);
        if (!surface)
        {
            fprintf(stderr, "vk: no surface\n");
            return 1;
        }
        // The pipeline must match the render pass's sample count; the test builds one per pass.
        VkPipeline pipeline = MakePipeline(context, vs, ps, layout, surface->pass, samples);
        if (!pipeline)
        {
            fprintf(stderr, "vk: no pipeline\n");
            return 1;
        }
        vk::Image resolved = vk::CreateImage(context, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        void* readback = nullptr;
        VkBuffer copy = vk::CreateBuffer(context, VkDeviceSize(width) * height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback);

        // Two passes: clear to blue, draw red over the left (1x) or a half triangle (MSAA).
        const bool ok = vk::Submit(context, [&](VkCommandBuffer cmd) {
            VkClearColorValue blue{};
            blue.float32[0] = 0.0f;
            blue.float32[1] = 0.0f;
            blue.float32[2] = 1.0f;
            blue.float32[3] = 1.0f;
            vk::SurfaceBegin(context, *surface, cmd, &blue);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            VkViewport v{ 0, 0, float(width), float(height), 0, 1 };
            VkRect2D s{ { 0, 0 }, { width, height } };
            vkCmdSetViewport(cmd, 0, 1, &v);
            vkCmdSetScissor(cmd, 0, 1, &s);
            const uint32_t offsets[2] = { 0, uint32_t(psOffset) };
            VkDescriptorSet sets[3] = { set0, set0, set0 }; // set 1/2 unused by this shader
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 3, sets, 2, offsets);
            vkCmdDraw(cmd, 4, 1, 0, 0); // the strip: left half of the surface
            vk::SurfaceEnd(context, *surface, cmd);
            vk::SurfaceResolve(context, *surface, cmd, resolved.image, width, height);
            VkBufferImageCopy region{};
            region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.imageExtent = { width, height, 1 };
            vkCmdCopyImageToBuffer(cmd, resolved.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy, 1, &region);
        });
        if (!ok)
        {
            fprintf(stderr, "vk: submission failed\n");
            return 1;
        }

        auto half = [](uint16_t h) {
            const uint32_t sign = uint32_t(h & 0x8000u) << 16;
            uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits = 0;
            if (exp == 0)
                bits = sign;
            else if (exp == 31)
                bits = sign | 0x7F800000u | (man << 13);
            else
                bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
            float f;
            memcpy(&f, &bits, 4);
            return f;
        };
        const uint16_t* px = static_cast<const uint16_t*>(readback);
        // The clear survives where the strip never covers: the top-right corner.
        const uint16_t* corner = px + (size_t(4) * width + (width - 5)) * 4;
        const float cornerBlue = half(corner[2]);
        // The drawn area: the bottom-left corner inside the triangle.
        const uint16_t* inside = px + (size_t(height - 5) * width + 4) * 4;
        const float insideRed = half(inside[0]);
        size_t blended = 0;
        for (uint32_t y = 0; y < height; y++)
            for (uint32_t x = 0; x < width; x++)
            {
                const uint16_t* p = px + (size_t(y) * width + x) * 4;
                const float r = half(p[0]), b = half(p[2]);
                blended += r > 0.1f && b > 0.1f;
            }
        printf("vk: %ux: corner blue %.2f, inside red %.2f, %zu blended pixels\n", uint32_t(samples), cornerBlue, insideRed, blended);
        failures += !(cornerBlue > 0.9f && insideRed > 0.9f);
        // 1x has hard edges: a pixel is one colour or the other. 4x must blend along the edge.
        failures += samples == VK_SAMPLE_COUNT_4_BIT ? (blended == 0) : (blended > width);
        vkDeviceWaitIdle(context.device);
        vkDestroyBuffer(context.device, copy, nullptr);
        vkDestroyPipeline(context.device, pipeline, nullptr);
        vk::DestroyImage(context, resolved);
    }

    vk::SurfaceCacheDestroy(context, surfaces);
    vk::Shutdown(context);
    printf("vk: %s\n", failures ? "FAILED" : "ok");
    return failures ? 2 : 0;
}
