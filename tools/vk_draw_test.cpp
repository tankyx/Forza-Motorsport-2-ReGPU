// Drives the Vulkan backend's per-draw submission the way the renderer will: build a
// VSConstants/PSConstants block, an expanded index list, bind the chain and draw — then check
// the result pixel by pixel. Covers what a draw needs besides the device and the shaders:
// the constant ring with dynamic offsets, the index ring, the bindless texture set and the
// push-constant-free bind chain.
//
//   FM2_TEXTURE=<rgba> FM2_TEXTURE_SIZE=<w>x<h> ./out/vk_draw_test
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gpu/vk/vk_context.h"
#include "gpu/vk/vk_draw.h"
#include "gpu/vk/vk_shader.h"
#include "gpu/vk/vk_state.h"
#include "gpu/vk/vk_texture.h"

namespace
{
// Reads the constant blocks exactly as the game's shaders do: vc[128] holds the quad's
// transform, gtex[0] the texture.
const char kQuadVs[] = R"(
cbuffer VSConstants : register(b0)
{
    float4 vc[256];
    uint4 vb[2];
    uint4 vf[96];
    uint4 vinfo;
    float4 vpoint;
    float4 vscale;
    float4 voffset;
    uint4 vte;
    uint4 vfl[32];
    float4 vsquash[4];
    float4 vwide;
    float4 vjitter;
};

struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};

// Vertex pull, like the translated shaders: the vertex stream lives in guest memory.
ByteAddressBuffer gphys : register(t1);

VSOut main(uint id : SV_VertexID)
{
    uint base = asuint(vc[128].x);
    float2 p = asfloat(gphys.Load2(base + id * 8));
    VSOut o;
    o.pos = float4(p * vc[128].y + vc[128].zw, 0, 1);
    o.uv = float4(p * 0.5 + 0.5, 0, 1);
    return o;
}
)";

const char kSamplePs[] = R"(
cbuffer PSConstants : register(b1)
{
    float4 pc[256];
    uint4 pb[2];
    uint4 pinfo;
    float4 pinfo2;
    uint4 tinfo[32];
    float4 tbias[32];
};
Texture2D<float4> gtex[] : register(t0, space1);
SamplerState sPoint : register(s0);

struct VSOut
{
    float4 pos : SV_Position;
    float4 uv : TEXCOORD0;
};

float4 main(VSOut input) : SV_Target0
{
    return gtex[0].Sample(sPoint, input.uv.xy) * pc[64];
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
}

int main()
{
    const uint32_t width = 320, height = 200;
    std::vector<uint8_t> source(size_t(width) * height * 4);
    for (uint32_t y = 0; y < height; y++)
        for (uint32_t x = 0; x < width; x++)
        {
            uint8_t* p = &source[(size_t(y) * width + x) * 4];
            const bool on = ((x / 16) + (y / 16)) & 1;
            p[0] = on ? 220 : 30;
            p[1] = uint8_t(255 * x / width);
            p[2] = on ? 60 : 30;
            p[3] = 255;
        }

    vk::Context context;
    if (!vk::Init(context, "out/vkcache"))
        return 1;
    printf("vk: %s\n", context.props.deviceName);

    // Guest memory, as the renderer maps it: vertex data at an offset inside it.
    void* gphysCpu = nullptr;
    VkBuffer gphys = vk::CreateBuffer(context, 1ull << 20, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &gphysCpu);
    void* gmemCpu = nullptr;
    VkBuffer gmem = vk::CreateBuffer(context, 1ull << 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &gmemCpu);
    const uint32_t vertexBase = 4096;
    const float quad[8] = { -1, -1, 1, -1, -1, 1, 1, 1 };
    memcpy(static_cast<uint8_t*>(gphysCpu) + vertexBase, quad, sizeof quad);

    vk::TextureCache textures;
    if (!vk::TextureCacheInit(context, textures, 1024))
        return 1;
    const uint32_t index = vk::TextureAdd(context, textures, width, height, source.data());
    if (index != 0)
    {
        fprintf(stderr, "unexpected bindless index %u\n", index);
        return 1;
    }

    VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samplerInfo.magFilter = samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.addressModeU = samplerInfo.addressModeV = samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VkSampler sampler = VK_NULL_HANDLE;
    vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler);
    VkSampler samplers[4] = { sampler, sampler, sampler, sampler };

    vk::DrawContext draw;
    if (!vk::DrawInit(context, draw, gmem, gphys, samplers, textures.set, VK_NULL_HANDLE, 4u << 20, 4u << 20))
        return 1;

    std::vector<uint32_t> vsCode, psCode;
    std::string error;
    if (!vk::CompileHlsl(kQuadVs, false, vk::HashSource(kQuadVs), "out/vkcache", vsCode, &error) ||
        !vk::CompileHlsl(kSamplePs, true, vk::HashSource(kSamplePs), "out/vkcache", psCode, &error))
    {
        fprintf(stderr, "vk: shader: %s\n", error.c_str());
        return 1;
    }

    // Target: one colour image, the way a minimal EDRAM surface will look.
    vk::Image target = vk::CreateImage(context, width, height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    VkAttachmentDescription color{};
    color.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference colorRef{ 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    VkRenderPassCreateInfo passInfo{ VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    passInfo.attachmentCount = 1;
    passInfo.pAttachments = &color;
    passInfo.subpassCount = 1;
    passInfo.pSubpasses = &subpass;
    VkRenderPass pass = VK_NULL_HANDLE;
    vkCreateRenderPass(context.device, &passInfo, nullptr, &pass);

    vk::StateKey key; // opaque blend, no depth: the state a UI draw uses
    key.blend = 0x00010001;           // src = ONE, dst = ZERO for colour and alpha: opaque
    key.stencilMaskMode = 0x0F << 24; // colour write mask (RB_COLOR_MASK = 0xF): no mask, no pixels
    key.flags = 2;                    // cullable, cull mode 0 (none)
    vk::TargetFormats targets;
    targets.color = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipeline pipeline = VK_NULL_HANDLE;
    const char* pipelineError = nullptr;
    VkShaderModule vs = Module(context, vsCode), ps = Module(context, psCode);
    if (!vk::CreatePipeline(context, key, vs, ps, draw.layout, pass, targets, pipeline, &pipelineError))
    {
        fprintf(stderr, "vk: %s\n", pipelineError ? pipelineError : "?");
        return 1;
    }

    // The draw's constants: the quad's transform in vc[128], a tint in pc[64].
    gpu::VSConstants vsConstants{};
    gpu::PSConstants psConstants{};
    reinterpret_cast<uint32_t*>(&vsConstants.c[128][0])[0] = vertexBase; // asuint, as the shader reads it
    vsConstants.c[128][1] = 1.0f; // 1:1, so every texel lands on its pixel
    vsConstants.c[128][2] = 0.0f;  // offset x
    vsConstants.c[128][3] = 0.0f;  // offset y
    psConstants.c[64][0] = 1.0f;
    psConstants.c[64][1] = 1.0f;
    psConstants.c[64][2] = 1.0f;
    psConstants.c[64][3] = 1.0f;

    vk::DrawBeginFrame(draw);

    VkFramebufferCreateInfo framebufferInfo{ VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    framebufferInfo.renderPass = pass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = &target.view;
    framebufferInfo.width = width;
    framebufferInfo.height = height;
    framebufferInfo.layers = 1;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    vkCreateFramebuffer(context.device, &framebufferInfo, nullptr, &framebuffer);

    void* readback = nullptr;
    VkBuffer copy = vk::CreateBuffer(context, VkDeviceSize(width) * height * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &readback);
    const uint32_t indices[6] = { 0, 1, 2, 2, 1, 3 }; // the quad as a triangle list
    const bool ok = vk::Submit(context, [&](VkCommandBuffer cb) {
        VkRenderPassBeginInfo begin{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        begin.renderPass = pass;
        begin.framebuffer = framebuffer;
        begin.renderArea = { { 0, 0 }, { width, height } };
        vkCmdBeginRenderPass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
        VkViewport v{ 0, 0, float(width), float(height), 0, 1 };
        VkRect2D s{ { 0, 0 }, { width, height } };
        vkCmdSetViewport(cb, 0, 1, &v);
        vkCmdSetScissor(cb, 0, 1, &s);
        vk::DrawIndexed(context, draw, cb, pipeline, textures.set, vsConstants, psConstants, indices, 6, true);
        vkCmdEndRenderPass(cb);
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { width, height, 1 };
        vkCmdCopyImageToBuffer(cb, target.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, copy, 1, &region);
    });
    if (!ok)
    {
        fprintf(stderr, "vk: submission failed\n");
        return 1;
    }

    // The quad covers the whole target (the vertex stream is in [-1,1]); compare with the source.
    auto half = [](uint16_t h) {
        const uint32_t sign = uint32_t(h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1F, man = h & 0x3FF, bits = 0;
        if (exp == 0)
        {
            bits = sign;
            if (man)
            {
                exp = 127 - 15 + 1;
                while (!(man & 0x400))
                {
                    man <<= 1;
                    exp--;
                }
                bits = sign | (exp << 23) | ((man & 0x3FF) << 13);
            }
        }
        else if (exp == 31)
            bits = sign | 0x7F800000u | (man << 13);
        else
            bits = sign | ((exp + 127 - 15) << 23) | (man << 13);
        float f;
        memcpy(&f, &bits, 4);
        return f;
    };
    const uint16_t* result = static_cast<const uint16_t*>(readback);
    size_t differ = 0, mirrored = 0, sampled = 0;
    float worst = 0;
    for (uint32_t y = 1; y < height - 1; y++)
        for (uint32_t x = 1; x < width - 1; x++, sampled++)
        {
            const uint16_t* px = result + (size_t(y) * width + x) * 4;
            const uint8_t* expected = &source[(size_t(y) * width + x) * 4];
            for (int c = 0; c < 3; c++)
                worst = std::max(worst, std::abs(half(px[c]) * 255.0f - expected[c]));
            const uint8_t* flippedRow = &source[(size_t(height - 1 - y) * width + x) * 4];
            const bool straight = std::abs(half(px[0]) * 255.0f - expected[0]) <= 1.5f &&
                std::abs(half(px[1]) * 255.0f - expected[1]) <= 1.5f &&
                std::abs(half(px[2]) * 255.0f - expected[2]) <= 1.5f;
            const bool flipped = std::abs(half(px[0]) * 255.0f - flippedRow[0]) <= 1.5f &&
                std::abs(half(px[1]) * 255.0f - flippedRow[1]) <= 1.5f &&
                std::abs(half(px[2]) * 255.0f - flippedRow[2]) <= 1.5f;
            differ += !straight;
            mirrored += !flipped;
        }
    const uint16_t* centre = result + (size_t(height / 2) * width + width / 2) * 4;
    {
        // Quick look at what came out.
        FILE* f = fopen("out/vk_draw.ppm", "wb");
        if (f)
        {
            fprintf(f, "P6\n%u %u\n255\n", width, height);
            for (uint32_t i = 0; i < width * height; i++)
            {
                uint8_t rgb[3];
                for (int c = 0; c < 3; c++)
                {
                    const float v = std::min(1.0f, std::max(0.0f, half(result[i * 4 + c])));
                    rgb[c] = uint8_t(v * 255.0f);
                }
                fwrite(rgb, 1, 3, f);
            }
            fclose(f);
        }
    }
    printf("vk: centre pixel %.3f %.3f %.3f, worst channel error %.1f/255\n", half(centre[0]), half(centre[1]), half(centre[2]),
        worst);
    printf("vk: %zu of %zu sampled pixels differ from the source texture (%zu if it were flipped)\n", differ, sampled,
        mirrored);
    vkDeviceWaitIdle(context.device);
    vk::DrawDestroy(context, draw);
    vk::Shutdown(context);
    return differ ? 2 : 0;
}
