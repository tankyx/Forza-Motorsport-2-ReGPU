#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_shaders.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "log.h"

#include "gpu/hlsl.h"
#include "gpu/ucode.h"
#include "gpu/vk/vk_shader.h"

namespace vk
{
void ShaderSetDestroy(Context& context, ShaderSet& set)
{
    for (auto& [hash, shader] : set.shaders)
        if (shader.module)
            vkDestroyShaderModule(context.device, shader.module, nullptr);
    set.shaders.clear();
}

const VkShader* ShaderGet(Context& context, ShaderSet& set, uint64_t hash, const uint32_t* microcodeBE, uint32_t dwords,
    bool pixel)
{
    if (auto it = set.shaders.find(hash); it != set.shaders.end())
        return &it->second;

    VkShader shader;
    shader.pixel = pixel;
    const ucode::Shader parsed = ucode::Parse(microcodeBE, dwords, pixel);
    const TranslatedShader translated = TranslateShader(parsed);
    if (!translated.ok)
    {
        shader.error = translated.error;
        set.failed++;
        auto& entry = set.shaders[hash];
        entry = std::move(shader);
        return &entry;
    }
    shader.hlsl = translated.hlsl;
    // FM2_DUMP_HLSL, as on the D3D12 path: keep each translated shader for reading (the Vulkan
    // backend translates in its own service, so the D3D12 side's dump never fired here).
    if (getenv("FM2_DUMP_HLSL"))
    {
        std::filesystem::create_directories("out/hlsl");
        char name[64];
        snprintf(name, sizeof(name), "out/hlsl/%s_%016llx.hlsl", pixel ? "ps" : "vs", (unsigned long long)hash);
        std::ofstream(name, std::ios::binary).write(shader.hlsl.data(), std::streamsize(shader.hlsl.size()));
    }
    shader.constantsUsed = translated.constantsUsed;
    shader.usedTextureSlots = translated.usedTextureSlots;
    for (const auto& entry : translated.vertexFetchLayout)
        shader.vertexFetchLayout.insert(shader.vertexFetchLayout.end(), entry.begin(), entry.end());
    shader.vertexFetchConstants = translated.vertexFetchConstants;
    {
        std::string list;
        for (uint32_t k : shader.vertexFetchConstants)
        {
            char t[8];
            snprintf(t, sizeof(t), " %u", k);
            list += t;
        }
        Log("shader_vk: %s %016llx fetches[%s ], layout %zu, constants %u", pixel ? "ps" : "vs", (unsigned long long)hash, list.c_str(),
            shader.vertexFetchLayout.size(), shader.constantsUsed);
    }

    std::vector<uint32_t> spirv;
    std::string error;
    if (!CompileHlsl(shader.hlsl, pixel, hash, set.cacheDir, spirv, &error))
    {
        shader.error = error.empty() ? "dxc failed" : error;
        set.failed++;
        auto& entry = set.shaders[hash];
        entry = std::move(shader);
        return &entry;
    }
    VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    info.codeSize = spirv.size() * 4;
    info.pCode = spirv.data();
    if (vkCreateShaderModule(context.device, &info, nullptr, &shader.module) != VK_SUCCESS)
    {
        shader.error = "vkCreateShaderModule failed";
        set.failed++;
        auto& entry = set.shaders[hash];
        entry = std::move(shader);
        return &entry;
    }
    shader.ok = true;
    set.compiled++;
    {
        static uint32_t logged = 0;
        if (logged < 12)
        {
            logged++;
            Log("shader_vk: %s %016llx texture slots %08x, constants %u", pixel ? "ps" : "vs", (unsigned long long)hash,
                shader.usedTextureSlots, shader.constantsUsed);
        }
    }
    auto& entry = set.shaders[hash];
    entry = std::move(shader);
    return &entry;
}
}
