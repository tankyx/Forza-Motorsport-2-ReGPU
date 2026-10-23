#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_shaders.h"

#include <cstdio>

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
    shader.constantsUsed = translated.constantsUsed;
    shader.usedTextureSlots = translated.usedTextureSlots;

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
    auto& entry = set.shaders[hash];
    entry = std::move(shader);
    return &entry;
}
}
