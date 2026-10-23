#pragma once
// The backend's shader service: microcode -> translated HLSL (hlsl.cpp) -> SPIR-V (dxc) ->
// VkShaderModule, cached on disk by the same key the D3D12 path uses, so a draw that has
// already been seen costs a lookup and a cache read.
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vulkan/vulkan.h>

#include "gpu/vk/vk_context.h"

namespace vk
{
struct VkShader
{
    VkShaderModule module = VK_NULL_HANDLE;
    std::string hlsl;        // kept for the renderer's per-shader edits (car body squash, dirt seam fix)
    uint32_t constantsUsed = 0;
    uint32_t usedTextureSlots = 0;
    bool pixel = false;
    bool ok = false;
    std::string error;
};

struct ShaderSet
{
    std::unordered_map<uint64_t, VkShader> shaders;
    std::filesystem::path cacheDir;
    uint32_t compiled = 0, failed = 0, cached = 0;
};

void ShaderSetDestroy(Context& context, ShaderSet& set);

// Returns the shader for `hash`, translating and compiling it if this is the first time.
// The microcode must be the bytes the hash was taken over (Fnv over the used dwords, xor the
// shader type) — the renderer's cache key.
const VkShader* ShaderGet(Context& context, ShaderSet& set, uint64_t hash, const uint32_t* microcodeBE, uint32_t dwords,
    bool pixel);
}
