#pragma once
// HLSL to SPIR-V for the Vulkan backend. The renderer's translated shaders are HLSL text,
// so the Vulkan path compiles them with dxc instead of fxc; the result is cached on disk by
// the same shader key the D3D12 path uses, so a start only compiles what changed.
//
// dxc is run as a child process for now (path in FM2_DXC, else "dxc" on PATH); linking
// libdxcompiler is the next step, once the pipeline cache is written too.
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace vk
{
// pixel selects ps_6_0/vs_6_0. cacheDir may be empty to skip the disk cache.
bool CompileHlsl(const std::string& hlsl, bool pixel, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error = nullptr);

// Same, but for a shader that must always be compiled (the renderer's own blit passes).
inline uint64_t HashSource(const std::string& text)
{
    uint64_t h = 0xCBF29CE484222325ull;
    for (unsigned char c : text)
        h = (h ^ c) * 0x100000001B3ull;
    return h;
}
}
