#include "gpu/vk/vk_loader.h"
#include "gpu/vk/vk_shader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#if defined(_WIN32)
#define popen _popen
#define pclose _pclose
#else
#include <unistd.h>
#endif

namespace vk
{
namespace
{
std::string TempPath(const char* ext)
{
    char dir[512] = {};
#if defined(_WIN32)
    const char* tmp = getenv("TEMP");
    snprintf(dir, sizeof dir, "%s\\fm2vkXXXXXX", tmp ? tmp : ".");
#else
    snprintf(dir, sizeof dir, "/tmp/fm2vkXXXXXX");
    if (int fd = mkstemp(dir); fd >= 0)
        close(fd);
#endif
    return std::string(dir) + ext;
}

bool ReadAll(const std::string& path, std::vector<uint32_t>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<char> bytes{ std::istreambuf_iterator<char>(f), {} };
    if (bytes.empty() || (bytes.size() & 3))
        return false;
    out.resize(bytes.size() / 4);
    memcpy(out.data(), bytes.data(), bytes.size());
    return true;
}
}

bool CompileHlsl(const std::string& hlsl, bool pixel, uint64_t key, const std::filesystem::path& cacheDir,
    std::vector<uint32_t>& spirv, std::string* error)
{
    if (!cacheDir.empty())
    {
        const auto cached = cacheDir / "spv" / (std::to_string(key) + ".spv");
        if (ReadAll(cached.string(), spirv))
            return true;
    }

    const std::string hlslPath = TempPath(".hlsl");
    const std::string spvPath = TempPath(".spv");
    {
        std::ofstream f(hlslPath, std::ios::binary);
        if (!f)
            return false;
        f.write(hlsl.data(), std::streamsize(hlsl.size()));
    }
    const char* dxc = getenv("FM2_DXC");
    char command[1024];
    snprintf(command, sizeof command,
        "%s -HV 2018 -T %s -E main -spirv -fvk-use-dx-layout -fspv-target-env=vulkan1.2%s"
        " -fvk-b-shift 0 0 -fvk-t-shift 100 0 -fvk-s-shift 200 0 -fvk-u-shift 300 0"
        " -Fo \"%s\" \"%s\" 2>&1",
        dxc ? dxc : "dxc", pixel ? "ps_6_0" : "vs_6_0", pixel ? "" : " -D FM2_VERTEX_STAGE", spvPath.c_str(),
        hlslPath.c_str());
    bool ok = false;
    if (FILE* pipe = popen(command, "r"))
    {
        char buffer[512];
        std::string log;
        while (fgets(buffer, sizeof buffer, pipe))
            log += buffer;
        ok = pclose(pipe) == 0;
        if (!ok && error)
            *error = log;
    }
    if (ok)
        ok = ReadAll(spvPath, spirv);
    std::error_code ignored;
    std::filesystem::remove(spvPath, ignored);
    std::filesystem::remove(hlslPath, ignored);

    if (ok && !cacheDir.empty())
    {
        std::filesystem::create_directories(cacheDir / "spv");
        std::ofstream out(cacheDir / "spv" / (std::to_string(key) + ".spv"), std::ios::binary);
        out.write(reinterpret_cast<const char*>(spirv.data()), std::streamsize(spirv.size() * 4));
    }
    return ok;
}
}
