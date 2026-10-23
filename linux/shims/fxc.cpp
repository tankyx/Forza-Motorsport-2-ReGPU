// fxc stand-in for Linux-hosted builds: enough of the command line for
// tools/embed_hlsl.py (/nologo /O3 /T <target> /E <entry> /Fo <out> <in.hlsl>), compiled
// against Microsoft's d3dcompiler_47 and run under Wine (linux/build.sh builds it and
// installs an "fxc" wrapper next to it). Without it the FSR 1 compute shaders cannot be
// compiled, and the build stops at "Compiling FSR 1 shaders".
#include <windows.h>
#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool g_skipOpt;

static char* Slurp(const char* path, SIZE_T* size)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n)
    {
        fclose(f);
        free(buf);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    *size = (SIZE_T)n;
    return buf;
}

int main(int argc, char** argv)
{
    const char* src = NULL, *out = NULL, *entry = NULL, *target = "cs_5_1";
    // Flags are matched by name: a unix input path also starts with '/'.
    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        if (a[0] != '/' && a[0] != '-')
        {
            if (!src)
                src = a;
            continue;
        }
        const char* name = a + 1;
        if (!_stricmp(name, "E") && i + 1 < argc) { entry = argv[++i]; continue; }
        if (!_stricmp(name, "T") && i + 1 < argc) { target = argv[++i]; continue; }
        if (!_stricmp(name, "Fo") && i + 1 < argc) { out = argv[++i]; continue; }
        if (!_strnicmp(name, "Fo", 2) && name[2]) { out = name + 2; continue; }
        if (!_stricmp(name, "Od")) { g_skipOpt = true; continue; }
        if (!_stricmp(name, "nologo") || !_stricmp(name, "O1") || !_stricmp(name, "O2") || !_stricmp(name, "O3") ||
            !_stricmp(name, "Zi") || !_stricmp(name, "Gec") || !_stricmp(name, "WX") || !_stricmp(name, "Vn"))
            continue;
        if (!src && strchr(name, '.'))
            src = a;
    }
    if (!entry)
        entry = "main";

    SIZE_T size = 0;
    char* text = src ? Slurp(src, &size) : NULL;
    if (!text)
    {
        fprintf(stderr, "fxc: cannot read %s\n", src ? src : "(no input)");
        return 2;
    }

    // The renderer's own flags: bindless, and unoptimised unless asked.
    UINT flags = D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES | D3DCOMPILE_ENABLE_STRICTNESS;
    flags |= g_skipOpt ? D3DCOMPILE_SKIP_OPTIMIZATION : D3DCOMPILE_OPTIMIZATION_LEVEL3;
    ID3DBlob* code = NULL, *err = NULL;
    HRESULT hr = D3DCompile(text, size, src, NULL, NULL, entry, target, flags, 0, &code, &err);
    free(text);
    if (FAILED(hr))
    {
        if (err)
            fprintf(stderr, "fxc: %.*s\n", (int)err->GetBufferSize(), (const char*)err->GetBufferPointer());
        else
            fprintf(stderr, "fxc: D3DCompile failed %08lX\n", (unsigned long)hr);
        return 1;
    }

    if (!out)
        out = "output.cso";
    FILE* f = fopen(out, "wb");
    if (!f)
    {
        fprintf(stderr, "fxc: cannot write %s\n", out);
        return 2;
    }
    fwrite(code->GetBufferPointer(), 1, code->GetBufferSize(), f);
    fclose(f);
    code->Release();
    return 0;
}
