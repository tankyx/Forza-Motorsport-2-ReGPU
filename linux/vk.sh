#!/bin/sh
# Builds and runs the native renderer's present pass natively against the real GPU:
#   linux/vk.sh                # writes out/vk_present.png
#
# Needs the Vulkan loader, the Vulkan headers (fetched once into $FM2_LINUX_HOME/vkinclude)
# and dxc on PATH or in FM2_DXC. Nothing here touches the Windows build.
set -e
LINUX_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$LINUX_DIR")
HOME_DIR=${FM2_LINUX_HOME:-$HOME/.fm2-linux}
INCLUDE=$HOME_DIR/vkinclude
OUT=$ROOT/out

if [ ! -f "$INCLUDE/vulkan/vulkan.h" ]; then
    echo "== fetching the Vulkan headers"
    mkdir -p "$HOME_DIR" "$INCLUDE"
    curl -sL -o "$HOME_DIR/vkheaders.tar.gz" \
        https://github.com/KhronosGroup/Vulkan-Headers/archive/refs/tags/vulkan-sdk-1.4.363.0.tar.gz
    tar xzf "$HOME_DIR/vkheaders.tar.gz" -C "$HOME_DIR"
    cp -r "$HOME_DIR"/Vulkan-Headers-*/include/* "$INCLUDE/"
fi

if [ -z "$FM2_DXC" ]; then
    command -v dxc > /dev/null || {
        echo "linux/vk.sh: set FM2_DXC=/path/to/dxc (the DirectXShaderCompiler release binary)" >&2
        exit 1
    }
fi

mkdir -p "$OUT"
g++ -O2 -std=c++20 -g -I "$ROOT/src" -I "$INCLUDE" \
    "$ROOT/tools/vk_present_test.cpp" "$ROOT/src/gpu/vk/vk_context.cpp" "$ROOT/src/gpu/vk/vk_shader.cpp" \
    "$ROOT/src/gpu/vk/vk_blit.cpp" "$ROOT/src/png.cpp" \
    -lvulkan -o "$OUT/vk_present_test"
g++ -O2 -std=c++20 -g -I "$ROOT/src" -I "$INCLUDE" \
    "$ROOT/tools/vk_pipelines_test.cpp" "$ROOT/src/gpu/vk/vk_context.cpp" "$ROOT/src/gpu/vk/vk_state.cpp" \
    -lvulkan -o "$OUT/vk_pipelines_test"

g++ -O2 -std=c++20 -g -I "$ROOT/src" -I "$INCLUDE" \
    "$ROOT/tools/vk_texture_test.cpp" "$ROOT/src/gpu/vk/vk_context.cpp" "$ROOT/src/gpu/vk/vk_shader.cpp" \
    "$ROOT/src/gpu/vk/vk_texture.cpp" \
    -lvulkan -o "$OUT/vk_texture_test"

echo "== running"
# FM2_SPV_DIR: a directory of SPIR-V modules (vs_*.spv / ps_*.spv) to also build pipelines
# from, which is where a first start's cost comes from. FM2_SPV_COUNT limits how many.
FM2_DXC=${FM2_DXC:-dxc} "$OUT/vk_present_test" "$OUT/vk_present.png" "${FM2_SPV_DIR:-$OUT/spv}" "${FM2_SPV_COUNT:-200}"
echo "wrote $OUT/vk_present.png"
