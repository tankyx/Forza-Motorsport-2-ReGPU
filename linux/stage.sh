#!/bin/sh
# Assembles a runnable folder: fm2.exe plus the DLLs a Windows install gets from the
# installer or from Windows itself.
#
#   linux/stage.sh                 # into $FM2_LINUX_HOME/install
#   FM2_D3DCOMPILER=/path/d3dcompiler_47.dll linux/stage.sh
#
# d3dcompiler_47.dll must be MICROSOFT's (4-5 MB). Wine's built-in one is vkd3d-shader,
# which cannot compile the renderer's bindless HLSL ("Unbounded resource arrays"): every
# shader fails. The DLL is redistributable and ships with Windows and with DirectX
# redistributables, so it is normally already on this machine — in a Steam game's folder,
# for instance. This script looks in the usual places if FM2_D3DCOMPILER is not set.
set -e
LINUX_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$LINUX_DIR")
HOME_DIR=${FM2_LINUX_HOME:-$HOME/.fm2-linux}
INSTALL=${FM2_INSTALL:-$HOME_DIR/install}

die() { printf 'linux/stage.sh: %s\n' "$*" >&2; exit 1; }
[ -f "$HOME_DIR/build/fm2.exe" ] || die "no fm2.exe yet: run linux/build.sh"

mkdir -p "$INSTALL"
cp "$HOME_DIR/build/fm2.exe" "$INSTALL/"

# FFmpeg (LGPL, shared): the XMA audio decoder.
cp "$ROOT"/thirdparty/ffmpeg/bin/avcodec-*.dll "$ROOT"/thirdparty/ffmpeg/bin/avutil-*.dll \
   "$ROOT"/thirdparty/ffmpeg/bin/swresample-*.dll "$INSTALL/" 2>/dev/null ||
    die "thirdparty/ffmpeg/bin is missing the DLLs (see the README)"

# AMD FidelityFX (MIT): the DX12 upscaler the launcher knows about.
cp "$ROOT"/thirdparty/fidelityfx/Kits/FidelityFX/signedbin/amd_fidelityfx_loader_dx12.dll \
   "$ROOT"/thirdparty/fidelityfx/Kits/FidelityFX/signedbin/amd_fidelityfx_upscaler_dx12.dll "$INSTALL/" ||
    die "thirdparty/fidelityfx is missing (see the README)"

# Microsoft's shader compiler.
if [ -z "$FM2_D3DCOMPILER" ]; then
    for candidate in "$ROOT"/thirdparty/d3dcompiler_47.dll $(find "$HOME/.steam/steam/steamapps/common" \
        "$HOME/.local/share/Steam/steamapps/common" -maxdepth 5 -iname d3dcompiler_47.dll 2>/dev/null); do
        # Wine's built-in compiler is ~1 MB and cannot do this port's shaders; Microsoft's is 4-5 MB.
        if [ -f "$candidate" ] && [ "$(stat -c%s "$candidate")" -gt 2000000 ]; then
            FM2_D3DCOMPILER=$candidate
            break
        fi
    done
fi
[ -n "$FM2_D3DCOMPILER" ] || die "no Microsoft d3dcompiler_47.dll found: set FM2_D3DCOMPILER=/path/to/d3dcompiler_47.dll"
FM2_D3DCOMPILER=$(readlink -f "$FM2_D3DCOMPILER")
[ "$(stat -c%s "$FM2_D3DCOMPILER")" -gt 2000000 ] ||
    die "$FM2_D3DCOMPILER is Wine's built-in shader compiler, not Microsoft's (which is 4-5 MB)"
cp "$FM2_D3DCOMPILER" "$INSTALL/d3dcompiler_47.dll"

# The game files (your own disc).
[ -d "$ROOT/game" ] || die "no game/ folder: run 'python3 tools/xiso.py extract <your disc>.iso game'"
[ -e "$INSTALL/game" ] || ln -s "$ROOT/game" "$INSTALL/game"

echo "staged $INSTALL"
ls -1 "$INSTALL"
