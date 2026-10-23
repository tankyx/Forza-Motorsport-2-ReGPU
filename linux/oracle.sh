#!/bin/sh
# Runs the same guest frame through both backends and diffs the two pictures.
#
#   linux/oracle.sh 700 [timeout-seconds]
#
# The D3D12 build under Proton is the oracle: it is the reference the Vulkan backend has to match.
# Both runs are headless, both skip the intro, so guest frame N is the same screen on both sides,
# and out/oracle/ ends up with d3d12-<N>.png, vulkan-<N>.png, diff-<N>.png and a line of numbers.
#
# Env passes through as everywhere else; FM2_TOUR can drive the menus to a deeper screen.
set -e
LINUX_DIR=$(cd "$(dirname "$0")" && pwd)
FRAME=${1:-700}
SECONDS_LIMIT=${2:-45}
INSTALL=${FM2_INSTALL:-${FM2_LINUX_HOME:-$HOME/.fm2-linux}/install}
OUT=$INSTALL/out/oracle
mkdir -p "$OUT"

run() # name exe extra-env...
{
    name=$1; exe=$2; shift 2
    rm -f "$INSTALL/frame.png" "$INSTALL"/out/vk_frame_*.png
    env FM2_EXE="$exe" FM2_HEADLESS=1 FM2_SKIP_INTRO=1 FM2_AUTOSTART=1 FM2_NO_LAUNCHER=1 \
        FM2_VK_DRAWS=1 "$@" \
        timeout "$SECONDS_LIMIT" "$LINUX_DIR/play.sh" >/tmp/oracle-$name.log 2>&1 || true
    pkill -f "[f]$exe" 2>/dev/null || true
    if [ "$name" = d3d12 ]; then
        [ -f "$INSTALL/frame.png" ] && cp "$INSTALL/frame.png" "$OUT/d3d12-$FRAME.png"
    else
        last=$(ls "$INSTALL"/out/vk_frame_*.png 2>/dev/null | tail -1)
        [ -n "$last" ] && cp "$last" "$OUT/vulkan-$FRAME.png"
    fi
}

echo "oracle: d3d12 frame $FRAME"
run d3d12 fm2.exe FM2_DUMP_FRAME="$FRAME"
echo "oracle: vulkan frame $FRAME"
run vulkan fm2-vk.exe FM2_RENDERER=vulkan FM2_VK_DUMP_FRAME="$FRAME"

python3 - "$OUT" "$FRAME" <<'PY'
import sys
from PIL import Image, ImageChops
out, frame = sys.argv[1], sys.argv[2]
a_path, b_path = f"{out}/d3d12-{frame}.png", f"{out}/vulkan-{frame}.png"
try:
    a, b = Image.open(a_path).convert("RGB"), Image.open(b_path).convert("RGB")
except FileNotFoundError as e:
    sys.exit(f"oracle: missing {e.filename}")
# One backend dumps the resolved target, the other the EDRAM surface; compare at a common size.
w, h = a.size
b = b.resize(a.size) if b.size != a.size else b
d = ImageChops.difference(a, b)
differing = sum(1 for p in d.get_flattened_data() if max(p) > 8)
side = Image.new("RGB", (w, h * 2 + 8), (0, 0, 0))
side.paste(a, (0, 0))
side.paste(b, (0, h + 8))
side.save(f"{out}/side-{frame}.png")
d.save(f"{out}/diff-{frame}.png")
print(f"oracle: {w}x{h} vs {Image.open(b_path).size}, {differing} px differ ({100.0 * differing / (w * h):.2f}%)")
print(f"oracle: {out}/side-{frame}.png  (d3d12 top, vulkan bottom)")
PY
