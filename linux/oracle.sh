#!/bin/sh
# Runs the same SCREEN through both backends and diffs the two pictures.
#
#   linux/oracle.sh [frame] [tour] [d3d12-seconds] [vulkan-seconds]
#
# Frame numbers do not line up between the backends - the menus advance on their own timers and the
# two renderers run at different speeds - so the capture is driven by the tour: the DUMP step asks
# both backends for a screenshot of whatever screen they are on, and the tour walks both to that
# screen the same way. The guest frame passed in is only the fallback for a run with no tour.
#
# out/oracle/ ends up with d3d12.png, vulkan.png, side.png (reference on top), diff.png and a line of
# numbers. FM2_TOUR passes through if set, so a deeper screen can be reached by hand.
set -e
FRAME=${1:-700}
TOUR=${2:-${FM2_TOUR:-"WAIT:22,DUMP,WAIT:2"}}
D3D12_SECONDS=${3:-120}
VULKAN_SECONDS=${4:-170}
LINUX_DIR=$(cd "$(dirname "$0")" && pwd)
INSTALL=${FM2_INSTALL:-${FM2_LINUX_HOME:-$HOME/.fm2-linux}/install}
OUT=$INSTALL/out/oracle
mkdir -p "$OUT"
# Stale captures from an earlier run are worse than none: they silently compare two different
# screens. Both are removed up front and each side must produce a fresh one.
rm -f "$OUT/d3d12.png" "$OUT/vulkan.png" "$OUT/side.png" "$OUT/diff.png"

run() # name exe extra-env...
{
    name=$1; exe=$2; shift 2
    rm -f "$INSTALL/frame.png" "$INSTALL"/out/vk_frame_*.png "$INSTALL"/out/capture.request
    seconds=$D3D12_SECONDS
    [ "$name" = vulkan ] && seconds=$VULKAN_SECONDS
    # No FM2_AUTOSTART: it drives the menus by itself, and the two backends advance at different
    # rates, which is exactly how a comparison lands on two different screens. The tour drives
    # both instead, so the dump happens on the same screen.
    env FM2_EXE="$exe" FM2_HEADLESS=1 FM2_SKIP_INTRO=1 FM2_NO_LAUNCHER=1 \
        FM2_TOUR="$TOUR" FM2_VK_DRAWS=1 "$@" \
        timeout "$seconds" "$LINUX_DIR/play.sh" >/tmp/oracle-$name.log 2>&1 || true
    sleep 2
    pkill -f "[f]$exe" 2>/dev/null || true
    if [ "$name" = d3d12 ]; then
        if [ -f "$INSTALL/frame.png" ]; then cp "$INSTALL/frame.png" "$OUT/d3d12.png"; return; fi
    else
        last=$(ls "$INSTALL"/out/vk_frame_*.png 2>/dev/null | tail -1)
        if [ -n "$last" ]; then cp "$last" "$OUT/vulkan.png"; return; fi
    fi
    echo "oracle: $name produced no capture (see /tmp/oracle-$name.log)" >&2
}

echo "oracle: tour '$TOUR'"
echo "oracle: d3d12"
run d3d12 fm2.exe FM2_DUMP_FRAME="$FRAME"
echo "oracle: vulkan"
run vulkan fm2-vk.exe FM2_RENDERER=vulkan FM2_VK_DUMP_FRAME="$FRAME"

python3 - "$OUT" <<'PY'
import sys
from PIL import Image, ImageChops
out = sys.argv[1]
try:
    a = Image.open(f"{out}/d3d12.png").convert("RGB")
    b = Image.open(f"{out}/vulkan.png").convert("RGB")
except FileNotFoundError as e:
    sys.exit(f"oracle: missing {e.filename}")
w, h = a.size
b = b.resize(a.size) if b.size != a.size else b
d = ImageChops.difference(a, b)
differing = sum(1 for p in d.get_flattened_data() if max(p) > 8)
side = Image.new("RGB", (w, h * 2 + 8), (0, 0, 0))
side.paste(a, (0, 0))
side.paste(b, (0, h + 8))
side.save(f"{out}/side.png")
d.save(f"{out}/diff.png")
print(f"oracle: {w}x{h}, {differing} px differ ({100.0 * differing / (w * h):.2f}%)")
print(f"oracle: {out}/side.png  (d3d12 top, vulkan bottom)")
PY
