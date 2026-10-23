#!/bin/sh
# The port's own benchmark — the FM2_TOUR script drives the menus into a Sunset race with
# 8 cars, holds the throttle for 25 s, and FM2_BENCHMARK writes the report.
#
#   linux/bench.sh                   # report on stdout
#   linux/bench.sh FM2_PERF=1        # extra FM2_* / VKD3D_CONFIG=... for one run
#
# Needs a shader cache already built (linux/play.sh once), otherwise the first run compiles
# thousands of shaders instead of measuring the race.
set -e
LINUX_DIR=$(cd "$(dirname "$0")" && pwd)
HOME_DIR=${FM2_LINUX_HOME:-$HOME/.fm2-linux}
INSTALL=${FM2_INSTALL:-$HOME_DIR/install}
COMPAT=${FM2_COMPATDATA:-$HOME_DIR/compatdata}
PROTON=${PROTON:-"$HOME/.local/share/Steam/steamapps/common/Proton - Experimental/proton"}

mkdir -p "$COMPAT" "$INSTALL/out"
rm -f "$INSTALL/out/benchmark.txt" "$INSTALL/fm2.log"
cd "$INSTALL"
env FM2_HEADLESS=1 FM2_NO_LAUNCHER=1 FM2_PROFILE=Benchmark FM2_MAX_FPS=360 FM2_VSYNC=0 \
    FM2_BENCHMARK=out/benchmark.txt \
    FM2_TOUR="WAIT:100,START,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,A,IDLE,WAIT:2,A,IDLE,WAIT:3,RT:25,QUIT" \
    "$@" env STEAM_COMPAT_CLIENT_INSTALL_PATH="$HOME/.steam/steam" STEAM_COMPAT_DATA_PATH="$COMPAT" \
    WINEDLLOVERRIDES="d3dcompiler_47=n" \
    "$PROTON" run "$INSTALL/fm2.exe" "$INSTALL/game" > bench.log 2>&1 || true

cat "$INSTALL/out/benchmark.txt" 2>/dev/null || { echo "no report; tail of the log:"; tail -20 "$INSTALL/fm2.log"; }
