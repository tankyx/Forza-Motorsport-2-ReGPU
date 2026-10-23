#!/bin/sh
# One-time setup: builds the game files from your own disc and unpacks your title update,
# the same steps the Windows installer runs (py/setup_game.py, plain Python 3).
#
#   ./setup.sh <disc.iso> <title update file> [dlc folder]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
HOME_DIR=${FM2_LINUX_HOME:-$HOME/.fm2-linux}
COMPAT=${FM2_COMPATDATA:-$HOME_DIR/compatdata}
DOCS="$COMPAT/pfx/drive_c/users/steamuser/Documents/fm2"

[ $# -ge 2 ] || { echo "usage: ./setup.sh <disc.iso> <title update file> [dlc folder]" >&2; exit 1; }
command -v python3 >/dev/null || { echo "setup.sh: python3 is needed" >&2; exit 1; }

echo "Checking the disc..."
python3 "$HERE/py/setup_game.py" check-iso "$1" || { echo "setup.sh: that disc image isn't the European Forza Motorsport 2 this build needs" >&2; exit 1; }
echo "Checking the title update..."
python3 "$HERE/py/setup_game.py" check-tu "$2" || { echo "setup.sh: that file isn't the title update this build needs" >&2; exit 1; }

mkdir -p "$DOCS"
echo "Building the game files (a few minutes)..."
python3 "$HERE/py/setup_game.py" install "$1" "$HERE/game" "$2" "${3:--}" "$DOCS"
echo "Done. Start the game with ./play.sh"
