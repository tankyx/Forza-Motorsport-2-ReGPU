# Running FM2 on Linux

The game code is already native x64: what stands between this port and Linux is the
Windows API surface, not a CPU emulator. Everything except the renderer now runs on
Linux as it is, and the renderer's Direct3D 12 goes through **vkd3d-proton**, so the
Windows build runs under Proton today at full speed. A native Vulkan backend is the
long-term plan (`FM2-native-scope.md`, phase 8); these scripts are the Proton path.

Measured here (Ryzen 9950X3D, RTX 5090, Proton 11):

| | |
|---|---|
| Race, 2560×1440, uncapped | 133 fps with `FM2_RESOLUTION=2560x1440`, GPU 2.2 ms/frame |
| Race, 1280×720, uncapped | 205 fps average, 1 % low 157, 0.1 % low 139 |
| Menu → race, 1080p, keyboard | works, including the throttle (`W`) |
| Audio | 5.1 at 48 kHz through XAudio2 → FAudio, 0 dropped frames |
| Shader cache | reused from disk, no recompiles after the first start |
| A 12 minute session | 103 200 frames at the 144 fps cap, 4 083 shaders, 1 044 pipelines, 7 088 textures, 0 compile failures, 0 untranslated shaders, 0 frames over 25 ms |

## Requirements

* Proton 11 or Experimental, and Steam installed (the scripts use Proton's own `proton`).
* Your own European disc image, extracted to `game/`:
  `python3 tools/xiso.py extract "<disc>.iso" game`
* `thirdparty/ffmpeg`, `thirdparty/fidelityfx`, `thirdparty/dlss` — see the main README.
* For building: clang/clang-cl, lld, llvm, cmake ≥ 3.25, ninja, python3, git, wine, curl.

## Build

```sh
linux/build.sh          # ~1 min warm, needs the network once to fetch the MSVC/SDK headers
linux/stage.sh          # assembles ~/.fm2-linux/install
linux/play.sh           # windowed
```

`build.sh` cross-compiles the Windows binary with clang-cl and an [xwin](https://github.com/Jake-Shadle/xwin)
sysroot (Microsoft's headers and import libraries, downloaded once). Two Windows-only build
steps are shimmed: `py -3` becomes `python3`, and `fxc` becomes a small `D3DCompile`
wrapper run under Wine (the FSR 1 shaders are compiled by fxc at build time).

## The three DLLs a Windows box already has

`linux/stage.sh` puts them next to `fm2.exe`:

1. **FFmpeg** (`avcodec`, `avutil`, `swresample`) — the XMA audio decoder. The Windows
   installer ships these too.
2. **AMD FidelityFX** (`amd_fidelityfx_loader_dx12`, `amd_fidelityfx_upscaler_dx12`).
3. **Microsoft's `d3dcompiler_47.dll`** — the important one. Wine's built-in compiler is
   vkd3d-shader, which cannot compile the renderer's bindless HLSL
   (`E5017: Unbounded resource arrays`): *every* shader fails and the screen stays black.
   The DLL is redistributable and already on most machines (any Steam game's `directx`
   redist folder); the script finds one, or point it at one with
   `FM2_D3DCOMPILER=/path/to/d3dcompiler_47.dll`. Wine is told to prefer it with
   `WINEDLLOVERRIDES=d3dcompiler_47=n`, which `play.sh` sets.

## Title update or not

This builds the **base disc** build (`FM2_TITLE_UPDATE` off), so no title update is needed.
If you have one (`TU_16L61VA_…`), the release `.exe` from GitHub works instead — unpack it
next to the game files and apply the update:

```sh
python3 tools/stfs.py update "<TU package>" "$FM2_LINUX_HOME/compatdata/pfx/drive_c/users/steamuser/Documents/fm2/update"
```

For a title-update build of your own, configure with `-DFM2_TITLE_UPDATE=ON` (see
`tools/xexmap.py`), as on Windows.

## Debugging and benchmarking

Every `FM2_*` variable from the main README's table passes through:

```sh
FM2_PERF=1 linux/play.sh                            # fps, renderer time by section, GPU time
FM2_HEADLESS=1 FM2_DUMP_AT=40 linux/play.sh         # dump the first frame after 40 s to frame.png
FM2_DEBUG=1 linux/play.sh                           # then tools/fm2ctl.py status|press|shot|log ...
linux/bench.sh                                      # the port's tour benchmark, in-race numbers
linux/bench.sh FM2_PERF=1                           # the same with the per-section breakdown
```

`tools/fm2ctl.py` works unchanged over the debug server: `press START`, `shot`, `log`,
`peek`, `switch`, and `quit`.

## Known gaps on this path

* **Frame-time hitches while the game loads assets**: an occasional frame of 20–55 ms mid-race.
  Measured with `FM2_PERF=1` they show up as `draw 0.2 ms` of a 34 ms frame with a guest
  thread starting at the same moment: it is the title's own file loading and decompression
  (the mesh archives are deflated in the track and car zips), not the renderer and not
  vkd3d. The one-off renderer spike at the start of a race (208 resources, ~20 ms of texture
  setup in one frame) is covered by the port's loading screen.
* **Tested with a controller**: an Xbox Series pad through Proton's XInput works — a 12 minute
  session (menus, car and track selection, races) with 103 200 frames, A/B/X/Y, START, the
  D-pad, RB, both triggers through their full range and both sticks through theirs, recorded
  with `FM2_RECORD_INPUT` and checked sample by sample. The keyboard mapping still works
  alongside it (`W`/`S` pedals, `A`/`D` steer, arrows + `Enter` for menus).
* **Still untested**: DLC packs, the title-update build, AMD/Steam Deck GPUs, ultrawide,
  LB/BACK/stick clicks (nothing in that session used them).
* DLSS is not available this way (it needs NVIDIA's NGX, which vkd3d-proton does not
  provide); FSR is in the source but hidden from the launcher.
