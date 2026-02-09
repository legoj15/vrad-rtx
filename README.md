# VRAD-RTX

GPU-accelerated lightmap compiler for the Source Engine, built on NVIDIA OptiX and CUDA.

VRAD-RTX is a fork of [Valve's Source SDK 2013](https://github.com/ValveSoftware/source-sdk-2013) that replaces the performance-critical ray tracing stages of **VRAD** (the Source Engine's radiosity lighting tool) with hardware-accelerated equivalents using NVIDIA's OptiX 9.1 RT cores.

## Features

- **GPU Visibility Matrix** — Accelerated VisMatrix generation via OptiX ray tracing
- **GPU Shadow Ray Batching** — Thread-local batched shadow rays for direct lighting, traced on the GPU
- **GPU Radiosity Bounces** — Light bounce calculations offloaded to CUDA with per-thread streams
- **Hardware Profiling** — Built-in system RAM, VRAM, CPU, and GPU usage logging
- **Automated Regression Testing** — PowerShell 7.6+ test scripts with bit-identical and visual parity checks against the original `vrad.exe`
- **CPU Parity** — CPU code path remains fully functional and produces identical results to stock VRAD

## Performance

The GPU path accelerates the most expensive phases of lightmap compilation:

| Phase | Speedup |
|---|---|
| Visibility Matrix | ~6 s faster |
| Direct Lighting | ~16 s faster |
| Radiosity Bounces | ~29 s faster |

Results measured on a map with 100 bounces; actual gains depend on map complexity and GPU hardware.

## Requirements

- **OS:** Windows 10/11 (x64)
- **CPU:** x64 processor; AVX2/FMA3 recommended for `-avx2` mode (Intel Haswell/2013+ or AMD Zen/2017+)
- **GPU:** NVIDIA GPU with RT cores (RTX 20-series or newer)
- **CUDA Toolkit:** 12.x
- **OptiX SDK:** 9.1
- **Compiler:** Visual Studio 2022 (MSVC v143, Windows SDK 10.0.22621.0+)
- **Runtime:** Source SDK Base 2013 Multiplayer (installed via Steam)
- **Testing:** PowerShell 7.6+, Python 3.13+

## Building

1. Clone the repository:
   ```
   git clone https://github.com/legoj15/vrad-rtx.git
   ```

2. Navigate to `src` and generate the Visual Studio solution:
   ```bat
   createallprojects.bat
   ```

3. Open `everything.sln` in Visual Studio 2022 and build the `vrad_dll` project in **Release | x64**.

4. The built `vrad_rtx.exe` and `vrad_dll.dll` are copied to `game/bin/x64/` automatically by the post-build step.

## Usage

Run VRAD-RTX as a drop-in replacement for `vrad.exe`:

```bash
# CPU only (identical to stock VRAD)
vrad_rtx.exe path/to/mapname

# GPU accelerated
vrad_rtx.exe -cuda path/to/mapname
```

All standard VRAD command-line options are supported.

## Testing

The test suite lives in `game/bin/x64/` and validates correctness against the stock compiler:

```powershell
# Quick regression test (reference → control → GPU, with visual diff)
.\test_vrad_optix_quick.ps1

# Full regression test
.\test_vrad_optix.ps1
```

Tests compare BSP lightmaps byte-for-byte against both the original `vrad.exe` and the CPU-only `vrad_rtx.exe` path, with a visual screenshot fallback using `tgadiff.exe` when binary comparison fails.

## Project Structure

```
src/utils/vrad/              # Modified VRAD source
  vrad.cpp                   # Main entry point (timing, GPU flag)
  lightmap.cpp               # Core lightmap sampling & direct lighting
  vismat.cpp                 # Visibility matrix (GPU path)
  direct_lighting_gpu_vrad.cpp  # GPU direct lighting integration
  hardware_profiling.cpp     # System resource monitoring
  vrad_dll.vpc               # VPC project definition
game/bin/x64/                # Runtime & test harness
  test_vrad_optix_quick.ps1  # Quick regression test
  test_vrad_optix.ps1        # Full regression test
  bsp_diff_lightmaps.py      # Lightmap comparison utility
  tgadiff.exe                # Visual diff tool
```

## License

This project is a fork of Valve's Source SDK 2013, licensed under the [SOURCE 1 SDK LICENSE](LICENSE).
