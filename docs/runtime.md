# U50 runtime guide

Run commands from the repository root. `./streamlic --help` lists commands, and each command accepts `--help`. Hardware and library requirements are listed in the [project README](../README.md#environment).

## Device selection

```bash
./streamlic doctor --device 0
```

`doctor` checks the runtime files and visible U50 device. Runtime commands default to device 0 and `/opt/xilinx/xrt/setup.sh`. Use `--device N` and `--xrt-settings FILE` to select another device or setup script.

## Single-image encoding

```bash
./streamlic encode examples/test_pattern.png outputs/test.slc
```

Input images are 1280 × 768 RGB PNG or binary P6 PPM. The encoder performs neural transforms and parameter estimation on the FPGA, then arithmetic encoding and bitstream assembly on the host. Each `.slc` file contains one hyperlatent payload and four main-latent entropy streams.

`--slots`, `--batch-size` and `--codec-workers` default to one for single-image encoding.

## Reconstruction and quality measurement

```bash
./streamlic decode outputs/test.slc outputs/reconstruction.png \
  --original examples/test_pattern.png --report outputs/result.json
```

| Option | Effect |
|---|---|
| `--original IMAGE` | Compute RGB PSNR against a source PNG/PPM |
| `--report FILE` | Write JSON timing, bitrate and quality results |
| `--frame-timings` | Include per-frame results in the report |
| `--slots N` | Set resident frame-buffer sets; default 1 |
| `--png-workers N` | Set PNG output workers; default 1 |
| `--png-compression N` | Set PNG compression level 0–9; default 1 |
| `--timeout-ms N` | Set per-command FPGA deadline; default 60000 ms |

## Batch evaluation

```bash
./streamlic benchmark encode images outputs/encoded --count 64
./streamlic benchmark decode outputs/encoded outputs/decoded \
  --original-dir images --report outputs/benchmark.json
```

Encoder defaults are 64 frames, four slots, batch size four and four CPU codec workers. Supply enough images for `--count`. `--timeout-seconds` defaults to 300. Temperature monitoring uses Linux hwmon; `--device-bdf` selects an explicit PCI address. The encoder benchmark resets the selected card after execution, including failed runs that reach cleanup.

Decoder defaults are six slots, four PNG workers and PNG compression level 1. `--original-dir DIR` supplies source images for quality measurement. `--no-png` runs reconstruction without writing PNG files.

## Runtime files

The default package resides under `runtime/`. Advanced path options are:

| Option | Expected content |
|---|---|
| `--model-package DIR` | Directory containing `encoder/`, `fpga_decoder/` and `cdf_p24.bin` |
| `--xclbin FILE` | FPGA image for the selected encoder or decoder |
| `--host FILE` | Corresponding host executable |

Use matching FPGA images and parameter files. `runtime/manifest.json` records the supplied configuration.

For a missing-library error, inspect the executable's dependencies:

```bash
ldd runtime/bin/streamlic_encoder_host
ldd runtime/bin/streamlic_fpga_decoder
```

Install the libraries listed in the project README. The decoder uses OpenSSL 1.1 (`libcrypto.so.1.1`). Check the installed FPGA platform with `xbutil examine`.
