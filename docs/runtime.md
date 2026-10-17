# Runtime usage

`./streamlic --help` lists commands; each command accepts `--help`.

## Encoder

`encode INPUT OUTPUT` accepts RGB PNG or P6 PPM at 1280 × 768. `--slots`, `--batch-size` and `--codec-workers` default to one for a single image. The encoder writes a compact `.slc` container with one hyperlatent payload and four independently coded main-latent streams.

`benchmark encode INPUT_DIR OUTPUT_DIR` defaults to 64 frames, four slots, batch size four and four CPU codec workers. Supply enough input files for the requested count. Its default timeout is 300 seconds. Device temperature monitoring requires Linux hwmon access; `--device-bdf` selects an explicit PCI address when needed. The benchmark wrapper resets the selected card after execution, including failed runs that reach cleanup.

## Decoder

`decode INPUT OUTPUT` writes an RGB PNG. `--original IMAGE` enables RGB PSNR, and `--report FILE` writes timing, bitrate and quality measurements. `--frame-timings` adds per-frame records. The default per-command FPGA timeout is 60000 ms. PNG compression defaults to level 1.

`benchmark decode INPUT_DIR OUTPUT_DIR` processes a directory of bitstreams. `--original-dir DIR` supplies corresponding source images; `--no-png` excludes PNG output.

## Packages and dependencies

`--model-package DIR` selects a directory containing `encoder/`, `fpga_decoder/` and `cdf_p24.bin`. `--xclbin FILE` and `--host FILE` override the corresponding image and executable.

`./streamlic doctor` checks the runtime files and visible U50 device. Use an encode/decode round trip to validate an installed accelerator.

If a host executable cannot load a shared library, inspect it with `ldd runtime/bin/streamlic_encoder_host` or `ldd runtime/bin/streamlic_fpga_decoder`. Install the runtime libraries listed in the project README. Check the platform reported by `xbutil examine` against `runtime/manifest.json`.
