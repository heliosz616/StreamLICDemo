# StreamLIC: Quantized Model and U50 Demo

StreamLIC is a learned image compression system designed for stream processing on FPGA. This repository provides an Alveo U50 encoder and decoder, Linux host applications, matched model parameters and entropy tables, and the quantization-aware model used for training experiments.

The encoder produces `.slc` bitstreams from RGB images. Neural transforms and autoregressive parameter estimation execute on the FPGA; the host performs arithmetic encoding and bitstream assembly. The decoder combines host-side hyperlatent decoding with FPGA entropy decoding and image synthesis.

This demo supports research and engineering evaluation of StreamLIC on the Alveo U50. It provides an end-to-end workflow for encoding images, reconstructing them from compressed bitstreams, and measuring bitrate, reconstruction quality, latency and throughput. The supplied FPGA images and model parameters define the configuration used for these experiments.

## Hardware and software

| Item | Configuration |
|---|---|
| Accelerator | AMD/Xilinx Alveo U50 |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` |
| Host | Linux x86-64 |
| FPGA build toolchain | Vitis 2022.1 |
| Reference XRT version | 2.13.466 |
| Input | 1280 × 768, 8-bit RGB PNG or binary P6 PPM |
| Quantization | 12-bit weights and activations |
| Entropy tables | 24-bit CDF precision |

Install the U50 platform and a compatible XRT runtime before running the demo. The supplied host binaries require `libxrt_coreutil.so.2`, `libpng16.so.16`, the GNU C++ runtime and standard Linux libraries. The decoder also requires `libcrypto.so.1.1`; install the OpenSSL 1.1 runtime library on the host. Vitis is not needed to run the supplied FPGA images.

## Quick start

```bash
git clone https://github.com/heliosz616/StreamLIC-Quantized-Implementation.git
cd StreamLIC-Quantized-Implementation
./streamlic doctor
mkdir -p outputs
./streamlic encode examples/test_pattern.png outputs/test.slc
./streamlic decode outputs/test.slc outputs/reconstruction.png \
  --original examples/test_pattern.png --report outputs/result.json
```

The example is a synthetic test pattern. Prepare inputs at 1280 × 768. Encoding and decoding program different FPGA images and should run sequentially on a shared device.

Use `--device N` to select an XRT device and `--xrt-settings FILE` for a non-default XRT setup script. Each command provides `--help`. Use the supplied FPGA images with their matching encoder parameters, decoder parameters and CDF bundle.

## Repository layout

```text
streamlic                 Runtime command-line entry point
runtime/bin/              Linux x86-64 host executables
runtime/xclbin/           Encoder and decoder FPGA images
runtime/parameters/       Matched weights, scales, biases and CDF tables
runtime/manifest.json     Hardware, ABI and model configuration
scripts/                  Runtime benchmark script
models/                   Quantized model archive and PDF parameters
training/                 QAT model, integer reference and parameter export
examples/                 Synthetic RGB input
```

## Training and quantization

See [training/README.md](training/README.md) for the PyTorch model, quantization controls, parameter export and golden reference workflow. The Python tools run independently of XRT using the supplied quantized model archive.

## Evaluation

For a directory containing at least 64 supported images:

```bash
./streamlic benchmark encode images outputs/encoded --count 64
./streamlic benchmark decode outputs/encoded outputs/decoded \
  --original-dir images --report outputs/benchmark.json
```

The encoder benchmark monitors device temperature and resets the selected U50 after the run. Run it on an exclusively allocated card. See [runtime usage](docs/runtime.md) for options and measurement boundaries.

Reference U50 throughput:

| Measurement boundary | Throughput |
|---|---:|
| Encoder, completed `.slc` output | 32.8–33.5 frames/s |
| Decoder, reconstructed RGB downloaded | 29.4 frames/s |
| Decoder, PNG output and RGB PSNR | 29.2 frames/s |

Encoder measurements use four slots, batches of four frames and four CPU codec workers. Decoder measurements use six slots and four PNG workers. Implemented kernel clocks are recorded in `runtime/manifest.json`.

## Reference

[StreamLIC: A Lightweight Learned Image Compression Model for Stream Processing on FPGA](https://doi.org/10.1109/VCIP67698.2025.11396907), VCIP 2025.

## License

The source code, model parameters and FPGA binaries in this repository are licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) for noncommercial use.
