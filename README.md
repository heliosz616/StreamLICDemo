# StreamLICDemo

StreamLIC is a learned image compression system designed for stream processing on FPGA. This repository provides a runnable Alveo U50 demonstration, the quantization-aware training model, quantized parameters, and a Python integer reference implementation.

The demo supports research and engineering evaluation through end-to-end image encoding and reconstruction, with bitrate, RGB PSNR, latency and throughput measurements. The U50 workflow uses the supplied host executables and FPGA images. The Python workflow supports model experiments, parameter export and golden reference generation.

## Repository contents

| Path | Contents |
|---|---|
| `streamlic` | Command-line entry point for the U50 demo |
| `runtime/bin/` | Linux x86-64 encoder and decoder executables |
| `runtime/xclbin/` | Encoder and decoder FPGA images |
| `runtime/parameters/` | Packed weights, scales, biases and entropy tables |
| `runtime/manifest.json` | Platform configuration and kernel clocks |
| `models/` | Quantized model archive and learned PDF parameters |
| `training/` | QAT model, integer reference, parameter export and tests |
| `scripts/` | Runtime benchmark script |
| `examples/` | Synthetic RGB test image |

## U50 demo

### Environment

| Component | Configuration |
|---|---|
| Accelerator | AMD/Xilinx Alveo U50 |
| Platform | `xilinx_u50_gen3x16_xdma_5_202210_1` |
| Host | Linux x86-64 |
| Reference XRT version | 2.13.466 |
| FPGA image build toolchain | Vitis 2022.1 |
| Input | 1280 × 768, 8-bit RGB PNG or binary P6 PPM |
| Quantization | 12-bit weights and activations |
| Entropy tables | 24-bit CDF precision |

Install the U50 platform and compatible XRT runtime. The host executables require `libxrt_coreutil.so.2`, `libpng16.so.16`, the GNU C++ runtime and standard Linux libraries. The decoder additionally requires `libcrypto.so.1.1` from OpenSSL 1.1. The supplied executables and xclbins run directly, without a Vitis build.

### Encode and reconstruct an image

```bash
git clone https://github.com/heliosz616/StreamLICDemo.git
cd StreamLICDemo
./streamlic doctor
mkdir -p outputs
./streamlic encode examples/test_pattern.png outputs/test.slc
./streamlic decode outputs/test.slc outputs/reconstruction.png \
  --original examples/test_pattern.png --report outputs/result.json
```

The encoder writes a `.slc` bitstream. The decoder writes the reconstructed PNG; `--original` and `--report` enable RGB PSNR and a JSON evaluation report. Use images prepared at 1280 × 768. Encoding and decoding program separate FPGA images and run sequentially on a shared card.

Select a device with `--device N`, or an XRT setup script with `--xrt-settings FILE`. See [runtime usage](docs/runtime.md) for command options and dependencies.

### Batch evaluation

For a directory containing at least 64 supported images:

```bash
./streamlic benchmark encode images outputs/encoded --count 64
./streamlic benchmark decode outputs/encoded outputs/decoded \
  --original-dir images --report outputs/benchmark.json
```

The encoder benchmark monitors temperature and resets the selected U50 after execution. Use an exclusively allocated card.

Reference U50 throughput:

| Measurement | Throughput |
|---|---:|
| Encoder, completed `.slc` output | 32.8–33.5 frames/s |
| Decoder, reconstructed RGB downloaded | 29.4 frames/s |
| Decoder, PNG output and RGB PSNR | 29.2 frames/s |

The encoder configuration uses four slots, batches of four frames and four CPU codec workers. The decoder configuration uses six slots and four PNG workers. Kernel clocks are recorded in `runtime/manifest.json`.

## Training and Python reference

From the repository root, install the Python package in a Python 3.10+ environment:

```bash
python -m pip install -r training/requirements.txt
python -m pip install -e training --no-deps
```

The [training and reference guide](training/README.md) covers:

- Quantization-aware model controls and tensor conventions.
- Quantized parameter and CDF export.
- Integer encode/decode with the supplied weights.
- Intermediate golden tensor generation and regression tests.

The Python workflow runs independently of XRT. Its reference encoder and decoder implement W12/A12 arithmetic and the `.slc` bitstream format used by the demo.

## Reference

[StreamLIC: A Lightweight Learned Image Compression Model for Stream Processing on FPGA](https://doi.org/10.1109/VCIP67698.2025.11396907), VCIP 2025.

## License

The source code, model parameters and FPGA binaries are licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) for noncommercial use.
