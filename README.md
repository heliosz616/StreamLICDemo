# StreamLICDemo

This repository provides a demo of **[StreamLIC: A Lightweight Learned Image Compression Model for Stream Processing on FPGA](https://doi.org/10.1109/VCIP67698.2025.11396907)**, VCIP 2025. It provides the quantization-aware model, parameter export tools, an integer golden reference, and a runnable Alveo U50 demo for research and engineering evaluation.

The workflow covers model training and quantization, export of integer parameters and entropy tables, and image encoding and reconstruction. The supplied weights and FPGA images support evaluation of bitrate, RGB PSNR, latency and throughput.

## Training

Clone the repository and install the Python package in a Python 3.10+ environment:

```bash
git clone https://github.com/heliosz616/StreamLICDemo.git
cd StreamLICDemo
python -m pip install -r training/requirements.txt
python -m pip install -e training --no-deps
```

`streamlic_q.ARLiteSparse_Q` implements the analysis/synthesis transforms, hyperprior and sparse context model. Its forward pass returns the reconstructed image and likelihoods for the main and hyperlatents, which can be used to form a rate-distortion training objective.

```python
import torch
from streamlic_q import ARLiteSparse_Q

model = ARLiteSparse_Q().train()
model.toggle_quantization(False)
model.toggle_mask(False)
images = torch.rand(1, 3, 64, 64) - 0.5
result = model(images)
reconstruction = result["x_hat"]
likelihoods = result["likelihoods"]
```

Inputs are NCHW RGB tensors normalized as `pixel / 255 - 0.5`. Reconstruction includes the inverse offset. See the [training guide](training/README.md) for model controls and tensor conventions.

## Quantization

The model supports 12-bit weights and activations, learned quantization scales, sparse masks and entropy PDF fitting.

| Method | Purpose |
|---|---|
| `toggle_quantization(enabled)` | Enable or disable simulated quantization |
| `set_all_mask()` | Derive sparse masks from current parameters |
| `toggle_mask(enabled)` | Enable or disable sparse masks |
| `fix_encoder_flag(enabled)` | Control encoder freezing and the associated forward path |
| `init_pdf_table()` | Initialize learned PDF tables |
| `toggle_pdf_training_flag(enabled)` | Select PDF fitting and freeze other model parameters when enabled |

`models/quant_weights_and_pdf.pt` contains the supplied integer model parameters and learned PDF tables. The integer reference uses W12/A12 arithmetic and 24-bit CDF tables.

## Export

For a trained QAT model, `model.export_quantized_parameters()` returns quantized layer parameter dictionaries.

Export tensors from the supplied quantized archive and generate its CDF tables:

```bash
python -m streamlic.export.parameters \
  --model models/quant_weights_and_pdf.pt --output outputs/quantized_parameters.npz
python -m streamlic.export.cdf \
  --model models/quant_weights_and_pdf.pt --precision 24 --output outputs/cdf_p24.bin
```

The NumPy archive preserves tensor names, shapes and dtypes. CDF export produces a binary table bundle and JSON summary. Packed parameters for the U50 demo are provided in `runtime/parameters/`.

## Inference

### Python integer reference

Run encoding, reconstruction and golden tensor generation on a 128 × 128 center crop:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m streamlic.reference.cli \
  --image examples/test_pattern.png --model models/quant_weights_and_pdf.pt \
  --cdf runtime/parameters/cdf_p24.bin --width 128 --height 128 \
  --output outputs/reference.slc --reconstruction outputs/reference.png \
  --golden-dir outputs/golden --verify-decode
```

The reference runs independently of XRT and checks reconstructed latents and pixels through an encode/decode round trip. Use `--width 1280 --height 768` for the U50 image dimensions. Golden tensor definitions and standalone decoding commands are in the [reference guide](training/README.md#integer-golden-reference).

### Alveo U50 demo

#### Environment

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

#### Encode and reconstruct an image

```bash
./streamlic doctor
mkdir -p outputs
./streamlic encode examples/test_pattern.png outputs/test.slc
./streamlic decode outputs/test.slc outputs/reconstruction.png \
  --original examples/test_pattern.png --report outputs/result.json
```

The encoder writes a `.slc` bitstream. The decoder writes the reconstructed PNG; `--original` and `--report` enable RGB PSNR and a JSON evaluation report. Use images prepared at 1280 × 768. Encoding and decoding program separate FPGA images and run sequentially on a shared card.

Select a device with `--device N`, or an XRT setup script with `--xrt-settings FILE`. See [runtime usage](docs/runtime.md) for command options and dependencies.

#### Batch evaluation

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

## License

The source code, model parameters and FPGA binaries are licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) for noncommercial use.
