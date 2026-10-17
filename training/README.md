# Training, quantization and golden reference

`streamlic_q` contains the `ARLiteSparse_Q` PyTorch model and quantization utilities. It includes analysis/synthesis transforms, hyperprior transforms, sparse context prediction, learned quantization scales and entropy PDF fitting. It uses the CompressionModel base class from CompressAI.

## Installation and smoke test

From the repository root, in a Python 3.10+ environment:

```bash
python -m pip install -r training/requirements.txt
python -m pip install -e training --no-deps
python -m unittest discover -s training/tests -v
```

The pinned requirements correspond to the CPU validation environment. Select an appropriate PyTorch build for GPU training. The smoke test checks forward/backward execution, sparse masks, quantization, PDF fitting and parameter export.

## Model interface

```python
from streamlic_q import ARLiteSparse_Q

model = ARLiteSparse_Q()
model.toggle_quantization(False)
model.toggle_mask(False)
result = model(images)
# result contains x_hat and likelihoods for y and z.
```

Images use NCHW tensors. The analysis transform expects RGB values centered around zero (pixel values divided by 255, then minus 0.5); reconstruction adds 0.5 internally. Use spatial dimensions compatible with the model's downsampling, such as 64 × 64 for the smoke test. The FPGA runtime is separately fixed to 1280 × 768.

Model controls:

| Method | Effect |
|---|---|
| `toggle_quantization(enabled)` | Enable or disable simulated quantization |
| `set_all_mask()` | Derive sparse masks from current parameters |
| `toggle_mask(enabled)` | Enable or disable sparse masks |
| `fix_encoder_flag(enabled)` | Freeze or unfreeze encoder parameters and select the associated forward path |
| `init_pdf_table()` | Initialize the learned entropy PDF tables |
| `toggle_pdf_training_flag(enabled)` | Select PDF fitting; enabling freezes the other model parameters |
| `export_quantized_parameters()` | Return quantized layer parameter dictionaries |

`export_quantized_parameters()` returns layer parameter dictionaries. FPGA deployment uses packed parameters with the matching hardware layout, scales and entropy tables, provided in `runtime/parameters/`.

## Quantized model and parameter export

`models/quant_weights_and_pdf.pt` contains the integer model parameters and learned PDF tables used by the supplied runtime. Export its tensors and regenerate its CDF tables with:

```bash
python -m streamlic.export.parameters \
  --model models/quant_weights_and_pdf.pt --output outputs/quantized_parameters.npz
python -m streamlic.export.cdf \
  --model models/quant_weights_and_pdf.pt --precision 24 --output outputs/cdf_p24.bin
```

The NumPy archive uses keys such as `g_a.0.weight`, preserving tensor shapes and dtypes. The CDF command also writes a JSON summary next to its binary output. The ready-to-run U50 parameter package is in `runtime/parameters/`.

## Integer golden reference

The Python reference implements W12/A12 integer inference and arithmetic coding. It exports intermediate tensors for numerical comparisons and reconstructs images from `.slc` bitstreams.

Run an encode/decode consistency check on a 128 × 128 center crop:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m streamlic.reference.cli \
  --image examples/test_pattern.png --model models/quant_weights_and_pdf.pt \
  --cdf runtime/parameters/cdf_p24.bin --width 128 --height 128 \
  --output outputs/reference.slc --reconstruction outputs/reference.png \
  --golden-dir outputs/golden --verify-decode
```

For a full-size comparison with the U50 runtime, use `--width 1280 --height 768`. The reference accepts dimensions that are multiples of 128 and center-crops the input to the requested size. The U50 runtime uses 1280 × 768 inputs.

Golden files contain one integer per line:

| File | Tensor |
|---|---|
| `test_patch.txt` | Padded, pixel-unshuffled input |
| `test_y.txt` | Analysis-transform output |
| `test_golden1.txt` | Hyperlatent symbols |
| `test_golden2.txt` | Interleaved main-latent symbols and CDF indexes |
| `test_hyper.txt` | Hyper-synthesis output |
| `test_y_hat.txt` | Reconstructed main latents |

Decode a reference or U50 bitstream with the Python model:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m streamlic.reference.decode_cli \
  --bitstream outputs/reference.slc --model models/quant_weights_and_pdf.pt \
  --cdf runtime/parameters/cdf_p24.bin --output outputs/reference_decoded.png
```
