"""Exact W12/A12 StreamLIC model, independent of the training framework."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .constants import (
    ACTIVATION_MAX, ACTIVATION_MIN, LATENT_FRACTIONAL_BITS,
    LATENT_ROUNDING, MINIMUM_TILE, Y_CHANNELS, Z_CHANNELS,
)
from .operators import (
    quantized_channel_addition, quantized_conv2d, quantized_lagc,
    quantized_leaky_relu, quantized_upsample, sparse_conv2d,
)


@dataclass
class EncodeResult:
    input_tensor: torch.Tensor
    y: torch.Tensor
    z_symbols: torch.Tensor
    hyper: torch.Tensor
    y_symbols: torch.Tensor
    y_indexes: torch.Tensor
    y_hat: torch.Tensor
    reconstruction: torch.Tensor
    z_clamped: int
    y_clamped: int


def _bounds_tensor(bounds: Sequence[Tuple[int, int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    minimum = torch.tensor([v[0] for v in bounds], dtype=torch.int32, device=device)
    maximum = torch.tensor([v[1] for v in bounds], dtype=torch.int32, device=device)
    return minimum, maximum


class IntegerStreamLIC:
    def __init__(self, parameters) -> None:
        self.parameters = parameters

    @classmethod
    def load(cls, path) -> "IntegerStreamLIC":
        params = torch.load(str(path), map_location="cpu", weights_only=False)
        return cls(params)

    @staticmethod
    def load_image(path, height: int, width: int) -> torch.Tensor:
        if height < MINIMUM_TILE or width < MINIMUM_TILE:
            raise ValueError("height and width must be at least 128")
        if height % MINIMUM_TILE or width % MINIMUM_TILE:
            raise ValueError("height and width must be multiples of 128")
        image = Image.open(path).convert("RGB")
        source = np.asarray(image, dtype=np.uint8)
        if source.shape[0] < height or source.shape[1] < width:
            raise ValueError("input image is smaller than requested crop")
        top = (source.shape[0] - height) // 2
        left = (source.shape[1] - width) // 2
        source = source[top:top + height, left:left + width]
        tensor = torch.from_numpy(source.copy()).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(torch.int32) - 128

    def analysis(self, image: torch.Tensor) -> torch.Tensor:
        p = self.parameters
        value = quantized_conv2d(image, p["g_a.0"], 2, 1)
        value = quantized_lagc(value, p["g_a.1"])
        value = quantized_conv2d(value, p["g_a.2"], 2, 1)
        value = quantized_lagc(value, p["g_a.3"])
        value = quantized_conv2d(value, p["g_a.4"], 2, 1)
        value = quantized_lagc(value, p["g_a.5"])
        return quantized_conv2d(value, p["g_a.6"], 2, 1)

    def hyper_analysis(self, y: torch.Tensor,
                       z_bounds: Sequence[Tuple[int, int]]) -> Tuple[torch.Tensor, int]:
        p = self.parameters
        value = quantized_conv2d(y, p["h_a.0"], 2, 1)
        value = quantized_lagc(value, p["h_a.1"])
        value = quantized_conv2d(value, p["h_a.2"], 2, 1)
        symbols = (value + LATENT_ROUNDING) >> LATENT_FRACTIONAL_BITS
        if symbols.shape[1] != len(z_bounds):
            raise ValueError("z CDF count does not match channel count")
        minimum, maximum = _bounds_tensor(z_bounds, symbols.device)
        bounded = torch.maximum(torch.minimum(
            symbols, maximum.view(1, -1, 1, 1)),
            minimum.view(1, -1, 1, 1)).to(torch.int32)
        return bounded, int(torch.count_nonzero(bounded != symbols))

    def hyper_synthesis(self, z_symbols: torch.Tensor) -> torch.Tensor:
        p = self.parameters
        value = quantized_upsample(z_symbols, p["h_s.0"], pad_zero=True)
        value = quantized_leaky_relu(value)
        return quantized_upsample(
            value, p["h_s.2"], remove_pad=True, alternate_order=True)

    def _predict_row(self, history: torch.Tensor, hyper: torch.Tensor,
                     row: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self.parameters
        context = history[:, :, row:row + 3, :].clone()
        context = sparse_conv2d(context, p["context_prediction.0"], 1, 0)
        context = quantized_leaky_relu(context)
        context = sparse_conv2d(context, p["context_prediction.2"], 1, (0, 1))
        combined = torch.cat((hyper[:, :, row:row + 1, :], context), dim=1)
        value = quantized_channel_addition(combined)
        value = quantized_leaky_relu(value)
        gaussian = sparse_conv2d(value, p["entropy_parameters.2"], 1, (0, 1))
        means = gaussian[:, 0::2]
        indexes = gaussian[:, 1::2].clamp(0, 127).to(torch.int32)
        return means, indexes

    def autoregressive_encode(self, y: torch.Tensor, hyper: torch.Tensor,
                              y_bounds: Sequence[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        if len(y_bounds) != 128:
            raise ValueError("expected 128 y CDF bounds")
        _, channels, height, width = y.shape
        if channels != Y_CHANNELS:
            raise ValueError("unexpected y channel count")
        history = F.pad(torch.zeros_like(y), (1, 1, 3, 0))
        symbols = torch.empty_like(y, dtype=torch.int32)
        indexes_all = torch.empty_like(y, dtype=torch.int32)
        reconstructed = torch.empty_like(y, dtype=torch.int32)
        minimum, maximum = _bounds_tensor(y_bounds, y.device)
        clamp_count = 0
        for row in range(height):
            means, indexes = self._predict_row(history, hyper, row)
            residual = ((y[:, :, row:row + 1] - means + LATENT_ROUNDING)
                        >> LATENT_FRACTIONAL_BITS)
            row_min = minimum[indexes.to(torch.int64)]
            row_max = maximum[indexes.to(torch.int64)]
            bounded = torch.maximum(torch.minimum(residual, row_max), row_min)
            clamp_count += int(torch.count_nonzero(bounded != residual))
            residual = bounded
            row_hat = ((residual << LATENT_FRACTIONAL_BITS) + means).clamp(
                ACTIVATION_MIN, ACTIVATION_MAX).to(torch.int32)
            history[:, :, row + 3:row + 4, 1:-1] = row_hat
            symbols[:, :, row:row + 1] = residual
            indexes_all[:, :, row:row + 1] = indexes
            reconstructed[:, :, row:row + 1] = row_hat
        return symbols, indexes_all, reconstructed, clamp_count

    def autoregressive_decode(self, symbols: torch.Tensor, hyper: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, channels, height, _ = symbols.shape
        if channels != Y_CHANNELS:
            raise ValueError("unexpected y channel count")
        history = F.pad(torch.zeros_like(symbols), (1, 1, 3, 0))
        indexes_all = torch.empty_like(symbols, dtype=torch.int32)
        reconstructed = torch.empty_like(symbols, dtype=torch.int32)
        for row in range(height):
            means, indexes = self._predict_row(history, hyper, row)
            row_hat = ((symbols[:, :, row:row + 1] << LATENT_FRACTIONAL_BITS)
                       + means).clamp(ACTIVATION_MIN, ACTIVATION_MAX).to(torch.int32)
            history[:, :, row + 3:row + 4, 1:-1] = row_hat
            indexes_all[:, :, row:row + 1] = indexes
            reconstructed[:, :, row:row + 1] = row_hat
        return indexes_all, reconstructed

    def synthesis(self, y_hat: torch.Tensor) -> torch.Tensor:
        p = self.parameters
        value = quantized_upsample(y_hat, p["g_s.0"], pad_zero=True)
        value = quantized_lagc(value, p["g_s.1"])
        value = quantized_upsample(value, p["g_s.2"])
        value = quantized_lagc(value, p["g_s.3"])
        value = quantized_upsample(value, p["g_s.4"])
        value = quantized_lagc(value, p["g_s.5"])
        value = quantized_upsample(value, p["g_s.6"], remove_pad=True)
        scale = float(p["g_s.6"]["float_output_scale"].item())
        return (((value.to(torch.float32) * scale) + 0.5) * 255.0).clamp(
            0, 255).to(torch.uint8)

    def encode(self, image: torch.Tensor, y_bounds, z_bounds) -> EncodeResult:
        y = self.analysis(image)
        z_symbols, z_clamped = self.hyper_analysis(y, z_bounds)
        hyper = self.hyper_synthesis(z_symbols)
        y_symbols, indexes, y_hat, y_clamped = self.autoregressive_encode(y, hyper, y_bounds)
        reconstruction = self.synthesis(y_hat)
        return EncodeResult(image, y, z_symbols, hyper, y_symbols, indexes,
                            y_hat, reconstruction, z_clamped, y_clamped)


def tensor_hwc_flat(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze(0).permute(1, 2, 0).contiguous().view(-1).cpu().numpy()


def save_image_tensor(tensor: torch.Tensor, path) -> None:
    array = tensor.squeeze(0).permute(1, 2, 0).contiguous().cpu().numpy()
    Image.fromarray(array).save(path)
