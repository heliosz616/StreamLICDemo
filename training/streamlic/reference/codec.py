"""Complete StreamLIC encoding and decoding loops."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from streamlic.entropy.arithmetic import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    InterleavedArithmeticDecoder,
    InterleavedArithmeticEncoder,
)
from streamlic.entropy.bitstream import FrameBitstream
from streamlic.entropy.cdf import CDFBundle
from streamlic.model.constants import ACTIVATION_MAX, ACTIVATION_MIN, LATENT_FRACTIONAL_BITS
from streamlic.model.integer_model import EncodeResult, IntegerStreamLIC, tensor_hwc_flat


@dataclass
class EncodedFrame:
    result: EncodeResult
    frame: FrameBitstream
    data: bytes


def _ape_native_flat(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten CHW data in the PCE=12 order emitted by the FPGA APE."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != 192:
        raise ValueError("expected one 192-channel latent tensor")
    height, width = tensor.shape[2:]
    if width % 2:
        raise ValueError("APE-native ordering requires an even latent width")
    return tensor.reshape(32, 6, height, width // 2, 2).permute(
        2, 0, 3, 4, 1).contiguous().view(-1)


def encode_frame(model: IntegerStreamLIC, image: torch.Tensor,
                 cdfs: CDFBundle) -> EncodedFrame:
    if cdfs.precision != 24:
        raise ValueError("StreamLIC .slc files require 24-bit CDFs")
    result = model.encode(image, cdfs.y_bounds, cdfs.z_bounds)
    z_encoder = ArithmeticEncoder()
    z_hwc = tensor_hwc_flat(result.z_symbols)
    z_channels = result.z_symbols.shape[1]
    for position, symbol in enumerate(z_hwc):
        z_encoder.encode(int(symbol), cdfs.z[position % z_channels])
    z_payload = z_encoder.finish()

    y_encoder = InterleavedArithmeticEncoder(4)
    y_symbols = _ape_native_flat(result.y_symbols).cpu().numpy()
    y_indexes = _ape_native_flat(result.y_indexes).cpu().numpy()
    for symbol, index in zip(y_symbols, y_indexes):
        y_encoder.encode(int(symbol), cdfs.y[int(index)])
    y_payloads = y_encoder.finish()

    height, width = image.shape[2:]
    frame = FrameBitstream(width, height, z_payload, y_payloads)
    return EncodedFrame(result, frame, frame.to_bytes())


def decode_frame(model: IntegerStreamLIC, data: bytes,
                 cdfs: CDFBundle) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                           torch.Tensor, torch.Tensor]:
    """Decode one complete frame and reconstruct its RGB image tensor."""
    frame = FrameBitstream.from_bytes(data)
    if cdfs.precision != 24:
        raise ValueError("StreamLIC .slc files require 24-bit CDFs")
    if frame.width % 128 or frame.height % 128:
        raise ValueError("unsupported frame dimensions")

    z_height, z_width = frame.height // 64, frame.width // 64
    y_height, y_width = frame.height // 16, frame.width // 16
    z_decoder = ArithmeticDecoder(frame.z_payload)
    z_hwc = torch.empty((z_height, z_width, len(cdfs.z)), dtype=torch.int32)
    for h in range(z_height):
        for w in range(z_width):
            for channel, table in enumerate(cdfs.z):
                z_hwc[h, w, channel] = z_decoder.decode(table)
    z_symbols = z_hwc.permute(2, 0, 1).unsqueeze(0).contiguous()
    hyper = model.hyper_synthesis(z_symbols)

    y_decoder = InterleavedArithmeticDecoder(frame.y_payloads)
    history = F.pad(torch.zeros((1, 192, y_height, y_width), dtype=torch.int32),
                    (1, 1, 3, 0))
    y_symbols = torch.empty((1, 192, y_height, y_width), dtype=torch.int32)
    y_hat = torch.empty_like(y_symbols)
    indexes_all = torch.empty_like(y_symbols)
    for row in range(y_height):
        means, indexes = model._predict_row(history, hyper, row)
        row_symbols_hwc = torch.empty((1, 192, 1, y_width), dtype=torch.int32)
        for channel_group in range(32):
            for width_pair in range(y_width // 2):
                for spatial in range(2):
                    width_index = 2 * width_pair + spatial
                    for lane in range(6):
                        channel = 6 * channel_group + lane
                        index = int(indexes[0, channel, 0, width_index])
                        row_symbols_hwc[0, channel, 0, width_index] = y_decoder.decode(
                            cdfs.y[index])
        row_hat = ((row_symbols_hwc << LATENT_FRACTIONAL_BITS) + means).clamp(
            ACTIVATION_MIN, ACTIVATION_MAX).to(torch.int32)
        history[:, :, row + 3:row + 4, 1:-1] = row_hat
        y_symbols[:, :, row:row + 1] = row_symbols_hwc
        indexes_all[:, :, row:row + 1] = indexes
        y_hat[:, :, row:row + 1] = row_hat
    reconstruction = model.synthesis(y_hat)
    return z_symbols, y_symbols, indexes_all, y_hat, reconstruction


# Keep the original verification API for existing scripts and test vectors.
decode_frame_for_test = decode_frame


def psnr(reference: torch.Tensor, reconstruction: torch.Tensor) -> float:
    difference = reference.to(torch.float64) - reconstruction.to(torch.float64)
    mse = float(torch.mean(difference * difference))
    return float("inf") if mse == 0 else 10.0 * math.log10((255.0 * 255.0) / mse)
