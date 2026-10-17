"""Bit-exact integer operators shared by the reference encoder and decoder.

The arithmetic follows the deployed HLS model: signed W12/A12 data, a wide
accumulator, per-channel integer scaling, symmetric saturation to [-2047,2047],
and four fractional bits for y/z latents.
"""

from typing import Dict

import torch
import torch.nn.functional as F

from .constants import ACTIVATION_BITS, ACTIVATION_MAX, ACTIVATION_MIN


def _scaled_output(accumulator: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    scale = params["scale_param"].to(torch.int64).view(1, -1, 1, 1)
    shift = params["scale_shift"].to(torch.int64).view(1, -1, 1, 1)
    value = (accumulator.to(torch.int64) * scale) >> shift
    bias = params.get("bias")
    if bias is not None:
        value += bias.to(torch.int64).view(1, -1, 1, 1)
    result_bits = 16 - ACTIVATION_BITS
    value = (value + (1 << (result_bits - 1))) >> result_bits
    return value.clamp(ACTIVATION_MIN, ACTIVATION_MAX).to(torch.int32)


def quantized_conv2d(q_in: torch.Tensor, params: Dict[str, torch.Tensor],
                     stride: int, padding, groups: int = 1) -> torch.Tensor:
    # torch has no integer conv2d. Double is exact for the accumulator magnitudes
    # in this network and matches the established golden-model implementation.
    accumulator = F.conv2d(
        q_in.to(torch.float64), params["weight"].to(torch.float64),
        stride=stride, padding=padding, groups=groups).round().to(torch.int64)
    return _scaled_output(accumulator, params)


def sparse_conv2d(q_in: torch.Tensor, params: Dict[str, torch.Tensor],
                  stride: int, padding, groups: int = 1) -> torch.Tensor:
    weights = params["weight"].to(torch.float64) * params["mask"].to(torch.float64)
    accumulator = F.conv2d(
        q_in.to(torch.float64), weights, stride=stride, padding=padding,
        groups=groups).round().to(torch.int64)
    return _scaled_output(accumulator, params)


_TANH_LUT = (
    0, 64, 127, 190, 251, 310, 367, 421, 473, 522, 568, 611, 650,
    687, 721, 752, 780, 805, 829, 850, 869, 886, 901, 915, 927, 938,
    948, 956, 964, 971, 977, 982, 987, 991, 995, 999, 1001, 1004,
    1006, 1008, 1010, 1012, 1013, 1015, 1016, 1017, 1018, 1018,
    1019, 1020, 1020, 1021, 1021, 1021, 1022, 1022, 1022, 1022,
    1023, 1023, 1023, 1023, 1023, 1023, 1023,
)


def quantized_lagc(q_in: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    affine = {
        "weight": params["affine_weight"],
        "bias": params["affine_bias"],
        "scale_param": params["affine_scale_param"],
        "scale_shift": params["affine_scale_shift"],
    }
    tanh_in = quantized_conv2d(q_in, affine, 1, 0).to(torch.int64)
    lut = torch.tensor(_TANH_LUT, dtype=torch.int64, device=q_in.device)
    magnitude = tanh_in.abs()
    index = magnitude >> 4
    residual = magnitude & 15
    safe_index = index.clamp(max=62)
    interpolated = lut[safe_index] + ((residual * (lut[safe_index + 1] - lut[safe_index])) >> 4)
    tanh_positive = torch.where(index >= 63, torch.full_like(index, 1024), interpolated)
    tanh_out = torch.where(tanh_in < 0, -tanh_positive, tanh_positive)

    score_scale = params["score_scale_param"].to(torch.int64).view(1, -1, 1, 1)
    score_shift = params["score_scale_shift"].to(torch.int64).view(1, -1, 1, 1)
    score_bias = params["score_bias"].to(torch.int64).view(1, -1, 1, 1)
    gain = ((tanh_out * score_scale) >> score_shift) + score_bias
    result_bits = 16 - ACTIVATION_BITS
    gain = ((gain + (1 << (result_bits - 1))) >> result_bits).clamp(
        ACTIVATION_MIN, ACTIVATION_MAX)

    output_scale = params["output_scale_param"].to(torch.int64).view(1, -1, 1, 1)
    output_shift = params["output_scale_shift"].to(torch.int64).view(1, -1, 1, 1)
    # The left shift mirrors the HLS intermediate precision and cancels in the
    # final shift; retaining it makes the contract explicit.
    intermediate = (q_in.to(torch.int64) << 16) * gain
    output = (intermediate * output_scale) >> (output_shift + 16)
    return output.clamp(ACTIVATION_MIN, ACTIVATION_MAX).to(torch.int32)


def quantized_leaky_relu(q_in: torch.Tensor) -> torch.Tensor:
    output = q_in.clone()
    negative = q_in < 0
    output[negative] = (q_in[negative] + 32) >> 6
    return output


def quantized_upsample(q_in: torch.Tensor, params: Dict[str, torch.Tensor],
                       pad_zero: bool = False, remove_pad: bool = False,
                       alternate_order: bool = False) -> torch.Tensor:
    value = F.pad(q_in, (1, 1, 1, 1)) if pad_zero else q_in
    convolved = quantized_conv2d(value, params, stride=1, padding=0)
    shuffled = F.pixel_shuffle(convolved, 2)
    if alternate_order:
        upper, lower = convolved.chunk(2, 1)
        shuffled[:, :, 0::2, 0::2] = upper[:, 0::2]
        shuffled[:, :, 0::2, 1::2] = upper[:, 1::2]
        shuffled[:, :, 1::2, 0::2] = lower[:, 0::2]
        shuffled[:, :, 1::2, 1::2] = lower[:, 1::2]
    return shuffled[:, :, 1:-1, 1:-1] if remove_pad else shuffled


def quantized_channel_addition(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=1)
    return (first + second).clamp(ACTIVATION_MIN, ACTIVATION_MAX)
