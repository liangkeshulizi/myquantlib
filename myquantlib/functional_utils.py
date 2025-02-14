import torch
from torch import Tensor

# A naive implementation of torch.quantize_per_tensor, but returns raw torch.int8
def my_quantize_per_tensor(tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    return torch.clamp(torch.round(tensor / scale + zero_point), -128, 127).to(torch.int8)

# A naive implementation of torch.dequantize, but inputs raw torch.int8
def my_dequantize(tensor: Tensor, scale: float, zero_point: int) -> Tensor:
    return (tensor.to(torch.int16) - zero_point).to(torch.float32) * scale

# see doc for detailed explaination
def quantize_multiplier_smaller_than_one(real_multiplier: float):
    assert 0. < real_multiplier  < 1.
    
    s = 0
    while real_multiplier < 0.5:
        real_multiplier *= 2.0
        s += 1

    q = int(round(real_multiplier * 2**31))

    assert q <= 2**31

    if q == 2**31:
        q /= 2
        s -= 1
    
    quantized_multiplier = q
    right_shift = s

    return (quantized_multiplier, right_shift)

# This function implements the same computation as the ARMv7 NEON VQRDMULH instruction.
def rounding_doubling_high_mul(a: Tensor, b: Tensor):
    ab_64 = (a.to(torch.int64) * b.to(torch.int64))
    nudge = torch.where(ab_64 >= 0, (1 << 30), (1 - (1 << 30))).to(torch.int64)
    ab_32 = ((ab_64 + nudge) >> 31).to(torch.int32)
    return ab_32
