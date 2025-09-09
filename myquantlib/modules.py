import abc
import torch
from torch import nn
from torch import Tensor
from torch.quantization import ObserverBase
from typing import Optional

from .functional_utils import *

__all__ = [
    "MyQuantizedModule",
    "MyQuantizedWrapper",
    "MyQuantizedIdentity",
    "MyQuantizedReLU",
    "MyQuantize",
    "MyDeQuantize",
    "MyQuantizedLinear",
    "MyQuantizedConv2d"
]


class MyQuantizedModule(nn.Module, abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forward = self._forward_qparams

    @abc.abstractmethod
    def _forward_qparams(self, input_qparam: Optional[tuple]) -> Optional[tuple]:
        raise NotImplementedError

    @abc.abstractmethod
    def _forward_tensor(self, input: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, input):
        '''This method will be replaced by _forward_qparams/_forward_tensor in the instance.'''
        raise RuntimeError("Calling forward from class.")

    @classmethod
    @abc.abstractmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        raise NotImplementedError


class MyQuantizedWrapper(MyQuantizedModule):
    def __init__(self, mod: nn.Module):
        super().__init__()
        self.mod = mod

    def _forward_qparams(self, input_qparam):
        self.forward = self._forward_tensor
        return input_qparam

    def _forward_tensor(self, input: Tensor) -> Tensor:
        return self.mod(input)

    @classmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        return cls(mod)


class MyQuantizedIdentity(MyQuantizedModule):
    def _forward_qparams(self, input_qparam):
        self.forward = self._forward_tensor
        return input_qparam

    def _forward_tensor(self, input: Tensor) -> Tensor:
        return input

    @classmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        return cls()


class MyQuantizedReLU(MyQuantizedModule):
    def __init__(self):
        super().__init__()
        self.zero_point = 0

    def _forward_qparams(self, input_qparam):
        self.forward = self._forward_tensor
        _, self.zero_point = input_qparam
        return input_qparam

    def _forward_tensor(self, input: Tensor) -> Tensor:
        assert input.dtype == torch.int8
        return torch.clamp(input, min=self.zero_point)

    @classmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        return cls()


class MyQuantize(MyQuantizedModule):
    def __init__(self, scale: float, zero_point: int):
        super().__init__()
        self.scale, self.zero_point = scale, zero_point

    def _forward_qparams(self, input_qparam: None = None) -> tuple:
        self.forward = self._forward_tensor
        return self.scale, self.zero_point

    def _forward_tensor(self, input: Tensor) -> Tensor:
        assert input.dtype == torch.float32
        return my_quantize_per_tensor(input, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        assert hasattr(mod, "activation_post_process"), "module not prepared before converting"

        scale, zero_point = mod.activation_post_process.calculate_qparams()
        return cls(float(scale), int(zero_point))


class MyDeQuantize(MyQuantizedModule):
    def __init__(self):
        super().__init__()
        self.scale, self.zero_point = 1.0, 0

    def _forward_qparams(self, input_qparam: tuple) -> None:
        self.forward = self._forward_tensor
        self.scale, self.zero_point = input_qparam

    def _forward_tensor(self, input: Tensor) -> Tensor:
        assert input.dtype == torch.int8
        return my_dequantize(input, self.scale, self.zero_point)

    @classmethod
    def from_float(cls, mod: nn.Module, use_precomputed_fake_quant: bool = False):
        return cls()


class MyQuantizedLinear(MyQuantizedModule):
    # NOTE:
    #   - Supports Static Post-Train Quantization and Quantization-Aware Training.
    #   - Only supports per tensor symmetric/asymmetric quantization. (per channel quantization will be added soon)
    #   - Serialization currently not supported.

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # raw qparams (used to derive qparams, but not directly used in forward pass)
        self.fweight = torch.zeros((out_features, in_features), dtype=torch.float32)
        self.fbias = torch.zeros(out_features, dtype=torch.float32)
        self.input_scale,  self.input_zero_point = 1., 0
        self.weight_scale, self.weight_zero_point = 1., 0
        self.output_scale, self.output_zero_point = 1., 0

        # qparams (actually used in forward pass)
        self.qweight = torch.zeros((out_features, in_features), dtype=torch.int8)
        if bias:
            self.qbias = torch.zeros(out_features, dtype=torch.int32)
        else:
            self.qbias = None
        self.quantized_multiplier, self.right_shift = 1, 0      # for output multiplication

    def _compute_qparams(self):
        # quantize weight and bias
        self.qweight = my_quantize_per_tensor(self.fweight, self.weight_scale, self.weight_zero_point)
        self.qbias = torch.round(self.fbias / (self.input_scale * self.weight_scale)).to(torch.int32)

        # compute qparams for output multiplication
        output_factor: float = (self.input_scale * self.weight_scale) / self.output_scale
        self.quantized_multiplier, self.right_shift = quantize_multiplier_smaller_than_one(output_factor)

        # delete the floating point data to release memory
        del self.fweight, self.fbias

    def _forward_qparams(self, input_qparam: tuple) -> tuple:
        self.forward = self._forward_tensor        # FIXME: In RNN each layer may be evaluated multiple times in a single pass, yet this should be executed after the last time
        self.input_scale, self.input_zero_point = input_qparam
        self._compute_qparams()
        return self.output_scale, self.output_zero_point

    def _forward_tensor(self, input: Tensor) -> Tensor:
        assert input.dtype == torch.int8

        if input.dim() == 1:
            input.unsqueeze_(dim=0)

        xq = input.to(torch.int32)
        wq = self.qweight.to(torch.int32)
        bias_q = self.qbias

        z_int = nn.functional.linear(xq, wq, bias_q)

        if self.input_zero_point or self.weight_zero_point:

            #   lhs * rhs
            # + lhs_offset * P * rhs
            # + lhs * rhs_offset * Q
            # + lhs_offset * rhs_offset * P * Q

            z_int += (
                - torch.sum(wq, dim=1, keepdim=True).T * self.input_zero_point
                - torch.sum(xq, dim=1, keepdim=True) * self.weight_zero_point
                + self.input_zero_point * self.weight_zero_point * self.in_features
            )

        # The following computation is the equivelent of:
        # z_out8 = torch.round((z_int) * (self.input_scale * self.weight_scale)/self.output_scale + self.output_zero_point).clamp(-127, 127).to(torch.int8)

        z_out32 = rounding_right_shift(rounding_doubling_high_mul(z_int, torch.tensor(self.quantized_multiplier)), self.right_shift)
        z_out8 = (z_out32 + self.output_zero_point).clamp(-127, 127).to(torch.int8)
        
        return z_out8

    @classmethod
    def from_float(cls, mod: nn.Linear, use_precomputed_fake_quant: bool = False):

        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert hasattr(mod, "activation_post_process"), "module not prepared before converting"

        # extract params from observer
        activation_observer: ObserverBase = mod.activation_post_process
        weight_observer: ObserverBase = (
            mod.weight_fake_quant
            if hasattr(mod, "weight_fake_quant")
            else mod.qconfig.weight()
        )
        assert activation_observer.dtype == torch.qint8 and weight_observer.dtype == torch.qint8

        if not use_precomputed_fake_quant:
            weight_observer(mod.weight)  # observe weight

        output_scale, output_zero_point = activation_observer.calculate_qparams()
        weight_scale, weight_zero_point = weight_observer.calculate_qparams()

        qlinear = cls(mod.in_features, mod.out_features, bias=(mod.bias is not None))
        qlinear.fweight = mod.weight.data
        qlinear.fbias = mod.bias.data

        qlinear.output_scale, qlinear.output_zero_point = float(output_scale), int(output_zero_point)
        qlinear.weight_scale, qlinear.weight_zero_point = float(weight_scale), int(weight_zero_point)

        return qlinear


class MyQuantizedConv2d(MyQuantizedModule):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride=1, padding=0, bias=True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernal_size = kernel_size
        self.stride = stride
        self.padding = padding

        # raw params (used to derive qparams)
        self.fweight = torch.zeros((out_channels, in_channels, *kernel_size))
        if bias:
            self.fbias = torch.zeros(out_channels)
        self.input_scale,  self.input_zero_point = 1., 0
        self.weight_scale, self.weight_zero_point = 1., 0
        self.output_scale, self.output_zero_point = 1., 0

        # qparams (actually used in forward pass)
        self.qweight = torch.zeros((out_channels, in_channels, *kernel_size), dtype=torch.int8)
        if bias:
            self.qbias = torch.zeros(out_channels, dtype=torch.int32)
        else:
            self.qbias = None
        self.quantized_multiplier, self.right_shift = 1, 0      # for output multiplication

    def _compute_qparams(self):
        # quantize weight and bias
        self.qweight = my_quantize_per_tensor(self.fweight, self.weight_scale, self.weight_zero_point)
        self.qbias = torch.round(self.fbias / (self.input_scale * self.weight_scale)).to(torch.int32)

        # compute qparams for output multiplication
        output_factor: float = (self.input_scale * self.weight_scale) / self.output_scale
        self.quantized_multiplier, self.right_shift = quantize_multiplier_smaller_than_one(output_factor)

        # delete the floating point data to release memory
        del self.fweight, self.fbias

    def _forward_qparams(self, input_qparam: tuple) -> tuple:
        self.forward = self._forward_tensor
        self.input_scale, self.input_zero_point = input_qparam
        self._compute_qparams()
        return self.output_scale, self.output_zero_point

    def _forward_tensor(self, input: Tensor) -> Tensor:
        assert input.dtype == torch.int8

        xq = input.to(torch.int32) - self.input_zero_point
        wq = self.qweight.to(torch.int32) - self.weight_zero_point
        bias_q = self.qbias

        z_int = nn.functional.conv2d(xq, wq, bias_q, self.stride, self.padding)  # NOTE: pad with zero since input_zero_point has already been subtracted

        # The following computation is the equivelent of:
        # z_out8 = torch.round((z_int) * (self.input_scale * self.weight_scale) / self.output_scale + self.output_zero_point).clamp(-127, 127).to(torch.int8)
        
        z_out32 = rounding_right_shift(rounding_doubling_high_mul(z_int, torch.tensor(self.quantized_multiplier)), self.right_shift)
        z_out8 = (z_out32 + self.output_zero_point).clamp(-127, 127).to(torch.int8)

        return z_out8

    @classmethod
    def from_float(cls, mod: nn.Conv2d, use_precomputed_fake_quant: bool = False):

        assert hasattr(mod, "qconfig"), "Input float module must have qconfig defined"
        assert hasattr(mod, "activation_post_process"), "module not prepared before converting"

        # extract params from observer
        activation_observer: ObserverBase = mod.activation_post_process
        weight_observer: ObserverBase = (
            mod.weight_fake_quant
            if hasattr(mod, "weight_fake_quant")
            else mod.qconfig.weight()
        )
        assert activation_observer.dtype == torch.qint8 and weight_observer.dtype == torch.qint8

        if not use_precomputed_fake_quant:
            weight_observer(mod.weight)  # observe weight

        output_scale, output_zero_point = activation_observer.calculate_qparams()
        weight_scale, weight_zero_point = weight_observer.calculate_qparams()

        qconv = cls(mod.in_channels, mod.out_channels, mod.kernel_size, mod.stride, mod.padding, bias=(mod.bias is not None))
        qconv.fweight = mod.weight.data
        qconv.fbias = mod.bias.data

        qconv.output_scale, qconv.output_zero_point = float(output_scale), int(output_zero_point)
        qconv.weight_scale, qconv.weight_zero_point = float(weight_scale), int(weight_zero_point)

        return qconv
