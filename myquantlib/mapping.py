from torch.quantization import QuantStub, DeQuantStub
from torch.nn import *
from .modules import *

__all__ = ["MY_QUANT_MODULE_MAPPINGS"]

MY_QUANT_MODULE_MAPPINGS = {
    QuantStub: MyQuantize,
    DeQuantStub: MyDeQuantize,
    Linear: MyQuantizedLinear,
    Conv2d: MyQuantizedConv2d,
    
    Identity: MyQuantizedIdentity,
    Dropout: MyQuantizedIdentity,

    MaxPool2d: MyQuantizedWrapper,      # FIXME: pad with zero_point instead of 0
    ReLU: MyQuantizedReLU,
    
    Flatten: MyQuantizedWrapper,
}
