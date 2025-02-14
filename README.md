# MyQuantLib

![Static Badge](https://img.shields.io/badge/license-MIT-red) ![Static Badge](https://img.shields.io/badge/language-Python-blue) ![Static Badge](https://img.shields.io/badge/author-Evans_Li-yellow)

A native and hackable neural network quantization library based on pytorch for research purposes.

**Author:** Evans Li
**Contact:** liangkeshulizi@gmail.com

## Introduction

Neural network quantization is a powerful technique and hot research topic that drastically reduces the memory and computational requirements to run a model by representing weights and activations with lower precision, such as 8-bit integers instead of 32-bit floating-point numbers. This makes models more efficient, especially on resource-constrained devices like mobile phones and edge devices.

Pytorch provides native API for quantization (Eager Mode & Graph FX), and they uses quantization libraries like `qnnpack` and `fbgemm` for inference under the hood. However, those quantization libraries are mostly written in C or Assembly, some are close-source, making it hard to understand what's actually happenning, impossible to tinker with the algorithm. Using them with custom backends (hardware accelerators) means heavy reliance on these 

MyQuantLib implements the quantization algorithm and quantized inference in plain Pytorch with naive `torch.int8`. It makes it easy to understand how quantied layers are computed, implement it on custom backends (hardware accelerators) and tinker with different algorithm. Besides, it innovatively uses an dry-run pass to connect quantization parameters (qparams) between layers and pre-compute all the paramters before the real forward pass, improving performance and eliminating any need of floating-point computation (especially bias) during inference, making it possible to be implemented on a hardware without floating-point support.

MyQuantLib supports 8-bit Static Post-Training Quantization (PTQ). This is a method where quantization occurs after the model is trained, without requiring retraining. It works by analyzing the activations of the pre-trained model on a representative dataset to gather statistics like the min and max values. These statistics are used to determine the scaling factors for the weights and activations.

This package is a side project of one of my undergrad research projects. It is inspired by [gemmlowp](https://github.com/google/gemmlowp).

Dynamic Quantization, variable bit quantization and more operations will be added in the near future.

## Installation

To install the project from PyPI, run:

```sh
pip install myquantlib
```

Or if you want to clone the repository directly from github and build locally (recommended), run:

```sh
git clone 
pip install -e /path/to/myquantlib
```

To run the example, cd into the repository and run:

```sh
cd myquantlib
python example.py
```

This will train and quantize an example CNN model on your computer. For detailed information about the usage, see `example.py`

## Contribution

All kinds of contributions are welcome. Please open an issue before pull request.
