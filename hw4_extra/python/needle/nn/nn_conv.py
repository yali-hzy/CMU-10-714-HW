"""The module."""

from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
from .nn_basic import BatchNorm2d, ReLU


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(
                in_channels * kernel_size**2,
                out_channels * kernel_size**2,
                shape=(kernel_size, kernel_size, in_channels, out_channels),
                device=device,
                dtype=dtype,
            )
        )
        bound = 1 / np.sqrt(in_channels * kernel_size**2)
        if bias:
            self.bias = Parameter(
                init.init_basic.rand(
                    out_channels, low=-bound, high=bound, device=device, dtype=dtype
                )
            )
        else:
            self.bias = None
        self.padding = kernel_size // 2
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = x.transpose((1, 2)).transpose((2, 3))
        x = ops.conv(x, self.weight, self.stride, self.padding)
        x = x.transpose((3, 2)).transpose((2, 1))
        if self.bias is not None:
            x = x + self.bias.reshape((1, self.out_channels, 1, 1)).broadcast_to(
                x.shape
            )
        return x
        ### END YOUR SOLUTION


class ConvBN(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        self.conv = Conv(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            device=device,
            dtype=dtype,
        )
        self.bn = BatchNorm2d(dim=out_channels, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
