"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad * rhs * power(lhs, rhs - 1)
        rhs_grad = out_grad * log(lhs) * power(lhs, rhs)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = out_grad / rhs
        rhs_grad = -out_grad * lhs / (rhs * rhs)
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is None:
            axes[-1], axes[-2] = axes[-2], axes[-1]
        else:
            axes[self.axes[0]], axes[self.axes[1]] = (
                axes[self.axes[1]],
                axes[self.axes[0]],
            )
        return a.permute(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        len_diff = len(self.shape) - len(node.inputs[0].shape)
        shape = (1,) * len_diff + node.inputs[0].shape
        axes = []
        for i in range(len(shape)):
            if shape[i] == 1 and out_grad.shape[i] != 1:
                axes.append(i)
        out_grad = summation(out_grad, axes=tuple(axes))
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = tuple(range(len(a.shape)))
        elif isinstance(self.axes, Number):
            axes = (self.axes,)
        else:
            axes = self.axes
        for axis in axes:
            a = array_api.sum(a, axis=axis, keepdims=True)
        shape = [s for i, s in enumerate(a.shape) if i not in axes]
        return array_api.reshape(a, tuple(shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            out_grad = reshape(out_grad, (1,) * len(node.inputs[0].shape))
        elif isinstance(self.axes, Number):
            shape = list(out_grad.shape)
            shape.insert(self.axes, 1)
            out_grad = reshape(out_grad, tuple(shape))
        else:
            shape = list(out_grad.shape)
            axes = self.axes
            if not isinstance(axes, (list, tuple)):
                axes = (axes,)
            for i in self.axes:
                shape.insert(i, 1)
            out_grad = reshape(out_grad, tuple(shape))
        return broadcast_to(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lhs_grad = matmul(out_grad, transpose(rhs))
        rhs_grad = lhs.transpose() @ out_grad
        if lhs_grad.shape != lhs.shape:
            lhs_grad = summation(
                lhs_grad, axes=tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            )
        if rhs_grad.shape != rhs.shape:
            rhs_grad = summation(
                rhs_grad, axes=tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            )
        return lhs_grad, rhs_grad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_cache = out_grad.realize_cached_data()
        a = node.inputs[0].realize_cached_data()
        return Tensor(out_cache * (a > 0), device=out_grad.device, dtype=out_grad.dtype)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ones = Tensor(
            array_api.full(
                node.inputs[0].shape,
                1,
                dtype=node.inputs[0].dtype,
                device=node.inputs[0].device,
            ),
            device=node.inputs[0].device,
            dtype=node.inputs[0].dtype,
            requires_grad=False,
        )
        return out_grad * (ones - power_scalar(tanh(node.inputs[0]), 2))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        shape.insert(self.axis, len(args))
        stacked_tensor = array_api.empty(
            shape, dtype=args[0].dtype, device=args[0].device
        )
        for i, arg in enumerate(args):
            slices = [slice(None)] * len(shape)
            slices[self.axis] = i
            stacked_tensor[tuple(slices)] = arg
        return stacked_tensor
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = list(A.shape)
        split_size = shape[self.axis]
        split_tensors = []
        new_shape = list(shape)
        new_shape.pop(self.axis)
        for i in range(split_size):
            slices = [slice(None)] * len(shape)
            slices[self.axis] = i
            split_tensors.append(
                A[tuple(slices)].compact().reshape(tuple(new_shape)),
            )
        return tuple(split_tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        for i in self.axes:
            shape[i] *= self.dilation + 1
        dilated_tensor = array_api.full(
            shape,
            0,
            dtype=a.dtype,
            device=a.device,
        )
        slices = [slice(None) for _ in range(len(shape))]
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)
        dilated_tensor[tuple(slices)] = a
        return dilated_tensor
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = list(a.shape)
        for i in self.axes:
            shape[i] //= self.dilation + 1
        slices = [slice(None) for _ in range(len(shape))]
        for axis in self.axes:
            slices[axis] = slice(None, None, self.dilation + 1)
        return a[tuple(slices)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, axes=self.axes, dilation=self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        A = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        )
        N, H, W, C_in = A.shape
        Kh, Kw, _, C_out = B.shape
        Ns, Hs, Ws, Cs = A.strides
        inner_dim = Kh * Kw * C_in
        newH = (H - Kh) // self.stride + 1
        newW = (W - Kw) // self.stride + 1
        if BACKEND == "np":
            A = array_api.lib.stride_tricks.as_strided(
                A,
                shape=(N, newH, newW, Kh, Kw, C_in),
                strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
            ).reshape(-1, inner_dim)
        elif BACKEND == "nd":
            A = (
                A.as_strided(
                    shape=(N, newH, newW, Kh, Kw, C_in),
                    strides=(Ns, Hs * self.stride, Ws * self.stride, Hs, Ws, Cs),
                )
                .compact()
                .reshape((N * newH * newW, inner_dim))
            )
        out = A @ B.compact().reshape((inner_dim, C_out))
        return out.reshape((N, newH, newW, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        out_grad = dilate(out_grad, axes=(1, 2), dilation=self.stride - 1)
        K, _, _, _ = B.shape

        W_flip = flip(B, axes=(0, 1)).transpose((2, 3))
        X_grad = conv(out_grad, W_flip, padding=K - 1 - self.padding)

        W_grad = (
            conv(
                A.transpose(axes=(0, 3)),
                out_grad.transpose(axes=(0, 1)).transpose(axes=(1, 2)),
                padding=self.padding,
            )
            .transpose(axes=(0, 1))
            .transpose(axes=(1, 2))
        )

        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
