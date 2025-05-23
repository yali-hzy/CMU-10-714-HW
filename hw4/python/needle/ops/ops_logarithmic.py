from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.maxZ = Z.max(axis=-1, keepdims=True)
        shiftedZ = Z - self.maxZ.broadcast_to(Z.shape)
        return Z - (
            array_api.log(
                array_api.sum(array_api.exp(shiftedZ), axis=-1, keepdims=True)
            )
            + self.maxZ
        ).broadcast_to(Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)
        shape[-1] = 1
        maxZ = Tensor(
            self.maxZ,
            device=node.inputs[0].device,
            dtype=node.inputs[0].dtype,
            requires_grad=False,
        )
        shiftedZ = node.inputs[0] - maxZ.broadcast_to(node.inputs[0].shape)
        expZ = exp(shiftedZ)
        sumExpZ = (
            summation(expZ, axes=(-1,))
            .reshape(shape)
            .broadcast_to(node.inputs[0].shape)
        )
        sum_out_grad = (
            summation(out_grad, axes=(-1,))
            .reshape(shape)
            .broadcast_to(node.inputs[0].shape)
        )
        gradLogSumExp = sum_out_grad * expZ / sumExpZ
        return out_grad - gradLogSumExp
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.maxZ = Z.max(axis=self.axes, keepdims=True)
        Z = Z - array_api.broadcast_to(self.maxZ, Z.shape)
        logSumExpZ = (
            array_api.log(
                array_api.sum(array_api.exp(Z), axis=self.axes, keepdims=True)
            )
            + self.maxZ
        )
        return array_api.sum(logSumExpZ, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = list(node.inputs[0].shape)
        if self.axes is None:
            shape = (1,) * len(shape)
        else:
            for i in self.axes:
                shape[i] = 1
        out_grad = reshape(out_grad, shape)
        out_grad = broadcast_to(out_grad, node.inputs[0].shape)
        maxZ = Tensor(
            self.maxZ,
            device=node.inputs[0].device,
            dtype=node.inputs[0].dtype,
            requires_grad=False,
        )
        shiftedZ = node.inputs[0] - maxZ.broadcast_to(node.inputs[0].shape)
        expZ = exp(shiftedZ)
        sumExpZ = (
            summation(expZ, axes=self.axes)
            .reshape(shape)
            .broadcast_to(node.inputs[0].shape)
        )
        out_grad = out_grad * expZ / sumExpZ
        return out_grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
