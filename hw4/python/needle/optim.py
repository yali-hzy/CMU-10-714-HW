"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for theta in self.params:
            data = theta.data
            if theta not in self.u:
                self.u[theta] = ndl.init.zeros_like(data)
            grad = theta.grad.data + self.weight_decay * data
            self.u[theta] = self.momentum * self.u[theta] + (1 - self.momentum) * grad
            theta.data = data - self.lr * self.u[theta]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        for theta in self.params:
            if theta.grad is not None:
                grad = theta.grad.data
                norm = np.sqrt(np.sum(grad.numpy() ** 2))
                if norm > max_norm:
                    grad *= max_norm / norm
                theta.grad.data = grad
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for theta in self.params:
            data = theta.data
            if theta not in self.m:
                self.m[theta] = ndl.init.zeros_like(data)
                self.v[theta] = ndl.init.zeros_like(data)

            grad = theta.grad.data + self.weight_decay * data.data
            self.m[theta] = self.beta1 * self.m[theta].data + (1 - self.beta1) * grad.data
            self.v[theta] = self.beta2 * self.v[theta].data + (1 - self.beta2) * (grad.data ** 2)

            m_hat = self.m[theta].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[theta].data / (1 - self.beta2 ** self.t)

            theta.data = data - (self.lr * m_hat.data) / (v_hat.data ** 0.5 + self.eps)
        ### END YOUR SOLUTION
