"""The module."""

from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = (1,) * len(x.shape)
        return Tensor(1, device=x.device, dtype=x.dtype, requires_grad=False).reshape(
            shape
        ).broadcast_to(x.shape) / (
            Tensor(1, device=x.device, dtype=x.dtype, requires_grad=False)
            .reshape(shape)
            .broadcast_to(x.shape)
            + ops.exp(-x)
        )
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.init_basic.rand(
                input_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.init_basic.rand(
                hidden_size,
                hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.init_basic.rand(
                    hidden_size, low=-bound, high=bound, device=device, dtype=dtype
                )
            )
            self.bias_hh = Parameter(
                init.init_basic.rand(
                    hidden_size, low=-bound, high=bound, device=device, dtype=dtype
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        if nonlinearity == "tanh":
            self.nonlinearity = ops.tanh
        elif nonlinearity == "relu":
            self.nonlinearity = ops.relu
        else:
            raise ValueError("nonlinearity must be either 'tanh' or 'relu'")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h = init.init_basic.zeros(
                X.shape[0],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
        h = X @ self.W_ih + h @ self.W_hh
        if self.bias_ih is not None:
            h += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to(h.shape)
        if self.bias_hh is not None:
            h += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to(h.shape)
        h = self.nonlinearity(h)
        return h
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        nonlinearity="tanh",
        device=None,
        dtype="float32",
    ):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = []
        for i in range(num_layers):
            if i == 0:
                input_size_ = input_size
            else:
                input_size_ = hidden_size
            self.rnn_cells.append(
                RNNCell(
                    input_size_,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype,
                )
            )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h0 is None:
            h0 = init.init_basic.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
        h = list(ops.split(h0, axis=0))
        output = []
        X = ops.split(X, axis=0)
        for i in range(len(X)):
            x = X[i]
            for j in range(self.num_layers):
                h[j] = self.rnn_cells[j].forward(x, h[j])
                x = h[j]
            output.append(h[-1])
        output = ops.stack(output, axis=0)
        h = ops.stack(h, axis=0)
        return output, h
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(
        self, input_size, hidden_size, bias=True, device=None, dtype="float32"
    ):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        bound = np.sqrt(1 / hidden_size)
        self.W_ih = Parameter(
            init.init_basic.rand(
                input_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        self.W_hh = Parameter(
            init.init_basic.rand(
                hidden_size,
                4 * hidden_size,
                low=-bound,
                high=bound,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.init_basic.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.init_basic.rand(
                    4 * hidden_size,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias_ih = None
            self.bias_hh = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = init.init_basic.zeros(
                X.shape[0],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            c0 = init.init_basic.zeros(
                X.shape[0],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
        else:
            h0, c0 = h
        gates = X @ self.W_ih + h0 @ self.W_hh
        if self.bias_ih is not None:
            gates += self.bias_ih.reshape((1, 4 * self.hidden_size)).broadcast_to(
                gates.shape
            )
        if self.bias_hh is not None:
            gates += self.bias_hh.reshape((1, 4 * self.hidden_size)).broadcast_to(
                gates.shape
            )
        i, f, g, o = ops.split(gates.reshape((X.shape[0], 4, self.hidden_size)), axis=1)
        i = Sigmoid()(i)
        f = Sigmoid()(f)
        g = ops.tanh(g)
        o = Sigmoid()(o)
        c = f * c0 + i * g
        h = o * ops.tanh(c)
        return h, c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        device=None,
        dtype="float32",
    ):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.lstm_cells = []
        for i in range(num_layers):
            if i == 0:
                input_size_ = input_size
            else:
                input_size_ = hidden_size
            self.lstm_cells.append(
                LSTMCell(
                    input_size_,
                    hidden_size,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            )
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.device = device
        self.dtype = dtype
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        if h is None:
            h0 = init.init_basic.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
            c0 = init.init_basic.zeros(
                self.num_layers,
                X.shape[1],
                self.hidden_size,
                device=self.device,
                dtype=self.dtype,
                requires_grad=True,
            )
        else:
            h0, c0 = h
        h = list(ops.split(h0, axis=0))
        c = list(ops.split(c0, axis=0))
        output = []
        X = ops.split(X, axis=0)
        for i in range(len(X)):
            x = X[i]
            for j in range(self.num_layers):
                h[j], c[j] = self.lstm_cells[j].forward(x, (h[j], c[j]))
                x = h[j]
            output.append(h[-1])
        output = ops.stack(output, axis=0)
        h = ops.stack(h, axis=0)
        c = ops.stack(c, axis=0)
        return output, (h, c)
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
