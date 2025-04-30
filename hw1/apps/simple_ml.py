"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filesname, 'rb') as f:
        # Read the magic number and number of images
        magic, num_images = struct.unpack('>II', f.read(8))
        if magic != 2051:
            raise ValueError("Invalid magic number in image file")
        # Read the dimensions of the images
        num_rows, num_cols = struct.unpack('>II', f.read(8))
        # Read the image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        X = data.reshape(num_images, num_rows * num_cols).astype(np.float32) / 255.0
    with gzip.open(label_filename, 'rb') as f:
        # Read the magic number and number of labels
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError("Invalid magic number in label file")
        # Read the label data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        y = data.reshape(num_labels).astype(np.uint8)
    # Check that the number of images and labels match
    if num_images != num_labels:
        raise ValueError("Number of images does not match number of labels")
    # Check that the number of rows and columns match the expected size
    if num_rows != 28 or num_cols != 28:
        raise ValueError("Expected image size of 28x28, but got {}x{}".format(num_rows, num_cols))
    # Check that the data is in the expected range
    if np.min(X) < 0.0 or np.max(X) > 1.0:
        raise ValueError("Data values should be in the range [0.0, 1.0]")
    # Check that the labels are in the expected range
    if np.min(y) < 0 or np.max(y) > 9:
        raise ValueError("Labels should be in the range [0, 9]")
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    logSumExp = ndl.log(ndl.summation(ndl.exp(Z), axes=(1,)))
    loss = logSumExp - ndl.summation(Z * y_one_hot, axes=(1,))
    loss = ndl.summation(loss) / Z.shape[0]
    return loss
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    for i in range(0, X.shape[0], batch):
        # Get the current batch
        X_batch = ndl.Tensor(X[i:i+batch], requires_grad=False)
        m = X_batch.shape[0]
        y_one_hot = np.zeros((m, W2.shape[1]))
        y_batch = y[i:i+batch]
        y_one_hot[np.arange(m), y_batch] = 1
        y_batch = ndl.Tensor(y_one_hot, requires_grad=False)

        Z1 = ndl.relu(ndl.matmul(X_batch, W1))
        Z = ndl.matmul(Z1, W2)
        loss = softmax_loss(Z, y_batch)
        loss.backward()

        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
