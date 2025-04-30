from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.X = self.X.reshape(-1, 28, 28, 1)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        x = self.X[index]
        y = self.y[index]
        if self.transforms is not None:
            for transform in self.transforms:
                x = transform(x)
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION


import gzip
import struct
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