import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None,
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """

        ### BEGIN YOUR SOLUTION
        def unpickle(file):
            import pickle

            with open(file, "rb") as fo:
                dict = pickle.load(fo, encoding="bytes")
            return dict

        if train:
            self.X = np.empty((0, 3, 32, 32), dtype=np.uint8)
            self.y = np.empty((0,), dtype=np.uint8)
            for i in range(1, 6):
                batch = unpickle(os.path.join(base_folder, f"data_batch_{i}"))
                self.X = np.append(
                    self.X, batch[b"data"].reshape(-1, 3, 32, 32), axis=0
                )
                self.y = np.append(self.y, batch[b"labels"])
        else:
            batch = unpickle(os.path.join(base_folder, "test_batch"))
            self.X = batch[b"data"].reshape(-1, 3, 32, 32)
            self.y = np.array(batch[b"labels"], dtype=np.uint8)
        self.p = p
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        return self.apply_transforms(self.X[index]), self.y[index]
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
