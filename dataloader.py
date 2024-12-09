import os
import random

import cv2
import h5py
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


def load_data(img_path, dataset, train=True):
    """
    Load image and corresponding ground truth (target) data.

    Args:
        img_path (str): Path to the image file.
        dataset (str): Name of the dataset (used to locate YAML file).
        train (bool): Flag to indicate whether loading training or validation data.

    Returns:
        tuple: The image (PIL.Image) and the processed ground truth target (numpy array).
    """
    # Load dataset configuration from the YAML file
    with open('{}.yaml'.format(dataset), "r") as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    # Get ground truth file path based on train/val mode
    gt_path = data_dict["train"] if train else data_dict["val"]

    # Load and convert the image to RGB
    img = Image.open(img_path).convert('RGB')

    # Read the ground truth data
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file[os.path.basename(img_path).replace(".jpg", "")][0][0])

    # Resize target and scale values
    target = cv2.resize(target, (int(target.shape[1] / 8), int(target.shape[0] / 8)), interpolation=cv2.INTER_CUBIC) * 64

    return img, target


class listDataset(Dataset):
    """
    Custom PyTorch Dataset for loading image data and corresponding targets.

    Args:
        root (list): List of image file paths.
        dataset (str): Name of the dataset (used in load_data).
        shape (tuple): Desired shape for images 
        shuffle (bool): Flag to shuffle data (only for training).
        transform (callable): Transform to be applied to images.
        train (bool): Flag to indicate training mode.
        seen (int): Number of samples seen (used for batch normalization or logging).
        batch_size (int): Batch size 
        num_workers (int): Number of workers 
    """
    def __init__(self, root, dataset=None, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root * 4  # Augment training data by repeating it 4 times
        random.shuffle(root)  # Shuffle the data

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = dataset

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return self.nSamples

    def __getitem__(self, index):
        """
        Fetch a single sample from the dataset.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: The image (transformed if applicable) and the corresponding target.
        """
        assert index <= len(self), 'Index out of range'

        # Get image path at the given index
        img_path = self.lines[index]

        # Load image and target using the load_data function
        img, target = load_data(img_path, self.dataset, self.train)

        # Apply transformations if specified
        if self.transform is not None:
            img = self.transform(img)

        return img, target
