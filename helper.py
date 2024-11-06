import os
import h5py

import numpy as np
from matplotlib import pyplot as plt

from utils import avg_box, gaussian_filter_density

class GenH5():
    def __init__(self, train_path: str, test_path: str, dataset_path: str, dataset: str):
        self.train_path = os.path.abspath(train_path)
        self.test_path = os.path.abspath(test_path)

        self._train_h5, self._valid_h5 = self.__create_hdf5(dataset_path, dataset)

    def __create_hdf5(self, dataset_path: str, dataset: str):
        """
        Create empty training and validation HDF5 files with placeholders
        for images and labels (density maps).

        Note:
        Datasets are saved in [dataset_name]/train.h5 and [dataset_name]/valid.h5.
        Existing files will be overwritten.

        Args:
            dataset_name: used to create a folder for train.h5 and valid.h5

        Returns:
            A tuple of pointers to training and validation HDF5 files.
        """
        # create output folder if it does not exist
        os.makedirs(dataset_path, exist_ok=True)

        # create HDF5 files: [dataset_name]/(train | valid).h5
        train_h5 = h5py.File(os.path.join(dataset_path, f'train_{dataset}.h5'), 'w')
        valid_h5 = h5py.File(os.path.join(dataset_path, f'valid_{dataset}.h5'), 'w')

        return train_h5, valid_h5

class Yosemite(GenH5):
    def __init__(self, train_path, test_path, dataset_path):
        super().__init__(train_path, test_path, dataset_path, "yosemite")

        self.train_images = []
        self.test_images = []

        train_zones = os.listdir(self.train_path)
        test_zones = os.listdir(self.test_path)

        for path in [os.path.join(self.train_path, zone) for zone in train_zones]:
            self.train_images += list([os.path.join(path, "images", file) for file in os.listdir(os.path.join(path, "images")) if file[-4:] == ".jpg"])

        for path in [os.path.join(self.test_path, zone) for zone in test_zones]:
            self.test_images += list([os.path.join(path, "images", file) for file in os.listdir(os.path.join(path, "images")) if file[-4:] == ".jpg"])

        self.train_size = len(self.train_images)
        self.test_size = len(self.test_images)

        self.X, self.Y, _ = plt.imread(self.train_images[0]).shape
        print(self.X, self.Y)
        print(self.train_size, self.test_size)

    def __generate_label(self, label_info: np.array):
        """
        Generate a density map based on objects positions.

        Args:
            label_info: (x, y) objects positions
            image_shape: (width, height) of a density map to be generated

        Returns:
            A density map.
        """
        image_shape = [self.Y, self.X]
        # create an empty density map
        label = np.zeros(image_shape, dtype=np.float32)

        # loop over objects positions and marked them with 100 on a label
        # note: *_ because some datasets contain more info except x, y coordinates
        for x, y in label_info:
            if y < image_shape[0] and x < image_shape[1]:
                label[int(y)][int(x)] = 1

        # apply a convolution with a Gaussian kernel
        # sigma = avg_box(label_info, image_shape[0])
        # label = gaussian_filter(label, sigma = 10)
        label = gaussian_filter_density(label)

        return label

    def __fill_h5(self, h5, label_path, train=True):
        """
        Save images and labels in given HDF5 file.

        Args:
            h5: HDF5 file
            label_path: path to label file
        """
        # source directory of the image

        labels = []

        with open(label_path, "r") as f:
            for tree in f.readlines():
                # _, x, y, width, height = tree.split(" ")
                x, y = tree.split(" ")
                # labels.append((int(float(x) * X), int(float(y) * Y), float(width), float(height)))
                labels.append((float(x), float(y)))

        # generate a density map by applying a Gaussian filter
        label = self.__generate_label(labels)

        # save data to HDF5 file
        h5.create_dataset(os.path.basename(label_path).replace(".txt", ""), (1, 1, *(self.X, self.Y)))
        h5[os.path.basename(label_path).replace(".txt", "")][0, 0] = label

    def gen(self):
        for i, img_path in enumerate(self.train_images):
            print("train", i)
            self.__fill_h5(self._train_h5, img_path.replace(".jpg", ".txt").replace("images", "labels"))

        self._train_h5.close()

        for i, img_path in enumerate(self.test_images):
            print("test", i)
            self.__fill_h5(self._valid_h5, img_path.replace(".jpg", ".txt").replace("images", "labels"), train=False)
        
        self._valid_h5.close()