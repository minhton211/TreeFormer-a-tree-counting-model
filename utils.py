import math
import os
import shutil

import h5py
import numpy as np
from pathlib import Path
import scipy
import torch


def save_net(fname, net):
    """
    Save the state dictionary of a neural network to an HDF5 file.

    Args:
        fname (str): File path to save the model.
        net (torch.nn.Module): The neural network to save.
    """
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    """
    Load a state dictionary from an HDF5 file into a neural network.

    Args:
        fname (str): File path of the HDF5 file.
        net (torch.nn.Module): The neural network to load the weights into.
    """
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def load_net_CSRNet(fname, net):
    """
    Load CSRNet weights from a HDF5 file.
    Because the layer names in the HDF5 does not match those in TreeVision
    We manually map each name in the HDF5 file to the corresponding one

    Args:
        fname (str): File path of the HDF5 file.
        net (torch.nn.Module): The neural network to load the weights into.
    """
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            first, second = k[:k.rfind(".")], k[k.rfind(".")+1:]
            group = None
            if first == "backend.0":
                group = "backend.00.18457959676095714"
            elif first == "backend.2":
                group = "backend.20.8889406933022874"
            elif first == "backend.4":
                group = "backend.40.22368311931475648"
            elif first == "backend.6":
                group = "backend.60.4622938577847384"
            elif first == "backend.8":
                group = "backend.80.7188271697364912"
            elif first == "backend.10":
                group = "backend.100.5739362006476921"
            elif first == "output_layer":
                group = "output_layer0.3888687823989262"

            if second == "weight":
                dataset = "kernel:0"
            else:
                dataset = "bias:0"
            if group:
                param = torch.from_numpy(np.asarray(h5f[group][group][dataset]).T)
                v.copy_(param)


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth'):
    """
    Save a model checkpoint and optionally save it as the best.

    Args:
        state (dict): State dictionary containing model and optimizer states.
        is_best (bool): Whether this is the best model so far.
        task_id (str): Unique task identifier.
        filename (str): Base filename for the checkpoint.
    """
    checkpoint_path = os.path.join("weights", f"{task_id}_{filename}")
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join("weights", f"{task_id}_model_best.pth")
        shutil.copyfile(checkpoint_path, best_path)


def gaussian_filter_density(gt):
    """
    Generate density map using a Gaussian filter, with sigma based on neighboring distances.

    Args:
        gt (np.ndarray): Ground truth binary map.

    Returns:
        np.ndarray: Density map.
    """
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    # Find all non-zero points
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048

    # Build KDTree for spatial queries
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    distances, _ = tree.query(pts, k=4)

    if np.isinf(distances).any():
        print("Infinite distances detected. Skipping.")
        return density

    print("Generating density map...")
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0  # For single point
        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    print("Density map generation completed.")
    return density


def avg_box(labels, img_size):
    """
    Calculate the average box size for labeled data.

    Args:
        labels (list): List of bounding box coordinates (x, y, width, height).
        img_size (int): Size of the image (assumed square).

    Returns:
        int: Average box size as an equivalent diameter.
    """
    size = 0
    count = 0
    for x, y, width, height in labels:
        size += float(width) * img_size * float(height) * img_size
        count += 1
    return int(math.sqrt((size / count) / math.pi))


def prepare_datasets(dataset_path, dataset_type):
    """
    Prepares train and validation datasets based on the dataset type.

    Args:
        dataset_path (str): Path to the dataset root directory.
        dataset_type (str): Type of dataset ('yosemite_512', 'yosemite_1536', or 'london').

    Returns:
        tuple: Lists of train and validation image paths.
    """
    train_list, val_list = [], []

    if dataset_type in ["yosemite_512", "yosemite_1536"]:
        # Prepare Yosemite dataset
        for zone in ["zone_B", "zone_D"]:
            train_list += list(Path(dataset_path, zone, "images").rglob("*.jpg"))
        for zone in ["zone_A", "zone_C"]:
            val_list += list(Path(dataset_path, zone, "images").rglob("*.jpg"))
    elif dataset_type == "london":
        # Prepare London dataset
        train_list += list(Path(dataset_path, "train", "images").rglob("*.jpg"))
        train_list += list(Path(dataset_path, "val", "images").rglob("*.jpg"))
        val_list += list(Path(dataset_path, "test", "images").rglob("*.jpg"))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Check if dataset is valid
    if not train_list or not val_list:
        raise FileNotFoundError("Dataset folders are empty or paths are incorrect.")

    print(f"Training size: {len(train_list)}, Validation size: {len(val_list)}")
    return train_list, val_list
