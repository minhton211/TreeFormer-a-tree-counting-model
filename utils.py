import math
import os
import shutil

import h5py
import numpy as np
import scipy
import torch


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def load_net_h5(fname, net):
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
    torch.save(state, os.path.join("Density_Map", "ICCE", "weights", f"{task_id}_{filename}"))
    if is_best:
        shutil.copyfile(os.path.join("Density_Map", "ICCE", "weights", f"{task_id}_{filename}"), 
                        os.path.join("Density_Map", "ICCE", "weights", f"{task_id}_model_best.pth"))


def gaussian_filter_density(gt):
    '''
    Implementation of geometric-adaptive Gaussian Filter from 
    https://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf
    '''
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    if np.isinf(distances).any():
        print("Infinit distances detected. Skipping.")
        return density

    print("generate density...")
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    print("done")
    return density


def avg_box(labels, img_size):
    size = 0
    count = 0
    for x, y, width, height in labels:
        size += float(width)*img_size * float(height)*img_size
        count += 1
    return int(math.sqrt((size/count)/3.14))
