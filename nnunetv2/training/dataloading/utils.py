from __future__ import annotations
import multiprocessing
import os
from typing import List
from pathlib import Path
from warnings import warn
from scipy.ndimage import gaussian_filter

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import isfile, subfiles
from nnunetv2.configuration import default_num_processes
from scipy import ndimage

import nibabel as nib
import cc3d
import numpy as np
# from cucim.core.operations import morphology
from nnunetv2.training.dataloading.nnInteractive_clicks import PointInteraction_stub
import torch



def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, npz_content['data'])

        if unpack_segmentation and (overwrite_existing or not isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", npz_content['seg'])

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if isfile(data_npy):
            os.remove(data_npy)
        if isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


def unpack_dataset(folder: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                   num_processes: int = default_num_processes,
                   verify: bool = False):
    """
    all npz files in this folder belong to the dataset, unpack them all
    """
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        npz_files = subfiles(folder, True, None, ".npz", True)
        p.starmap(_convert_to_npy, zip(npz_files,
                                       [unpack_segmentation] * len(npz_files),
                                       [overwrite_existing] * len(npz_files),
                                       [verify] * len(npz_files))
                  )


def preprocess_point(point, data_properties, shape):
        """
        Preprocess the points to map them to the correct coordinate system.
        I.e. from the original image space to the cropped/resized image space.
        """
        point = [float(i) for i in point]
        x, y, z = point
        bbox_used_for_cropping = data_properties["bbox_used_for_cropping"]
        shape_after_cropping_and_before_resampling = data_properties["shape_after_cropping_and_before_resampling"]
        # Adapt the centroid to cropped data
        x = max(0, x - bbox_used_for_cropping[2][0])
        x = min(shape_after_cropping_and_before_resampling[2], x - bbox_used_for_cropping[2][0])
        y = max(0, y - bbox_used_for_cropping[1][0])
        y = min(shape_after_cropping_and_before_resampling[1], y - bbox_used_for_cropping[1][0])
        z = max(0, z - bbox_used_for_cropping[0][0])
        z = min(shape_after_cropping_and_before_resampling[0], z - bbox_used_for_cropping[0][0])

        # Adjust for resampling
        factor = [shape[i] / shape_after_cropping_and_before_resampling[i] for i in range(3)]
        x = np.round(x * factor[2]).astype(np.int16)
        y = np.round(y * factor[1]).astype(np.int16)
        z = np.round(z * factor[0]).astype(np.int16)
        return [z, y, x]

# Experimental speedup tech
from scipy.ndimage import convolve1d

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    x = np.arange(-size // 2 + 1., size // 2 + 1.)
    kernel = np.exp(-x**2 / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def separable_gaussian_3d(arr: np.ndarray, sigma: float) -> np.ndarray:
    size = int(2 * np.ceil(3 * sigma) + 1)
    kernel = gaussian_kernel(size, sigma)
    arr = convolve1d(arr, kernel, axis=0, mode='constant')
    arr = convolve1d(arr, kernel, axis=1, mode='constant')
    arr = convolve1d(arr, kernel, axis=2, mode='constant')
    return arr


import time

def sparse_to_dense_point_gauss_timed(points: dict[str, np.ndarray], shape: tuple[int, ...], properties: dict, sigma: float = 1.0) -> np.ndarray:
    start_time = time.time()  # Start timing
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            coord = preprocess_point(coord, properties, shape)
            if label == 'tumor':
                pos_clicks[*coord] = 1.0
            elif label == 'background':
                neg_clicks[*coord] = 1.0 # self.place_point(coord, neg_clicks, n_clck + 1)
            else:
                raise ValueError(f"Unknown label {label} in click json")
        start_gaussioan_filter_time = time.time()  # Start timing for Gaussian filter
        pos_clicks = separable_gaussian_3d(pos_clicks, sigma=sigma)
        neg_clicks = separable_gaussian_3d(neg_clicks, sigma=sigma)
        print(f"Gaussian filter execution time: {time.time() - start_gaussioan_filter_time:.4f} seconds")
    end_time = time.time()  # End timing
    print(f"sparse_to_dense_point_gauss execution time: {end_time - start_time:.4f} seconds")
    return pos_clicks, neg_clicks

def restructure_clicks(click_json):
    """
    Restructure the clicks to a format that is easier to work with.
    from {"lesion": [[x,y,z] ,[x,y,z]], "background": [[x,y,z] ,[x,y,z]]}
    """
    clicks = {'points': []}
    for label in click_json:
        for coords in click_json[label]:
            if label == 'lesion':
                label = 'tumor'
            singe_point = {'point': coords, 'name': label}
            clicks["points"].append(singe_point)
    return clicks


def simulate_clicks(input_label, input_liver, center_offset: int = None, edge_offset: int = None, use_gpu: bool = False) -> dict[str, List[List[int]]]:


    SEED = 42
    np.random.seed(SEED)
    #cp.random.seed(SEED)

    label_im = input_label

    clicks = {'lesion':[], 'background': []}


    if np.sum(label_im) == 0:
        print("[WARNING] GT is empty, generating background clicks only!")
    else:
        ##### Tumor Clicks #####
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(10, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        # Sample center clicks for 10 random components
        for label in sampled_labels:
            labeled_mask = connected_components == label
            labeled_mask = np.array(labeled_mask)
            if use_gpu:
                # not implemented yet error
                raise NotImplementedError("GPU-based EDT computation is not implemented yet.")
                # edt = morphology.distance_transform_edt(labeled_mask)
            else:


                edt = ndimage.morphology.distance_transform_edt(labeled_mask)
                edt = np.array(edt)
                # labeled_mask = np.array(labeled_mask)


            center = np.unravel_index(np.argmax(edt), edt.shape)
            if center_offset is not None:
                center = perturb_click(center_offset, center, label_im)

            clicks['lesion'].append([int(center[0]), int(center[1]), int(center[2])])
            assert label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['lesion'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        while n_clicks < 10:
            for label in sampled_labels:
                labeled_mask = connected_components == label
                labeled_mask = np.array(labeled_mask)
                if use_gpu:
                    #not implemented yet error
                    raise NotImplementedError("GPU-based EDT computation is not implemented yet.")
                    # edt = morphology.distance_transform_edt(labeled_mask)
                else:

                    edt = ndimage.morphology.distance_transform_edt(labeled_mask)
                    edt = np.array(edt)
                    labeled_mask = np.array(labeled_mask)
                edt_inverted = (np.max(edt) - edt) * (edt > 0)
                boundary_elements = (edt_inverted == np.max(edt_inverted)) * (labeled_mask > 0)
                indices = np.array(np.nonzero(boundary_elements)).T  # Shape: (num_true, ndim)
                if indices.shape[0] == 0:
                    print("wat?")
                boundary_click = indices[np.random.choice(indices.shape[0])]

                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, label_im)

                clicks['lesion'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == 10:
                    break

    ##### Background Clicks #####
    in_liver = input_liver
    if np.sum(in_liver) == 0:
        print("[WARNING] Liver mask is empty no bg clicks")
        clicks['background'] = []
    else:
        bg_clicks = uniform_sample_coordinates(in_liver, label_im) # sample non-tumor clicks in the liver
        clicks['background'] = bg_clicks

    return clicks

def perturb_click(offset, click, label_im):
    import random
    random_offset = [random.randint(0, int(offset)) for _ in range(3)]
    try:
        if label_im[
            int(click[0] + random_offset[0]),
            int(click[1] + random_offset[1]),
            int(click[2] + random_offset[2])
        ]:
            return [
                int(click[0] + random_offset[0]),
                int(click[1] + random_offset[1]),
                int(click[2] + random_offset[2])
            ]
        else:
            # Fallback to original click if perturbed click is invalid
            return [int(click[0]), int(click[1]), int(click[2])]
    except IndexError:
        # If the perturbed click is out of bounds, return the original click
        return [int(click[0]), int(click[1]), int(click[2])]

def uniform_sample_coordinates(in_liver, label_im, num_samples=10):
    """
    Samples `num_samples` coordinates from `in_liver` where `label_im == 0`.

    Args:
        in_liver (numpy.ndarray): 3D binary mask of the liver (1 = liver, 0 = outside).
        label_im (numpy.ndarray): 3D label image where 0 indicates the region of interest.
        num_samples (int): Number of points to sample.

    Returns:
        numpy.ndarray: Array of shape (num_samples, 3) containing sampled coordinates.
    """
    # Get coordinates where the liver is present and label_im == 0

    valid_coords = np.argwhere((in_liver == 1) & (label_im == 0))

    # Ensure there are points to sample from
    if len(valid_coords) == 0:
        raise ValueError("No valid voxels found where in_liver == 1 and label_im == 0.")

    # Randomly sample from available coordinates
    sampled_indices = np.random.choice(len(valid_coords), size=min(num_samples, len(valid_coords)), replace=False)
    sampled_coords = valid_coords[sampled_indices]
    sampled_coords = [[int(el[0]), int(el[1]), int(el[2])] for el in sampled_coords]

    return sampled_coords


def sparse_to_dense_point_gauss(points: dict[str, np.ndarray], shape: tuple[int, ...], properties: dict, sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            coord = preprocess_point(coord, properties, shape)
            if label == 'tumor':
                pos_clicks[*coord] = 1.0
            elif label == 'background':
                neg_clicks[*coord] = 1.0 # self.place_point(coord, neg_clicks, n_clck + 1)
            else:
                raise ValueError(f"Unknown label {label} in click json")
        pos_clicks = separable_gaussian_3d(pos_clicks, sigma=sigma)
        neg_clicks = separable_gaussian_3d(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks

def select_num_points_exp(alpha: float = 0.2, max_points: int = 11) -> tuple[int, int]:
    """
    Sample a number of points from an exponential distribution.
    """
    # Create a decaying probability distribution over values 0 to 10
    values = np.arange(max_points)
    alpha = alpha # Controls how steeply the distribution decays; higher = more skewed
    prob_dist = np.exp(-alpha * values)
    prob_dist /= prob_dist.sum()  # Normalize to sum to 1

    # Sample using the computed probability distribution
    num_pos_clicks = np.random.choice(values, p=prob_dist)
    num_neg_clicks = np.random.choice(values, p=prob_dist)
    return num_pos_clicks, num_neg_clicks


def generated_sparse_to_dense_point_gauss(clicks: dict, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(clicks["background"]) > 0:
        for clck in clicks["lesion"]:
            pos_clicks[*clck] = 1.0
        for clck in clicks["background"]:
            neg_clicks[*clck] = 1.0
        pos_clicks = gaussian_filter(pos_clicks, sigma=sigma)
        neg_clicks = gaussian_filter(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks

def generated_sparse_to_dense_point_nnInteractive(clicks: dict, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    point_interaction = PointInteraction_stub(point_radius=sigma, use_distance_transform=True)
    if len(clicks["background"]) > 0:
        for clck in clicks["lesion"]:
            pos_clicks = point_interaction.place_point(clck, pos_clicks, binarize=False)
        for clck in clicks["background"]:
            neg_clicks = point_interaction.place_point(clck, neg_clicks, binarize=False)
    return pos_clicks, neg_clicks

def sparse_to_dense_point_nnInteractive(points: dict[str, np.ndarray], shape: tuple[int, ...], properties: dict, sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = torch.zeros(shape, dtype=torch.float32), torch.zeros(shape, dtype=torch.float32)
    point_interaction = PointInteraction_stub(point_radius=sigma, use_distance_transform=True)
    if len(points) > 0:
        for clck in points:
            coord = clck['point']
            label = clck['name']
            coord = preprocess_point(coord, properties, shape)
            if label == 'tumor':
                pos_clicks = point_interaction.place_point(coord, pos_clicks, binarize=False)
            elif label == 'background':
                neg_clicks = point_interaction.place_point(coord, neg_clicks, binarize=False)
            else:
                raise ValueError(f"Unknown label {label} in click json")
    return pos_clicks, neg_clicks


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')