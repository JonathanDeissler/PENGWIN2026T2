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

import edt as fastedt
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


def simulate_clicks(input_label, input_liver, pos_click_budget=10, neg_click_budget=10, center_offset: int = None, edge_offset: int = None, use_gpu: bool = False) -> dict[str, List[List[int]]]:


    SEED = 42
    np.random.seed(SEED)
    #cp.random.seed(SEED)

    label_im = input_label

    clicks = {'lesion':[], 'background': []}


    if np.sum(label_im) == 0:
        pass
        # print("[WARNING] GT is empty, generating background clicks only!")
    else:
        ##### Tumor Clicks #####
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:] # Skip background label 0
        size = min(pos_click_budget, len(unique_labels))
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

                edt = fastedt.edt(labeled_mask)

            center = np.unravel_index(np.argmax(edt), edt.shape)
            if center_offset is not None:
                center = perturb_click(center_offset, center, label_im)

            clicks['lesion'].append([int(center[0]), int(center[1]), int(center[2])])
            assert label_im[int(center[0]), int(center[1]), int(center[2])]
        n_clicks = len(clicks['lesion'])

        # Sample boundary clicks if center clicks were not enough to fill the click budget (n=10)
        while n_clicks < pos_click_budget:
            for label in sampled_labels:
                labeled_mask = connected_components == label
                labeled_mask = np.array(labeled_mask)
                if use_gpu:
                    #not implemented yet error
                    raise NotImplementedError("GPU-based EDT computation is not implemented yet.")
                    # edt = morphology.distance_transform_edt(labeled_mask)
                else:
                    edt = fastedt.edt(labeled_mask)
                    # boundary_elements = (edt_inverted == np.max(edt_inverted)) * (labeled_mask > 0)
                    # indices = np.array(np.nonzero(boundary_elements)).T  # Shape: (num_true, ndim)
                    #
                    # edt = ndimage.morphology.distance_transform_edt(labeled_mask)
                    # edt = np.array(edt)
                    # labeled_mask = np.array(labeled_mask)
                edt_inverted = (np.max(edt) - edt) * (edt > 0)
                boundary_elements = (edt_inverted == np.max(edt_inverted)) * (labeled_mask > 0)
                indices = np.array(np.nonzero(boundary_elements)).T  # Shape: (num_true, ndim)
                # center = np.unravel_index(np.argmax(edt), edt.shape)
                #
                # import napari
                # viewer = napari.Viewer()
                # viewer.add_image(edt, name='edt')
                # viewer.add_points(center, name='center', size=2, face_color='red')
                # viewer.add_image(edt_inverted, name='inverted')
                # viewer.add_points(indices, name='boundary', size=2, face_color='blue')
                # napari.run()

                if indices.shape[0] == 0:
                    print("wat?")
                boundary_click = indices[np.random.choice(indices.shape[0])]

                if edge_offset is not None:
                    boundary_click = perturb_click(edge_offset, boundary_click, label_im)

                clicks['lesion'].append([int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])])
                assert label_im[int(boundary_click[0]), int(boundary_click[1]), int(boundary_click[2])]
                n_clicks += 1
                if n_clicks == pos_click_budget:
                    break

    ##### Background Clicks #####
    in_liver = input_liver
    if np.sum(in_liver) == 0:
        # print("[WARNING] Liver mask is empty no bg clicks")
        clicks['background'] = []
    else:
        bg_clicks = uniform_sample_coordinates(in_liver, label_im,neg_click_budget) # sample non-tumor clicks in the liver
        clicks['background'] = bg_clicks

    return clicks

# def perturb_click(offset, click, label_im):
#     import random
#     random_offset = [random.randint(0, int(offset)) for _ in range(3)]
#     try:
#         if label_im[
#             int(click[0] + random_offset[0]),
#             int(click[1] + random_offset[1]),
#             int(click[2] + random_offset[2])
#         ]:
#             return [
#                 int(click[0] + random_offset[0]),
#                 int(click[1] + random_offset[1]),
#                 int(click[2] + random_offset[2])
#             ]
#         else:
#             # Fallback to original click if perturbed click is invalid
#             return [int(click[0]), int(click[1]), int(click[2])]
#     except IndexError:
#         # If the perturbed click is out of bounds, return the original click
#         return [int(click[0]), int(click[1]), int(click[2])]

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
    if len(clicks["lesion"]) > 0:
        for clck in clicks["lesion"]:
            pos_clicks = point_interaction.place_point(clck, pos_clicks, binarize=False)
    if len(clicks["background"]) > 0:
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


###### place precomputed click representations in the click maps #######

def place_precomputed_clicks(clicks: dict[str, np.ndarray],precomputed_point_reprentation, map_shape, device):

    """
    Place precomputed click representations in the click maps.
    Args:
        clicks (dict): Dictionary containing 'lesion' and 'background' keys with lists of click coordinates.
        precomputed_point_reprentation (torch.Tensor): Precomputed point representation for clicks.
        map_shape (tuple): Shape of the output click maps.
    Returns:
        pos_clicks (torch.Tensor): Click map for positive clicks.
        neg_clicks (torch.Tensor): Click map for negative clicks.
    """
    pos_clicks = torch.zeros(map_shape, dtype=torch.float32, device=device)
    neg_clicks = torch.zeros(map_shape, dtype=torch.float32, device=device)

    for clck in clicks['lesion']:
        pos_clicks = do_place_at_location(clck, precomputed_point_reprentation, pos_clicks)
    for clck in clicks['background']:
        neg_clicks = do_place_at_location(clck, precomputed_point_reprentation, neg_clicks)

    return pos_clicks.unsqueeze(0), neg_clicks.unsqueeze(0)


def select_interactions_based_on_epochs(interactions , current_epoch, num_epochs, max_interactions=10, increase_every=None ):
    """
    Select interactions based on the current epoch and total number of epochs.
    Args:
        interactions (list): List of interactions to select from.
        current_epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    Returns:
        selected_interactions (list): Interactions selected for the current epoch.
    """
    if not isinstance(interactions, list):
        raise ValueError("interactions must be a list")

    if current_epoch < 0 or current_epoch >= num_epochs:
        raise ValueError("current_epoch must be between 0 and num_epochs - 1")

    if increase_every is None:
        increase_every = num_epochs // max_interactions

    if increase_every == 0:
        current_ammount = 0
    else:
        current_ammount = np.min((current_epoch // increase_every, max_interactions))

    for b in range(len(interactions)):
        if interactions[b]['lesion'] != []:
            interactions[b]['lesion'] = interactions[b]['lesion'][:np.min((current_ammount -1,len(interactions[b]['lesion'])-1))]
        if interactions[b]['background'] != []:
            interactions[b]['background'] = interactions[b]['background'][:np.min((current_ammount -1,len(interactions[b]['background'])-1))]

    return interactions

def do_place_at_location(click, precomputed_point_reprentation, interaction_map):


    big_shape = interaction_map.shape
    small_shape = precomputed_point_reprentation.shape
    offset = [s // 2 for s in small_shape]

    # Compute start and end indices for big
    start_big = [max(0, l - o) for l, o in zip(click, offset)]
    end_big = [min(dim, l - o + s) for l, o, s, dim in zip(click, offset, small_shape, big_shape)]

    # Corresponding start/end for small
    start_small = [max(0, o - l) if l - o < 0 else 0 for l, o in zip(click, offset)]
    end_small = [start + (end - st) for start, end, st in zip(start_small, end_big, start_big)]

    # Now, slice
    region_big = interaction_map[start_big[0]:end_big[0], start_big[1]:end_big[1], start_big[2]:end_big[2]]
    region_small = precomputed_point_reprentation[start_small[0]:end_small[0], start_small[1]:end_small[1], start_small[2]:end_small[2]]

    # In-place max
    region_big.copy_(torch.maximum(region_big, region_small))

    return interaction_map
####### Simulate clicks advanced ########



def random_point_within_mask(mask):
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return None
    idx = np.random.choice(coords.shape[0])
    return tuple(coords[idx])


def perturb_click(offset, center, mask):
    max_attempts = 10

    for _ in range(max_attempts):
        perturbed = np.array(center) + np.random.randint(-offset, offset + 1, size=3)
        perturbed = np.clip(perturbed, 0, np.array(mask.shape) - 1)
        if mask[tuple(perturbed)]:
            return tuple(perturbed)
    return center


def sample_point_within_region(mask, edt_weight=0.7):
    # Mix of center and random sampling
    edt = fastedt.edt(mask.astype(np.uint8))
    edt = edt / (np.max(edt) + 1e-5)
    noise = np.random.rand(*mask.shape)
    score = edt_weight * edt + (1 - edt_weight) * noise
    score *= mask
    idx = np.unravel_index(np.argmax(score), score.shape)

    return idx


def simulate_clicks_advanced(input_label, input_liver, fg=True, bg=True, center_offset=None, edge_offset=None,
                             pos_click_budget=10, neg_click_budget=10, use_gpu=True):
    if isinstance(input_label, np.ndarray):
        label_im = input_label.copy()
        label_im[label_im < 0] = 0
    else:
        raise ValueError("input_label must be numpy array")

    clicks = {'lesion': [], 'background': []}

    if fg and np.sum(label_im) > 0:
        connected_components = cc3d.connected_components(label_im, connectivity=26)
        unique_labels = np.unique(connected_components)[1:]
        size = min(pos_click_budget, len(unique_labels))
        sampled_labels = np.random.choice(unique_labels, size=size, replace=False)

        n_clicks = 0
        for label in sampled_labels:
            mask = connected_components == label
            point = sample_point_within_region(mask, edt_weight=np.random.uniform(0.4, 0.9))
            if center_offset is not None:
                point = perturb_click(center_offset, point, mask)
            clicks['lesion'].append(list(map(int, point)))
            n_clicks += 1

        # Fill remaining clicks with near-boundary points inside the object
        counter = 0
        while n_clicks < pos_click_budget:
            for label in sampled_labels:
                mask = connected_components == label
                edt = fastedt.edt(mask.astype(np.uint8))
                # get boundary by thresholding the EDT at a small value (low percentile)
                soft_boundary_mask = mask & (edt < 3)
                point = sample_point_within_region(soft_boundary_mask, edt_weight=np.random.uniform(0.2, 0.6))
                if edge_offset is not None:
                    point = perturb_click(edge_offset, point, mask)
                clicks['lesion'].append(list(map(int, point)))
                n_clicks += 1
                if n_clicks == pos_click_budget:
                    break
            counter += 1
            if counter > 10:
                print("Warning: Unable to sample enough background clicks. Please check your label image.")
                break

    if bg:
        assert isinstance(input_liver, np.ndarray), "input_pet must be numpy array"
        n_clicks = 0
        point = sample_point_within_region(input_liver, edt_weight=np.random.uniform(0.4, 0.9))
        if center_offset is not None:
            point = perturb_click(center_offset, point, input_liver)
        clicks['background'].append(list(map(int, point)))
        n_clicks += 1

        # Fill remaining clicks with near-boundary points inside the object
        counter = 0
        while n_clicks < neg_click_budget:

            edt = fastedt.edt(input_liver.astype(np.uint8))
            # get boundary by thresholding the EDT at a small value (low percentile)
            soft_boundary_mask = input_liver & (edt < 3)
            point = sample_point_within_region(soft_boundary_mask, edt_weight=np.random.uniform(0.2, 0.6))
            if edge_offset is not None:
                point = perturb_click(edge_offset, point, input_liver)
            clicks['background'].append(list(map(int, point)))
            n_clicks += 1
            if n_clicks == neg_click_budget:
                break
            counter += 1
            if counter > 10:
                print("Warning: Unable to sample enough background clicks. Please check your label image.")
                break

    return clicks


if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')