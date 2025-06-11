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
        x = np.round(x * factor[2]).astype(np.uint16)
        y = np.round(y * factor[1]).astype(np.uint16)
        z = np.round(z * factor[0]).astype(np.uint16)
        return [z, y, x]


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
        pos_clicks = gaussian_filter(pos_clicks, sigma=sigma)
        neg_clicks = gaussian_filter(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks


def generated_sparse_to_dense_point_gauss(clicks: dict, shape: tuple[int, ...], sigma: float = 1.0) -> np.ndarray:
    pos_clicks, neg_clicks = np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    if len(clicks["background"]) > 0:
        for clck in clicks["tumor"]:
            pos_clicks[*clck] = 1.0
        for clck in clicks["background"]:
            neg_clicks[*clck] = 1.0
        pos_clicks = gaussian_filter(pos_clicks, sigma=sigma)
        neg_clicks = gaussian_filter(neg_clicks, sigma=sigma)
    return pos_clicks, neg_clicks



if __name__ == '__main__':
    unpack_dataset('/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/2d')