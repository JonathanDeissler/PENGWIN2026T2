"""
Deterministic assembly of per-fragment binary predictions into the final PENGWIN instance
label map (values 0-200). In-memory numpy ports of the baseline's inference scripts
(`keep_clicked_fragment.py`, `merge_all_fragment_predictions.py`,
`expand_fragments_to_anatomy.py`, `filter_left_right.py`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import cc3d

from nnunetv2.training.dataloading.pengwin_clicks import ANATOMY_RANGES, ANATOMY_NAMES

# Later anatomies overwrite earlier ones on overlap (matches baseline ordering).
ANATOMY_ORDER = [2, 3, 4, 1]  # left_hip, right_hip, femur, sacrum


def keep_component_overlapping_click(pred_mask: np.ndarray, fg_heatmap: np.ndarray) -> np.ndarray:
    """Keep only the connected component of ``pred_mask`` with the largest overlap with the
    positive-click heatmap (baseline `keep_component_by_heatmap`)."""
    binary = pred_mask > 0
    if not binary.any():
        return np.zeros_like(pred_mask, dtype=np.uint8)
    cc, n = cc3d.connected_components(binary, connectivity=26, return_N=True)
    best, best_score = 0, -1.0
    for comp in range(1, n + 1):
        score = float(fg_heatmap[cc == comp].sum())
        if score > best_score:
            best_score, best = score, comp
    return (cc == best).astype(np.uint8)


def keep_component_containing_point(pred_mask: np.ndarray, point_zyx) -> np.ndarray:
    """Keep the connected component that contains the click voxel. If the click is not inside
    any predicted foreground (model missed), fall back to the largest component."""
    binary = pred_mask > 0
    if not binary.any():
        return np.zeros_like(pred_mask, dtype=np.uint8)
    cc, n = cc3d.connected_components(binary, connectivity=26, return_N=True)
    z, y, x = (int(v) for v in point_zyx)
    cid = int(cc[z, y, x]) if (0 <= z < cc.shape[0] and 0 <= y < cc.shape[1] and 0 <= x < cc.shape[2]) else 0
    if cid == 0:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0
        cid = int(np.argmax(sizes))
    return (cc == cid).astype(np.uint8)


def merge_fragments(fragments: Dict[int, List[np.ndarray]], shape: Tuple[int, ...]) -> np.ndarray:
    """
    Merge per-anatomy lists of binary fragment masks into one instance map with PENGWIN label
    ranges. ``fragments`` maps anatomy_id -> [binary mask per fragment].
    """
    merged = np.zeros(shape, dtype=np.uint16)
    for aid in ANATOMY_ORDER:
        masks = fragments.get(aid, [])
        lo, hi = ANATOMY_RANGES[aid]
        for k, m in enumerate(masks):
            label = min(lo + k, hi)
            merged[m > 0] = label
    return merged


def fill_anatomy_with_best_fragment(anat: np.ndarray, frag: np.ndarray) -> np.ndarray:
    """For each anatomy region, fill its *background* fragment voxels with the most frequent
    fragment label in that region (baseline `expand_fragments_to_anatomy`)."""
    out = frag.copy()
    for aid in np.unique(anat):
        if aid == 0:
            continue
        region = anat == aid
        labels_here = frag[region]
        labels_here = labels_here[labels_here > 0]
        if labels_here.size == 0:
            continue
        vals, counts = np.unique(labels_here, return_counts=True)
        best = vals[int(np.argmax(counts))]
        out[region & (frag == 0)] = best
    return out


def filter_left_right(anat: np.ndarray) -> np.ndarray:
    """Reassign hip components to left(2)/right(3) by their centroid relative to the sacrum
    centroid (baseline `filter_left_right`). Sacrum(1)/femur(4) untouched. Assumes the
    array is in (z, y, x) and that +x is the patient's left; flip if needed downstream."""
    out = anat.copy()
    sacrum = anat == 1
    if not sacrum.any():
        return out
    sacrum_cx = np.argwhere(sacrum)[:, 2].mean()
    hips = (anat == 2) | (anat == 3)
    if not hips.any():
        return out
    cc, n = cc3d.connected_components(hips, connectivity=26, return_N=True)
    for comp in range(1, n + 1):
        m = cc == comp
        cx = np.argwhere(m)[:, 2].mean()
        out[m] = 2 if cx >= sacrum_cx else 3
    return out
