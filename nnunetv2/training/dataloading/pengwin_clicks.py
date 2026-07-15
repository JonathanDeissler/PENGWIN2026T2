"""
PENGWIN 2026 Task 2 (PENGWIN-Interact) click utilities.

This module centralises everything related to the *fracture-fragment* clicks used by
the interactive segmentation pipeline:

* parsing the challenge-provided ``peripelvic-fragment-clicks.json`` files,
* mapping between PENGWIN instance labels (0-200) and the four anatomical structures
  (sacrum / left hip / right hip / femur) and their label ranges,
* simulating clicks on the fly with the same four strategies the challenge uses
  (uniformly sampled, euclidean distance transform, center of mass, boundary internal
  margin) -- implemented on the CPU so we do not require ``cupy``/``cucim``.

Coordinate convention
---------------------
PENGWIN points are stored in ``(z, y, x)`` numpy/array order (they come from
``SimpleITK.GetArrayFromImage``). We keep everything in that order internally. The only
place a different order is needed is :func:`nnunetv2.training.dataloading.utils.preprocess_point`,
which expects ``[x, y, z]`` -- use :func:`zyx_to_xyz` right before calling it.

The label scheme (packs anatomy + instance), matching the baseline:
    0          background
    1   - 50   sacrum fragments
    51  - 100  left hip fragments
    101 - 150  right hip fragments
    151 - 200  femur fragments
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import cc3d
import edt as fastedt
from scipy import ndimage


# --------------------------------------------------------------------------------------
# Anatomy <-> label-range bookkeeping
# --------------------------------------------------------------------------------------
ANATOMY_RANGES: Dict[int, Tuple[int, int]] = {
    1: (1, 50),     # sacrum
    2: (51, 100),   # left hip
    3: (101, 150),  # right hip
    4: (151, 200),  # femur
}
ANATOMY_NAMES: Dict[int, str] = {1: "sacrum", 2: "left_hip", 3: "right_hip", 4: "femur"}
# The JSON point "name" begins with one of these keywords (longest-match first so that
# "Left Hip"/"Right Hip" are matched before any single word).
KEYWORD_MAP: Dict[str, int] = {"Sacrum": 1, "Left Hip": 2, "Right Hip": 3, "Femur": 4}

STRATEGIES: Tuple[str, ...] = (
    "uniformly_sampled",
    "euclidean_distance_transform",
    "center_of_mass",
    "boundary_internal_margin",
)

Coord = Tuple[int, int, int]  # (z, y, x)


def label_to_anatomy_id(label: int) -> Optional[int]:
    """Map a PENGWIN instance label (1-200) to its anatomy id (1-4). 0 -> None."""
    if label <= 0:
        return None
    for aid, (lo, hi) in ANATOMY_RANGES.items():
        if lo <= label <= hi:
            return aid
    return None


def anatomy_name_from_keyword(name: str) -> Optional[int]:
    """Return the anatomy id (1-4) for a JSON point name, or None if no keyword matches."""
    for keyword, aid in KEYWORD_MAP.items():
        if keyword in name:
            return aid
    return None


def zyx_to_xyz(point_zyx: Sequence[int]) -> List[int]:
    """Reorder a ``(z, y, x)`` point to ``[x, y, z]`` (for ``preprocess_point``)."""
    z, y, x = point_zyx
    return [x, y, z]


# --------------------------------------------------------------------------------------
# Parsing the challenge-provided clicks JSON
# --------------------------------------------------------------------------------------
def parse_pengwin_clicks(click_json: dict) -> Dict[int, List[Coord]]:
    """
    Parse a ``peripelvic-fragment-clicks.json`` dict into per-anatomy click lists.

    Returns ``{anatomy_id: [(z, y, x), ...]}`` (anatomy ids 1-4, only those present).
    The order of points within an anatomy is preserved -- it corresponds to the order of
    the fragments and therefore to the order in which fragment label ids are assigned.
    """
    out: Dict[int, List[Coord]] = {}
    for p in click_json.get("points", []):
        aid = anatomy_name_from_keyword(p["name"])
        if aid is None:
            continue
        z, y, x = p["point"]
        out.setdefault(aid, []).append((int(z), int(y), int(x)))
    return out


# --------------------------------------------------------------------------------------
# On-the-fly click simulation (CPU, no cupy) -- one click per fragment
# --------------------------------------------------------------------------------------
def _uniform_click(mask: np.ndarray) -> Coord:
    idx = np.argwhere(mask)
    return tuple(int(v) for v in idx[np.random.randint(idx.shape[0])])


def _edt_max_click(edt: np.ndarray) -> Coord:
    return tuple(int(v) for v in np.unravel_index(int(np.argmax(edt)), edt.shape))


def _center_of_mass_click(mask: np.ndarray, edt: np.ndarray) -> Coord:
    com = ndimage.center_of_mass(mask)
    com = tuple(int(round(c)) for c in com)
    # Fall back to the EDT maximum if the geometric centroid falls outside the mask
    # (concave fragments), exactly as the challenge simulator does.
    if not mask[com]:
        return _edt_max_click(edt)
    return com


def _boundary_internal_margin_click(mask: np.ndarray, edt: np.ndarray) -> Coord:
    edt_inverted = (edt.max() - edt) * (edt > 0)
    ridge = (edt_inverted == edt_inverted.max()) & (mask > 0)
    idx = np.argwhere(ridge)
    if idx.shape[0] == 0:  # degenerate (e.g. single-voxel fragment)
        return _uniform_click(mask)
    return tuple(int(v) for v in idx[np.random.randint(idx.shape[0])])


def simulate_click_in_fragment(frag_mask: np.ndarray, strategy: Optional[str] = None) -> Coord:
    """
    Simulate a single click ``(z, y, x)`` inside a binary fragment mask, mirroring the four
    challenge strategies. ``strategy=None`` picks one at random. The EDT-based strategies run on
    the fragment's bounding box (not the whole patch) for speed.
    """
    if strategy is None:
        strategy = STRATEGIES[np.random.randint(len(STRATEGIES))]

    coords = np.argwhere(frag_mask)
    if coords.shape[0] == 0:
        return (0, 0, 0)
    if strategy == "uniformly_sampled":
        return tuple(int(v) for v in coords[np.random.randint(coords.shape[0])])

    # crop to the fragment bbox so the EDT is over a small array, then map the click back.
    # Pad with a 1-voxel background border so surface voxels see background exactly as they do
    # in the full volume -- this keeps the EDT (and the boundary_internal_margin shell) identical
    # to computing it on the whole patch. Map back with the -1 pad offset.
    mn = coords.min(0); mx = coords.max(0) + 1
    sub = frag_mask[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    sub = np.pad(sub, 1)
    edt = fastedt.edt(sub.astype(np.uint8))
    if strategy == "euclidean_distance_transform":
        loc = _edt_max_click(edt)
    elif strategy == "center_of_mass":
        loc = _center_of_mass_click(sub, edt)
    elif strategy == "boundary_internal_margin":
        loc = _boundary_internal_margin_click(sub, edt)
    else:
        raise ValueError(f"Unknown click strategy {strategy!r}")
    return tuple(int(loc[i]) - 1 + int(mn[i]) for i in range(3))


def sample_fragment_clicks(
    seg_instance: np.ndarray,
    strategy: Optional[str] = None,
    target_label: Optional[int] = None,
) -> Tuple[Optional[int], Optional[Coord], List[Coord]]:
    """
    Given an instance label patch (0 = background, >0 = distinct fragment ids), pick one target
    fragment and return ``(target_label, fg_click, [bg_clicks])``. ``fg_click`` uses ``strategy``;
    the negatives (one per other fragment) use a cheap uniform interior point -- a negative
    prompt does not need a strategic location, and this removes an EDT per other fragment.

    Uses ``find_objects`` for one-pass bounding boxes, so per-fragment work is confined to small
    crops. Returns ``(None, None, [])`` if the patch contains no fragment.
    """
    maxid = int(seg_instance.max())
    if maxid <= 0:
        return None, None, []
    seg32 = seg_instance if seg_instance.dtype == np.int32 else seg_instance.astype(np.int32)
    objs = ndimage.find_objects(seg32)
    labels = [l for l in range(1, maxid + 1) if l - 1 < len(objs) and objs[l - 1] is not None]
    if not labels:
        return None, None, []

    if strategy is None:
        strategy = STRATEGIES[np.random.randint(len(STRATEGIES))]
    if target_label is None or target_label not in labels:
        target_label = int(np.random.choice(labels))

    def click(lbl: int, strat: str) -> Coord:
        sl = objs[lbl - 1]
        loc = simulate_click_in_fragment(seg_instance[sl] == lbl, strat)
        return tuple(int(loc[i]) + int(sl[i].start) for i in range(3))

    fg_click = click(target_label, strategy)
    bg_clicks = [click(l, "uniformly_sampled") for l in labels if l != target_label]
    return target_label, fg_click, bg_clicks


# --------------------------------------------------------------------------------------
# Localised Gaussian rendering (fast + sparse; shared by conversion and the dataloader)
# --------------------------------------------------------------------------------------
def stamp_gaussian(vol: np.ndarray, center_zyx: Coord, sigma: float, trunc: float = 4.0) -> None:
    """Max-combine a unit-height Gaussian into ``vol`` in a local window around the click."""
    r = max(1, int(np.ceil(trunc * sigma)))
    sl, offs = [], []
    for c, s in zip(center_zyx, vol.shape):
        lo, hi = max(0, c - r), min(s, c + r + 1)
        if lo >= hi:
            return
        sl.append(slice(lo, hi))
        offs.append(lo - c)
    sl = tuple(sl)
    grids = np.ogrid[tuple(slice(o, o + (s.stop - s.start)) for o, s in zip(offs, sl))]
    d2 = sum(g * g for g in grids)
    g = np.exp(-d2 / (2.0 * sigma ** 2)).astype(np.float32)
    np.maximum(vol[sl], g, out=vol[sl])


def render_points_gauss(shape: Tuple[int, ...], points_zyx: Sequence[Coord], sigma: float) -> np.ndarray:
    """Render a set of clicks as a single ``float32`` heatmap (max-combined unit Gaussians)."""
    heat = np.zeros(shape, dtype=np.float32)
    for pt in points_zyx:
        stamp_gaussian(heat, tuple(int(v) for v in pt), sigma)
    return heat


def relabel_instances_within_anatomy(seg_instance: np.ndarray) -> np.ndarray:
    """
    Split each PENGWIN anatomy region into connected components and return a clean
    instance map (0 = bg, 1..N = fragments). Useful when the raw label only encodes the
    anatomy id rather than per-fragment ids, or to guarantee spatial connectivity before
    sampling clicks. Distinct anatomies never share an instance id.
    """
    out = np.zeros_like(seg_instance, dtype=np.int32)
    next_id = 0
    for aid in ANATOMY_RANGES:
        lo, hi = ANATOMY_RANGES[aid]
        anatomy_mask = (seg_instance >= lo) & (seg_instance <= hi)
        if not anatomy_mask.any():
            continue
        cc = cc3d.connected_components(anatomy_mask, connectivity=26)
        for comp_id in range(1, int(cc.max()) + 1):
            next_id += 1
            out[cc == comp_id] = next_id
    return out
