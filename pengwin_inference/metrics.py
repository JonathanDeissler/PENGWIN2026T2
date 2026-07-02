"""
PENGWIN-style fragment instance metrics for offline (per-fold) evaluation.

Scoring follows the challenge description: within each anatomical region, each GT fragment is
matched to the predicted fragment with the highest IoU; small components (~1 cm3) are removed
first. We report the headline metrics -- Fracture Dice, Instance Precision/Recall/F1, HD95,
ASSD. (Local Dice / Topology Consistency / Merge-Split counts can be layered on later or taken
from the official Grand Challenge evaluator; these cover the primary ranking signals.)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import cc3d
from scipy.ndimage import distance_transform_edt

from nnunetv2.training.dataloading.pengwin_clicks import ANATOMY_RANGES


def remove_small_components(labelmap: np.ndarray, spacing_xyz, min_cm3: float = 1.0) -> np.ndarray:
    """Drop connected components smaller than ~min_cm3 (spacing is SimpleITK (x,y,z) mm)."""
    voxel_vol_mm3 = float(np.prod(spacing_xyz))
    min_voxels = max(1, int(round(min_cm3 * 1000.0 / voxel_vol_mm3)))
    out = labelmap.copy()
    for lbl in np.unique(labelmap):
        if lbl == 0:
            continue
        m = labelmap == lbl
        if int(m.sum()) < min_voxels:
            out[m] = 0
    return out


def _instances_in_range(labelmap: np.ndarray, lo: int, hi: int) -> Dict[int, np.ndarray]:
    return {int(l): (labelmap == l)
            for l in np.unique(labelmap) if lo <= int(l) <= hi}


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * float((a & b).sum()) / float(s)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return 0.0 if u == 0 else float((a & b).sum()) / float(u)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing_zyx) -> np.ndarray:
    """Symmetric surface distances (mm) between the boundaries of masks a and b."""
    if not a.any() or not b.any():
        return np.array([np.inf])
    # boundary voxels = foreground minus its erosion (approximate via EDT==0 on the mask edge)
    def border(m):
        inside = distance_transform_edt(m, sampling=spacing_zyx)
        return m & (inside <= min(spacing_zyx))
    dt_b = distance_transform_edt(~b, sampling=spacing_zyx)
    dt_a = distance_transform_edt(~a, sampling=spacing_zyx)
    da = dt_b[border(a)]
    db = dt_a[border(b)]
    return np.concatenate([da, db]) if da.size and db.size else np.array([np.inf])


def score_case(gt: np.ndarray, pred: np.ndarray, spacing_xyz, iou_thr: float = 0.5,
               min_cm3: float = 1.0) -> Dict:
    """Score one case. spacing is SimpleITK (x,y,z); arrays are numpy (z,y,x)."""
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    gt = remove_small_components(gt, spacing_xyz, min_cm3)
    pred = remove_small_components(pred, spacing_xyz, min_cm3)

    dices, hd95s, assds = [], [], []
    tp = fp = fn = 0
    per_anatomy = {}

    for aid, (lo, hi) in ANATOMY_RANGES.items():
        gts = _instances_in_range(gt, lo, hi)
        prs = _instances_in_range(pred, lo, hi)
        if not gts and not prs:
            continue
        # greedy highest-IoU matching within this anatomy
        pairs = sorted(((_iou(gm, pm), gl, pl) for gl, gm in gts.items() for pl, pm in prs.items()),
                       reverse=True)
        used_g, used_p, matched = set(), set(), []
        for iou, gl, pl in pairs:
            if gl in used_g or pl in used_p or iou < iou_thr:
                continue
            used_g.add(gl); used_p.add(pl); matched.append((gl, pl, iou))
        a_tp = len(matched)
        a_fn = len(gts) - a_tp
        a_fp = len(prs) - a_tp
        tp += a_tp; fn += a_fn; fp += a_fp
        for gl, pl, _ in matched:
            dices.append(_dice(gts[gl], prs[pl]))
            sd = _surface_distances(gts[gl], prs[pl], spacing_zyx)
            hd95s.append(float(np.percentile(sd, 95)))
            assds.append(float(sd.mean()))
        per_anatomy[aid] = {"tp": a_tp, "fp": a_fp, "fn": a_fn,
                            "n_gt": len(gts), "n_pred": len(prs)}

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    finite = lambda xs: [x for x in xs if np.isfinite(x)]
    return {
        "fracture_dice": float(np.mean(dices)) if dices else 0.0,
        "hd95": float(np.mean(finite(hd95s))) if finite(hd95s) else float("nan"),
        "assd": float(np.mean(finite(assds))) if finite(assds) else float("nan"),
        "instance_precision": prec, "instance_recall": rec, "instance_f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "per_anatomy": per_anatomy,
        "out_of_range_pred_labels": sorted(
            int(l) for l in np.unique(pred)
            if l > 0 and not any(lo <= l <= hi for lo, hi in ANATOMY_RANGES.values())),
    }


def aggregate(cases: List[Dict]) -> Dict:
    keys = ["fracture_dice", "hd95", "assd", "instance_precision", "instance_recall", "instance_f1"]
    out = {}
    for k in keys:
        vals = [c[k] for c in cases if k in c and np.isfinite(c[k])]
        out["mean_" + k] = float(np.mean(vals)) if vals else float("nan")
    out["n_cases"] = len(cases)
    out["total_tp"] = int(sum(c["tp"] for c in cases))
    out["total_fp"] = int(sum(c["fp"] for c in cases))
    out["total_fn"] = int(sum(c["fn"] for c in cases))
    return out
