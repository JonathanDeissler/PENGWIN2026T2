"""
PENGWIN-style fragment instance metrics for offline (per-fold) evaluation.

Scoring follows the challenge description: within each anatomical region, each GT fragment is
matched to the predicted fragment with the highest IoU; small components (~1 cm3) are removed
first. We report the headline metrics -- Fracture Dice, Instance Precision/Recall/F1, HD95,
ASSD. (Local Dice / Topology Consistency / Merge-Split counts can be layered on later or taken
from the official Grand Challenge evaluator; these cover the primary ranking signals.)

Performance: everything operates on per-label bounding-box crops (via ``find_objects``) rather
than full-volume arrays -- fragments are tiny relative to the volume, and full-volume
distance transforms / boolean ops per pair were the dominant cost.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.ndimage import find_objects, distance_transform_edt, binary_erosion

from nnunetv2.training.dataloading.pengwin_clicks import ANATOMY_RANGES


def remove_small_components(labelmap: np.ndarray, spacing_xyz, min_cm3: float = 1.0) -> np.ndarray:
    """Drop labels smaller than ~min_cm3 in ONE pass via bincount (spacing is SimpleITK
    (x,y,z) mm). Each PENGWIN label is a single fragment, so per-label ~= per-component."""
    voxel_vol_mm3 = float(np.prod(spacing_xyz))
    min_voxels = max(1, int(round(min_cm3 * 1000.0 / voxel_vol_mm3)))
    counts = np.bincount(labelmap.reshape(-1))
    small = np.nonzero(counts < min_voxels)[0]
    small = small[small > 0]
    if small.size == 0:
        return labelmap
    out = labelmap.copy()
    out[np.isin(labelmap, small)] = 0
    return out


def _union_slices(s1, s2, shape, margin=2):
    out = []
    for a, b, dim in zip(s1, s2, shape):
        lo = max(0, min(a.start, b.start) - margin)
        hi = min(dim, max(a.stop, b.stop) + margin)
        out.append(slice(lo, hi))
    return tuple(out)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing_zyx) -> np.ndarray:
    """Symmetric surface distances (mm) between boundaries of two small cropped masks."""
    if not a.any() or not b.any():
        return np.array([np.inf])
    sa = a ^ binary_erosion(a)
    sb = b ^ binary_erosion(b)
    dt_b = distance_transform_edt(~b, sampling=spacing_zyx)
    dt_a = distance_transform_edt(~a, sampling=spacing_zyx)
    return np.concatenate([dt_b[sa], dt_a[sb]])


def score_case(gt: np.ndarray, pred: np.ndarray, spacing_xyz, iou_thr: float = 0.5,
               min_cm3: float = 1.0) -> Dict:
    """Score one case. spacing is SimpleITK (x,y,z); arrays are numpy (z,y,x)."""
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    gt = remove_small_components(gt, spacing_xyz, min_cm3)
    pred = remove_small_components(pred, spacing_xyz, min_cm3)

    # per-label bounding boxes (index label-1 -> slices, or None if absent), computed once
    gt_obj = find_objects(gt.astype(np.int32))
    pr_obj = find_objects(pred.astype(np.int32))

    def obj(objs, lbl):
        return objs[lbl - 1] if lbl - 1 < len(objs) else None

    dices, hd95s, assds = [], [], []
    tp = fp = fn = 0
    per_anatomy = {}

    for aid, (lo, hi) in ANATOMY_RANGES.items():
        gt_labels = [l for l in range(lo, hi + 1) if obj(gt_obj, l) is not None]
        pred_labels = [l for l in range(lo, hi + 1) if obj(pr_obj, l) is not None]
        if not gt_labels and not pred_labels:
            continue

        # candidate (gt, pred) overlaps: only pred labels present inside the gt label's bbox
        pairs = []
        for gl in gt_labels:
            gsl = obj(gt_obj, gl)
            g_sub = gt[gsl] == gl
            cand = np.unique(pred[gsl][g_sub])
            for pl in cand:
                if pl == 0 or not (lo <= pl <= hi):
                    continue
                u = _union_slices(gsl, obj(pr_obj, pl), gt.shape)
                a = gt[u] == gl
                b = pred[u] == pl
                inter = int(np.logical_and(a, b).sum())
                union = int(np.logical_or(a, b).sum())
                if union:
                    pairs.append((inter / union, gl, pl))

        pairs.sort(reverse=True)
        used_g, used_p, matched = set(), set(), []
        for iou, gl, pl in pairs:
            if gl in used_g or pl in used_p or iou < iou_thr:
                continue
            used_g.add(gl); used_p.add(pl); matched.append((gl, pl))

        tp += len(matched); fn += len(gt_labels) - len(matched); fp += len(pred_labels) - len(matched)
        for gl, pl in matched:
            u = _union_slices(obj(gt_obj, gl), obj(pr_obj, pl), gt.shape)
            a = gt[u] == gl
            b = pred[u] == pl
            inter = int(np.logical_and(a, b).sum())
            dices.append(2.0 * inter / (int(a.sum()) + int(b.sum())))
            sd = _surface_distances(a, b, spacing_zyx)
            hd95s.append(float(np.percentile(sd, 95)))
            assds.append(float(sd.mean()))
        per_anatomy[aid] = {"tp": len(matched), "fp": len(pred_labels) - len(matched),
                            "fn": len(gt_labels) - len(matched),
                            "n_gt": len(gt_labels), "n_pred": len(pred_labels)}

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
