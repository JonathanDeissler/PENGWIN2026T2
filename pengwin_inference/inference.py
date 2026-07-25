"""
PENGWIN 2026 Task 2 (PENGWIN-Interact) inference orchestrator / Grand Challenge entrypoint.

Pipeline (per case):
  1. read CT (.mha) + peripelvic-fragment-clicks.json
  2. parse clicks -> {anatomy_id: [(z,y,x), ...]} (order == fragment order)
  3. [optional] route pelvic/femur and restrict anatomies
  4. [optional] anatomy model -> 5-class map (mask + hole-fill + single-fragment copy)
  5. per fragment: fragment model -> binary, keep CC overlapping the click, mask by anatomy
  6. merge -> instance labels in PENGWIN ranges (sacrum 1-50, L-hip 51-100, R-hip 101-150, femur 151-200)
  7. [optional] hole-fill anatomy regions
  8. write the integer label map to /output

Configuration via environment variables (with sensible defaults):
  PENGWIN_FRAG_MODEL    fragment model folder (nnU-Net results dir)  [/opt/ml/model/fragment]
  PENGWIN_FRAG_FOLD     fold (default 0)
  PENGWIN_CLICK_LAYOUT  "pair" (ResEnc points, ablation A/B) | "nninteractive" (ablation C)  [pair]
  PENGWIN_USE_ANATOMY   "1" to enable the anatomy model                      [0]
  PENGWIN_ANAT_MODEL    anatomy model folder                                 [/opt/ml/model/anatomy]
  PENGWIN_ROUTING       "on" to restrict anatomies via the spacing/FOV rule  [off]
  PENGWIN_TTA           "1" to enable test-time mirroring (slower)           [0]
"""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import SimpleITK as sitk
import torch

from nnunetv2.training.dataloading.pengwin_clicks import (
    ANATOMY_RANGES, parse_pengwin_clicks, strategy_index_from_json, Coord,
)
from pengwin_inference.postprocess import (
    keep_component_containing_point, merge_fragments, fill_anatomy_with_best_fragment,
    ANATOMY_ORDER,
)
from pengwin_inference.predict import FragmentPredictor, AnatomyPredictor, restore_labelmap_to_original
from pengwin_inference.routing import classify_image, ROUTING_ALLOWED_ANATOMIES

# "seeded" assembly knobs (env-overridable so the container can use them too):
#   SEED_THR    hysteresis low threshold to grow each fragment down to (vs the hard 0.5)
#   SEED_RADIUS radius (grid voxels) of the guaranteed-foreground ball pinned at the click
SEED_THR = float(os.environ.get("PENGWIN_SEED_THR", "0.3"))
SEED_RADIUS = int(os.environ.get("PENGWIN_SEED_RADIUS", "3"))


def _seed_ball(shape, click, r: int) -> np.ndarray:
    """A small binary ball around the click -- voxels we KNOW are the clicked fragment."""
    m = np.zeros(shape, dtype=bool)
    z, y, x = (int(v) for v in click)
    if not (0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]):
        return m
    zsl = slice(max(0, z - r), min(shape[0], z + r + 1))
    ysl = slice(max(0, y - r), min(shape[1], y + r + 1))
    xsl = slice(max(0, x - r), min(shape[2], x + r + 1))
    dz = (np.arange(zsl.start, zsl.stop) - z)[:, None, None]
    dy = (np.arange(ysl.start, ysl.stop) - y)[None, :, None]
    dx = (np.arange(xsl.start, xsl.stop) - x)[None, None, :]
    m[zsl, ysl, xsl] = (dz * dz + dy * dy + dx * dx) <= r * r
    return m

# Grand Challenge default mount points
GC_INPUT_CT_DIR = "/input/images/peripelvic-fracture-ct"
GC_INPUT_CLICKS = "/input/peripelvic-fragment-clicks.json"
GC_OUTPUT_DIR = "/output/images/peripelvic-fracture-ct-segmentation"


def _read_clicks(path: str) -> Dict[int, List[Coord]]:
    with open(path) as f:
        return parse_pengwin_clicks(json.load(f))


def merge_strategy_fragments(click_jsons: List[dict]) -> Dict[int, List[List[Coord]]]:
    """Combine several click JSONs (one per strategy) into per-fragment click lists.

    Each strategy lists points in the same fragment order per anatomy, so the k-th point of
    an anatomy refers to the same fragment across strategies. Returns
    ``{anatomy_id: [[clicks for fragment 0], [clicks for fragment 1], ...]}`` with up to one
    click per strategy per fragment.
    """
    parsed = [parse_pengwin_clicks(cj) for cj in click_jsons]
    out: Dict[int, List[List[Coord]]] = {}
    for aid in ANATOMY_RANGES:
        n = max((len(p.get(aid, [])) for p in parsed), default=0)
        if n == 0:
            continue
        out[aid] = [[p[aid][k] for p in parsed if aid in p and k < len(p[aid])] for k in range(n)]
    return out


def _predict_and_merge_grid(frag: FragmentPredictor, ct_data, properties,
                            per_anatomy_frags: Dict[int, List[List[Coord]]],
                            assembly: str = "overwrite", strategy_idx: int = 0) -> np.ndarray:
    """Per-fragment prediction -> keep clicked CC -> merge into the running instance label map,
    all in the model's GRID space. Merges incrementally (peak memory = a couple of grids, not a
    list of N masks -- important for the 16 GB budget).

    assembly:
      "overwrite" -> write each fragment's binary mask with its id; later fragments overwrite
                     earlier on overlap (ANATOMY_ORDER). Order decides contested voxels.
      "argmax"    -> probability competition: each contested voxel goes to the fragment with the
                     highest foreground probability (online argmax over `best_prob`/`best_label`).
      "ownership" -> keep each fragment's binary mask, but resolve OVERLAPS by nearest click
                     (straight-line Voronoi tie-break). Cheap, but the split is at the click
                     midplane -- often below IoU 0.5, so it tends not to help merged blobs.
      "watershed" -> click-seeded watershed of the union of all fragment masks: floods each
                     click's basin along the shape and splits at the thin fracture NECK
                     (geodesic tie-break). Best shot at splitting over-merged blobs correctly.
      "smaller"   -> overwrite, but the SMALLER fragment wins contested voxels (write largest
                     first), so a small fragment inside a big over-segmentation is preserved.
    """
    grid_shape = tuple(ct_data.shape[1:])
    all_clicks = [c for frags in per_anatomy_frags.values() for fr in frags for c in fr]

    def _iter_masks():
        """Yield (label, binary_mask_grid, grid_click) for every fragment (binary modes)."""
        for aid in ANATOMY_ORDER:
            lo, hi = ANATOMY_RANGES[aid]
            for k, fg_list in enumerate(per_anatomy_frags.get(aid, [])):
                fg_set = {tuple(c) for c in fg_list}
                bg = [c for c in all_clicks if tuple(c) not in fg_set]
                seg_grid, grid_click = frag.predict_fragment(ct_data, properties, fg_list, bg,
                                                             strategy_idx=strategy_idx)
                yield min(lo + k, hi), keep_component_containing_point(seg_grid, grid_click), grid_click

    if assembly == "argmax":
        merged = np.zeros(grid_shape, dtype=np.uint16)
        best_prob = np.zeros(grid_shape, dtype=np.float32)
        for aid in ANATOMY_ORDER:
            lo, hi = ANATOMY_RANGES[aid]
            for k, fg_list in enumerate(per_anatomy_frags.get(aid, [])):
                fg_set = {tuple(c) for c in fg_list}
                bg = [c for c in all_clicks if tuple(c) not in fg_set]
                prob, grid_click = frag.predict_fragment(ct_data, properties, fg_list, bg,
                                                         return_prob=True, strategy_idx=strategy_idx)
                cc = keep_component_containing_point(prob > 0.5, grid_click)
                fp = prob * cc
                win = fp > best_prob
                merged[win] = min(lo + k, hi)
                best_prob[win] = fp[win]
        return merged

    if assembly == "seeded":
        # Like argmax, but each fragment is grown by HYSTERESIS from a guaranteed-FG seed:
        # keep the (prob > SEED_THR) component that owns the click -- lower than 0.5 so weak/
        # cold fragments are recovered -- union a small pinned ball at the click, then let the
        # fragments compete for contested voxels by probability. The pinned seed can never be
        # stolen by a later fragment (we KNOW its label), so no clicked fragment is ever lost.
        merged = np.zeros(grid_shape, dtype=np.uint16)
        best_prob = np.zeros(grid_shape, dtype=np.float32)
        for aid in ANATOMY_ORDER:
            lo, hi = ANATOMY_RANGES[aid]
            for k, fg_list in enumerate(per_anatomy_frags.get(aid, [])):
                fg_set = {tuple(c) for c in fg_list}
                bg = [c for c in all_clicks if tuple(c) not in fg_set]
                prob, grid_click = frag.predict_fragment(ct_data, properties, fg_list, bg,
                                                         return_prob=True, strategy_idx=strategy_idx)
                label = min(lo + k, hi)
                cc = keep_component_containing_point(prob > SEED_THR, grid_click).astype(bool)
                seed = _seed_ball(grid_shape, grid_click, SEED_RADIUS)
                cc |= seed
                fp = prob * cc
                win = cc & (fp > best_prob)
                merged[win] = label
                best_prob[win] = fp[win]
                merged[seed] = label          # pin: guaranteed-FG voxels are final
                best_prob[seed] = np.inf
        return merged

    if assembly == "watershed":
        from scipy.ndimage import distance_transform_edt
        from skimage.segmentation import watershed
        union = np.zeros(grid_shape, dtype=bool)
        markers = np.zeros(grid_shape, dtype=np.int32)
        labels_list = []
        for label, m, click in _iter_masks():
            if m.sum() == 0:
                continue
            union |= m > 0
            labels_list.append(label)
            z, y, x = (int(v) for v in click)
            if 0 <= z < grid_shape[0] and 0 <= y < grid_shape[1] and 0 <= x < grid_shape[2]:
                markers[z, y, x] = len(labels_list)
        merged = np.zeros(grid_shape, dtype=np.uint16)
        if union.any():
            ws = watershed(-distance_transform_edt(union), markers, mask=union)
            for i, label in enumerate(labels_list, start=1):
                merged[ws == i] = label
        return merged

    if assembly == "smaller":
        items = [(int(m.sum()), label, m > 0) for label, m, _ in _iter_masks() if m.sum() > 0]
        items.sort(key=lambda t: -t[0])   # largest first -> smaller written last -> smaller wins overlaps
        merged = np.zeros(grid_shape, dtype=np.uint16)
        for _, label, mask in items:
            merged[mask] = label
        return merged

    # overwrite / ownership
    merged = np.zeros(grid_shape, dtype=np.uint16)
    owner_dist = np.full(grid_shape, np.inf, dtype=np.float32) if assembly == "ownership" else None
    for label, m, grid_click in _iter_masks():
        if assembly == "ownership":
            coords = np.argwhere(m)
            if coords.size == 0:
                continue
            sl = tuple(slice(int(coords[:, i].min()), int(coords[:, i].max()) + 1) for i in range(3))
            cz, cy, cx = grid_click
            zs = (np.arange(sl[0].start, sl[0].stop) - cz)[:, None, None]
            ys = (np.arange(sl[1].start, sl[1].stop) - cy)[None, :, None]
            xs = (np.arange(sl[2].start, sl[2].stop) - cx)[None, None, :]
            d = np.sqrt((zs ** 2 + ys ** 2 + xs ** 2).astype(np.float32))
            take = (m[sl] > 0) & (d < owner_dist[sl])
            merged[sl][take] = label
            owner_dist[sl][take] = d[take]
        else:  # overwrite
            merged[m > 0] = label
    return merged


def _assemble(ct_path: str, per_anatomy_frags: Dict[int, List[List[Coord]]],
              output_path: Optional[str], frag: FragmentPredictor,
              anat: Optional[AnatomyPredictor], routing: bool, assembly: str = "overwrite",
              strategy_idx: int = 0):
    ref_img = sitk.ReadImage(ct_path)
    if routing:
        case_type = classify_image(ref_img)
        allowed = ROUTING_ALLOWED_ANATOMIES[case_type]
        per_anatomy_frags = {a: v for a, v in per_anatomy_frags.items() if a in allowed}
        print(f"[routing] case classified as {case_type}; keeping anatomies {sorted(allowed)}")

    # fragment model: assemble the instance map in GRID space, then restore to original ONCE
    ct_data, properties = frag.preprocess_ct(ct_path)
    merged_grid = _predict_and_merge_grid(frag, ct_data, properties, per_anatomy_frags, assembly, strategy_idx)
    del ct_data
    merged = restore_labelmap_to_original(
        merged_grid, frag.predictor.plans_manager, frag.predictor.configuration_manager, properties)

    # optional anatomy model (its own grid): mask each anatomy's labels to its region + hole-fill
    if anat is not None:
        flat = {a: [c for fr in frags for c in fr] for a, frags in per_anatomy_frags.items()}
        anat_map, _ = anat.predict(ct_path, flat)
        for aid, (lo, hi) in ANATOMY_RANGES.items():
            in_range = (merged >= lo) & (merged <= hi)
            merged[in_range & (anat_map != aid)] = 0
        merged = fill_anatomy_with_best_fragment(anat_map, merged)

    if output_path is not None:
        out = sitk.GetImageFromArray(merged.astype(np.uint8))
        out.CopyInformation(ref_img)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(out, output_path, True)
        print(f"[done] wrote {output_path} (labels {sorted(np.unique(merged).tolist())})")
    return merged, ref_img


def run_case(ct_path: str, clicks_path: str, output_path: Optional[str],
             frag: FragmentPredictor, anat: Optional[AnatomyPredictor] = None,
             routing: bool = False, assembly: str = "overwrite"):
    """Submission pipeline for one case with a SINGLE clicks JSON (one click per fragment)."""
    with open(clicks_path) as f:
        cj = json.load(f)
    per = parse_pengwin_clicks(cj)
    strategy_idx = strategy_index_from_json(cj)  # only used by the "strategy" click layout
    per_frags = {aid: [[p] for p in pts] for aid, pts in per.items()}
    return _assemble(ct_path, per_frags, output_path, frag, anat, routing, assembly, strategy_idx)


def run_case_combined(ct_path: str, clicks_paths: List[str], output_path: Optional[str],
                      frag: FragmentPredictor, anat: Optional[AnatomyPredictor] = None,
                      routing: bool = False, assembly: str = "overwrite"):
    """Submission pipeline using SEVERAL clicks JSONs (e.g. all 4 strategies), giving each
    fragment multiple positive clicks."""
    jsons = []
    for p in clicks_paths:
        with open(p) as f:
            jsons.append(json.load(f))
    per_frags = merge_strategy_fragments(jsons)
    # combined mixes strategies; the "strategy" layout would need per-click routing, so just
    # use the first strategy here (combined is intended for the pair/nninteractive layouts).
    strategy_idx = strategy_index_from_json(jsons[0]) if jsons else 0
    return _assemble(ct_path, per_frags, output_path, frag, anat, routing, assembly, strategy_idx)


def _build_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = os.environ.get("PENGWIN_CLICK_LAYOUT", "pair")
    tta = os.environ.get("PENGWIN_TTA", "0") == "1"
    # ROI (patch-around-click) is ON by default in the container: full-volume sliding window
    # per fragment does not fit the 10-min-per-case T4 budget. Set PENGWIN_ROI_MULT=0 to disable.
    roi_env = float(os.environ.get("PENGWIN_ROI_MULT", "1.5"))
    roi_mult = roi_env if roi_env > 0 else None
    ckpt = os.environ.get("PENGWIN_CHECKPOINT", "checkpoint_final.pth")
    # Iterative refinement: extra passes feeding the previous prediction into the initSeg
    # channel. nnInteractive layout only; tightens boundaries and splits over-merged fragments.
    refine = int(os.environ.get("PENGWIN_REFINE", "0"))
    frag = FragmentPredictor(
        os.environ.get("PENGWIN_FRAG_MODEL", "/opt/ml/model/fragment"),
        fold=int(os.environ.get("PENGWIN_FRAG_FOLD", "0")),
        click_layout=layout, device=device, use_mirroring=tta, roi_mult=roi_mult,
        checkpoint_name=ckpt, refine_iters=refine)
    anat = None
    if os.environ.get("PENGWIN_USE_ANATOMY", "0") == "1":
        anat = AnatomyPredictor(
            os.environ.get("PENGWIN_ANAT_MODEL", "/opt/ml/model/anatomy"),
            fold=int(os.environ.get("PENGWIN_ANAT_FOLD", "0")), device=device,
            use_mirroring=tta, checkpoint_name=ckpt)
    return frag, anat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", default=None, help="CT .mha (local mode; default: Grand Challenge input dir)")
    ap.add_argument("--clicks", default=None, help="clicks JSON (local mode)")
    ap.add_argument("--output", default=None, help="output .mha (local mode)")
    args = ap.parse_args()

    routing = os.environ.get("PENGWIN_ROUTING", "off") == "on"
    assembly = os.environ.get("PENGWIN_ASSEMBLY", "overwrite")
    frag, anat = _build_models()

    if args.ct is not None:  # local single-case mode
        out = args.output or "prediction.mha"
        run_case(args.ct, args.clicks, out, frag, anat, routing, assembly)
        return

    # Grand Challenge mode
    ct_files = sorted(glob(os.path.join(GC_INPUT_CT_DIR, "*.mha")))
    assert len(ct_files) == 1, f"expected exactly one CT under {GC_INPUT_CT_DIR}, found {len(ct_files)}"

    # locate the clicks JSON: the documented path, else any *.json under /input except inputs.json
    clicks = GC_INPUT_CLICKS
    if not os.path.isfile(clicks):
        cands = [p for p in glob("/input/**/*.json", recursive=True)
                 if os.path.basename(p) != "inputs.json"]
        assert cands, "no clicks JSON found under /input"
        clicks = cands[0]
    print(f"[input] CT={ct_files[0]}  clicks={clicks}")

    out_path = os.path.join(GC_OUTPUT_DIR, os.path.basename(ct_files[0]))
    run_case(ct_files[0], clicks, out_path, frag, anat, routing, assembly)


if __name__ == "__main__":
    main()
