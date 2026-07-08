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
    ANATOMY_RANGES, parse_pengwin_clicks, Coord,
)
from pengwin_inference.postprocess import (
    keep_component_containing_point, merge_fragments, fill_anatomy_with_best_fragment,
    ANATOMY_ORDER,
)
from pengwin_inference.predict import FragmentPredictor, AnatomyPredictor, restore_labelmap_to_original
from pengwin_inference.routing import classify_image, ROUTING_ALLOWED_ANATOMIES

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
                            assembly: str = "overwrite") -> np.ndarray:
    """Per-fragment prediction -> keep clicked CC -> merge into the running instance label map,
    all in the model's GRID space. Merges incrementally (peak memory = a couple of grids, not a
    list of N masks -- important for the 16 GB budget).

    assembly:
      "overwrite" -> write each fragment's binary mask with its id; later fragments overwrite
                     earlier on overlap (ANATOMY_ORDER). Order decides contested voxels.
      "argmax"    -> probability competition: each contested voxel goes to the fragment with the
                     highest foreground probability (online argmax over `best_prob`/`best_label`).
      "ownership" -> keep each fragment's binary mask, but resolve OVERLAPS by nearest click:
                     a voxel claimed by >1 fragment goes to the one whose click is closest
                     (Voronoi tie-break). Fixes the duplicate/collapse cases without touching
                     non-contested voxels.
    """
    grid_shape = tuple(ct_data.shape[1:])
    all_clicks = [c for frags in per_anatomy_frags.values() for fr in frags for c in fr]
    merged = np.zeros(grid_shape, dtype=np.uint16)
    best_prob = np.zeros(grid_shape, dtype=np.float32) if assembly == "argmax" else None
    owner_dist = np.full(grid_shape, np.inf, dtype=np.float32) if assembly == "ownership" else None

    for aid in ANATOMY_ORDER:
        frags = per_anatomy_frags.get(aid, [])
        lo, hi = ANATOMY_RANGES[aid]
        for k, fg_list in enumerate(frags):
            fg_set = {tuple(c) for c in fg_list}
            bg = [c for c in all_clicks if tuple(c) not in fg_set]
            label = min(lo + k, hi)
            if assembly == "argmax":
                prob, grid_click = frag.predict_fragment(ct_data, properties, fg_list, bg, return_prob=True)
                cc = keep_component_containing_point(prob > 0.5, grid_click)  # restrict to clicked fragment
                fp = prob * cc                                               # >0.5 inside the fragment, else 0
                win = fp > best_prob
                merged[win] = label
                best_prob[win] = fp[win]
                del prob, cc, fp, win
            else:
                seg_grid, grid_click = frag.predict_fragment(ct_data, properties, fg_list, bg)
                m = keep_component_containing_point(seg_grid, grid_click)
                if assembly == "ownership":
                    coords = np.argwhere(m)
                    if coords.size == 0:
                        del seg_grid, m; continue
                    sl = tuple(slice(int(coords[:, i].min()), int(coords[:, i].max()) + 1) for i in range(3))
                    cz, cy, cx = grid_click
                    zs = (np.arange(sl[0].start, sl[0].stop) - cz)[:, None, None]
                    ys = (np.arange(sl[1].start, sl[1].stop) - cy)[None, :, None]
                    xs = (np.arange(sl[2].start, sl[2].stop) - cx)[None, None, :]
                    d = np.sqrt((zs ** 2 + ys ** 2 + xs ** 2).astype(np.float32))
                    take = (m[sl] > 0) & (d < owner_dist[sl])   # only claim where this click is nearer
                    merged[sl][take] = label
                    owner_dist[sl][take] = d[take]
                    del d, take
                else:  # overwrite
                    merged[m > 0] = label
                del seg_grid, m
    return merged


def _assemble(ct_path: str, per_anatomy_frags: Dict[int, List[List[Coord]]],
              output_path: Optional[str], frag: FragmentPredictor,
              anat: Optional[AnatomyPredictor], routing: bool, assembly: str = "overwrite"):
    ref_img = sitk.ReadImage(ct_path)
    if routing:
        case_type = classify_image(ref_img)
        allowed = ROUTING_ALLOWED_ANATOMIES[case_type]
        per_anatomy_frags = {a: v for a, v in per_anatomy_frags.items() if a in allowed}
        print(f"[routing] case classified as {case_type}; keeping anatomies {sorted(allowed)}")

    # fragment model: assemble the instance map in GRID space, then restore to original ONCE
    ct_data, properties = frag.preprocess_ct(ct_path)
    merged_grid = _predict_and_merge_grid(frag, ct_data, properties, per_anatomy_frags, assembly)
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
        per = parse_pengwin_clicks(json.load(f))
    per_frags = {aid: [[p] for p in pts] for aid, pts in per.items()}
    return _assemble(ct_path, per_frags, output_path, frag, anat, routing, assembly)


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
    return _assemble(ct_path, per_frags, output_path, frag, anat, routing, assembly)


def _build_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = os.environ.get("PENGWIN_CLICK_LAYOUT", "pair")
    tta = os.environ.get("PENGWIN_TTA", "0") == "1"
    # ROI (patch-around-click) is ON by default in the container: full-volume sliding window
    # per fragment does not fit the 10-min-per-case T4 budget. Set PENGWIN_ROI_MULT=0 to disable.
    roi_env = float(os.environ.get("PENGWIN_ROI_MULT", "1.5"))
    roi_mult = roi_env if roi_env > 0 else None
    ckpt = os.environ.get("PENGWIN_CHECKPOINT", "checkpoint_final.pth")
    frag = FragmentPredictor(
        os.environ.get("PENGWIN_FRAG_MODEL", "/opt/ml/model/fragment"),
        fold=int(os.environ.get("PENGWIN_FRAG_FOLD", "0")),
        click_layout=layout, device=device, use_mirroring=tta, roi_mult=roi_mult,
        checkpoint_name=ckpt)
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
