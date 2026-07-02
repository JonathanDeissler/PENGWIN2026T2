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
)
from pengwin_inference.predict import FragmentPredictor, AnatomyPredictor
from pengwin_inference.routing import classify_image, ROUTING_ALLOWED_ANATOMIES

# Grand Challenge default mount points
GC_INPUT_CT_DIR = "/input/images/peripelvic-fracture-ct"
GC_INPUT_CLICKS = "/input/peripelvic-fragment-clicks.json"
GC_OUTPUT_DIR = "/output/images/peripelvic-fracture-ct-segmentation"


def _read_clicks(path: str) -> Dict[int, List[Coord]]:
    with open(path) as f:
        return parse_pengwin_clicks(json.load(f))


def run_case(ct_path: str, clicks_path: str, output_path: Optional[str],
             frag: FragmentPredictor, anat: Optional[AnatomyPredictor] = None,
             routing: bool = False):
    """Run the full submission pipeline for one case. Writes the label map to
    ``output_path`` if given, and returns ``(merged_labelmap, reference_image)``."""
    ref_img = sitk.ReadImage(ct_path)
    per_anatomy = _read_clicks(clicks_path)

    allowed = set(ANATOMY_RANGES)
    if routing:
        case_type = classify_image(ref_img)
        allowed = ROUTING_ALLOWED_ANATOMIES[case_type]
        per_anatomy = {a: p for a, p in per_anatomy.items() if a in allowed}
        print(f"[routing] case classified as {case_type}; keeping anatomies {sorted(allowed)}")

    # Optional anatomy model (for masking / hole-fill / single-fragment copy)
    anat_map = None
    if anat is not None:
        anat_map, _ = anat.predict(ct_path, per_anatomy)

    # Preprocess CT once for the fragment model
    ct_data, properties = frag.preprocess_ct(ct_path)

    fragments: Dict[int, List[np.ndarray]] = {}
    for aid, points in per_anatomy.items():
        masks: List[np.ndarray] = []
        if len(points) <= 1 and anat_map is not None:
            # single-fragment anatomy: copy the whole anatomy region from Phase 1
            masks.append((anat_map == aid).astype(np.uint8))
        else:
            for fg_click in points:
                bg_clicks = [p for _a, pts in per_anatomy.items() for p in pts if p != fg_click]
                pred, _ = frag.predict_fragment(ct_data, properties, fg_click, bg_clicks)
                # keep only the connected component the positive click landed in (original space)
                pred = keep_component_containing_point(pred, fg_click)
                if anat_map is not None:
                    pred = pred * (anat_map == aid)
                masks.append(pred.astype(np.uint8))
        fragments[aid] = masks

    merged = merge_fragments(fragments, sitk.GetArrayFromImage(ref_img).shape)
    if anat_map is not None:
        merged = fill_anatomy_with_best_fragment(anat_map, merged)

    if output_path is not None:
        out = sitk.GetImageFromArray(merged.astype(np.uint8))
        out.CopyInformation(ref_img)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(out, output_path, True)
        print(f"[done] wrote {output_path} (labels {sorted(np.unique(merged).tolist())})")
    return merged, ref_img


def _build_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout = os.environ.get("PENGWIN_CLICK_LAYOUT", "pair")
    tta = os.environ.get("PENGWIN_TTA", "0") == "1"
    frag = FragmentPredictor(
        os.environ.get("PENGWIN_FRAG_MODEL", "/opt/ml/model/fragment"),
        fold=int(os.environ.get("PENGWIN_FRAG_FOLD", "0")),
        click_layout=layout, device=device, use_mirroring=tta)
    anat = None
    if os.environ.get("PENGWIN_USE_ANATOMY", "0") == "1":
        anat = AnatomyPredictor(
            os.environ.get("PENGWIN_ANAT_MODEL", "/opt/ml/model/anatomy"),
            fold=int(os.environ.get("PENGWIN_ANAT_FOLD", "0")), device=device, use_mirroring=tta)
    return frag, anat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", default=None, help="CT .mha (local mode; default: Grand Challenge input dir)")
    ap.add_argument("--clicks", default=None, help="clicks JSON (local mode)")
    ap.add_argument("--output", default=None, help="output .mha (local mode)")
    args = ap.parse_args()

    routing = os.environ.get("PENGWIN_ROUTING", "off") == "on"
    frag, anat = _build_models()

    if args.ct is not None:  # local single-case mode
        out = args.output or "prediction.mha"
        run_case(args.ct, args.clicks, out, frag, anat, routing)
        return

    # Grand Challenge mode
    ct_files = sorted(glob(os.path.join(GC_INPUT_CT_DIR, "*.mha")))
    assert len(ct_files) == 1, f"expected exactly one CT under {GC_INPUT_CT_DIR}, found {len(ct_files)}"
    out_path = os.path.join(GC_OUTPUT_DIR, os.path.basename(ct_files[0]))
    run_case(ct_files[0], GC_INPUT_CLICKS, out_path, frag, anat, routing)


if __name__ == "__main__":
    main()
