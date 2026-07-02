#!/usr/bin/env python3
"""
Convert the PENGWIN 2026 Task 2 (PENGWIN-Interact) training data into nnU-Net datasets.

Source layout (Zenodo / `…/Extracted`):
    <src>/<case_id>/image.mha          CT
    <src>/<case_id>/label.mha          instance labels, values 0 + {1-50 sacrum,
                                       51-100 left hip, 101-150 right hip, 151-200 femur}
    <src>/PENGWIN26_task2_train_clicks/<strategy>/<case_id>/peripelvic-fragment-clicks.json

This script can emit any subset of three nnU-Net datasets (select with flags):

  --anatomy      Dataset456_PENGWINanat   CT + 4 anatomy click heatmaps (5 ch) -> labels 0-4
                 (baseline Phase-1 model; OPTIONAL here)
  --frag-precomp Dataset457_PENGWINfrag   CT + fg + bg click heatmap (3 ch) -> labels 0/1
                 (baseline Phase-2 model, one case per fragment; ablation A)
  --frag-otf     Dataset458_PENGWINfragOTF CT (1 ch) + contiguous instance seg + clicksTr JSON
                 (clicks simulated on the fly by trialsTrainerPengwinFrag; ablation B / primary)

Coordinate convention: PENGWIN JSON points are (z, y, x) numpy/array order (validated on
the real data -- the baseline anatomy-heatmap script flips axes, we do NOT replicate that).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p, save_json,
)
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from nnunetv2.training.dataloading.pengwin_clicks import (
    ANATOMY_NAMES, ANATOMY_RANGES, STRATEGIES, label_to_anatomy_id, parse_pengwin_clicks,
    render_points_gauss,
)

CLICKS_DIRNAME = "PENGWIN26_task2_train_clicks"
FILE_ENDING = ".mha"


# --------------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------------
def list_cases(src: Path) -> List[str]:
    """Numbered case dirs that contain both image.mha and label.mha."""
    cases = []
    for d in sorted(p for p in src.iterdir() if p.is_dir() and p.name.isdigit()):
        if (d / "image.mha").exists() and (d / "label.mha").exists():
            cases.append(d.name)
    return cases


def read_arr(path: Path) -> Tuple[np.ndarray, sitk.Image]:
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def write_like(arr: np.ndarray, ref: sitk.Image, path: Path, dtype):
    img = sitk.GetImageFromArray(arr.astype(dtype))
    img.CopyInformation(ref)
    sitk.WriteImage(img, str(path), True)  # compressed


def copy_or_link_ct(src_ct: Path, dst_ct: Path, link: bool):
    if dst_ct.exists() or dst_ct.is_symlink():
        dst_ct.unlink()
    if link:
        os.symlink(os.path.abspath(src_ct), dst_ct)
    else:
        import shutil
        shutil.copy(str(src_ct), str(dst_ct))


def load_clicks(clicks_root: Path, strategy: str, case_id: str) -> Optional[dict]:
    p = clicks_root / strategy / case_id / "peripelvic-fragment-clicks.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# --------------------------------------------------------------------------------------
# Dataset 458: fragment, on-the-fly clicks (primary)
# --------------------------------------------------------------------------------------
def build_frag_otf(src: Path, cases: List[str], dataset_id: int, link_ct: bool):
    name = f"Dataset{dataset_id:03d}_PENGWINfragOTF"
    out = Path(nnUNet_raw) / name
    imagestr, labelstr, clickstr = out / "imagesTr", out / "labelsTr", out / "clicksTr"
    for d in (imagestr, labelstr, clickstr):
        maybe_mkdir_p(str(d))
    clicks_root = src / CLICKS_DIRNAME

    max_instances = 0
    for cid in cases:
        lab, ref = read_arr(src / cid / "label.mha")
        # Remap the (non-contiguous) PENGWIN instance ids to a contiguous 1..k map so
        # nnU-Net preprocessing accepts it. The mapping new_id -> {orig_label, anatomy}
        # is stored alongside the clicks for assembly/validation.
        uniq = [int(v) for v in np.unique(lab) if v > 0]
        remap = {orig: i + 1 for i, orig in enumerate(uniq)}
        inst = np.zeros_like(lab, dtype=np.int16)
        for orig, new in remap.items():
            inst[lab == orig] = new
        max_instances = max(max_instances, len(uniq))

        copy_or_link_ct(src / cid / "image.mha", imagestr / f"{cid}_0000{FILE_ENDING}", link_ct)
        write_like(inst, ref, labelstr / f"{cid}{FILE_ENDING}", np.int16)

        # Store every strategy's clicks + the instance mapping for this case.
        strategies = {s: load_clicks(clicks_root, s, cid) for s in STRATEGIES}
        instance_meta = {
            str(new): {"orig_label": orig, "anatomy_id": label_to_anatomy_id(orig)}
            for orig, new in remap.items()
        }
        save_json({"strategies": strategies, "instances": instance_meta},
                  str(clickstr / f"{cid}_clicks.json"), sort_keys=False)
        print(f"[458] {cid}: {len(uniq)} fragments")

    labels = {"background": 0}
    labels.update({f"frag_{i}": i for i in range(1, max_instances + 1)})
    generate_dataset_json(
        str(out), {0: "CT"}, labels=labels, num_training_cases=len(cases),
        file_ending=FILE_ENDING, dataset_name=name,
        description="PENGWIN Task2 fragments, on-the-fly clicks (binary target derived at runtime). "
                    f"Instance ids are per-case contiguous; max {max_instances} fragments/case.",
        overwrite_image_reader_writer="SimpleITKIO",
    )
    print(f"[458] wrote {name} ({len(cases)} cases, max {max_instances} fragments/case)")


# --------------------------------------------------------------------------------------
# Dataset 457: fragment, precomputed fg/bg heatmap channels (baseline-faithful)
# --------------------------------------------------------------------------------------
def build_frag_precomp(src: Path, cases: List[str], dataset_id: int, sigma: float,
                       strategy: Optional[str], link_ct: bool):
    name = f"Dataset{dataset_id:03d}_PENGWINfrag"
    out = Path(nnUNet_raw) / name
    imagestr, labelstr = out / "imagesTr", out / "labelsTr"
    for d in (imagestr, labelstr):
        maybe_mkdir_p(str(d))
    clicks_root = src / CLICKS_DIRNAME

    n_written = 0
    for cid in cases:
        lab, ref = read_arr(src / cid / "label.mha")
        strat = strategy or STRATEGIES[np.random.randint(len(STRATEGIES))]
        cj = load_clicks(clicks_root, strat, cid)
        if cj is None:
            print(f"[457] {cid}: missing clicks for {strat}, skipping")
            continue
        per_anat = parse_pengwin_clicks(cj)  # {anatomy_id: [(z,y,x), ...]}
        all_pts = [pt for pts in per_anat.values() for pt in pts]

        for aid, pts in per_anat.items():
            if len(pts) <= 1:
                continue  # single-fragment anatomy: copied from Phase 1 at inference
            for k, fg_pt in enumerate(pts):
                frag_label = int(lab[fg_pt])
                if frag_label == 0 or label_to_anatomy_id(frag_label) != aid:
                    continue  # click not inside its anatomy (shouldn't happen)
                target = (lab == frag_label).astype(np.uint8)
                fg = render_points_gauss(lab.shape, [fg_pt], sigma)
                bg = render_points_gauss(lab.shape, [p for p in all_pts if p != fg_pt], sigma)

                stem = f"case_{cid}_frag_{ANATOMY_NAMES[aid]}_{frag_label:03d}"
                copy_or_link_ct(src / cid / "image.mha", imagestr / f"{stem}_0000{FILE_ENDING}", link_ct)
                write_like(fg, ref, imagestr / f"{stem}_0001{FILE_ENDING}", np.float32)
                write_like(bg, ref, imagestr / f"{stem}_0002{FILE_ENDING}", np.float32)
                write_like(target, ref, labelstr / f"{stem}{FILE_ENDING}", np.uint8)
                n_written += 1
        print(f"[457] {cid}: strategy={strat}")

    generate_dataset_json(
        str(out), {0: "CT", 1: "Fragment clicks", 2: "Background clicks"},
        labels={"background": 0, "foreground": 1}, num_training_cases=n_written,
        file_ending=FILE_ENDING, dataset_name=name,
        description="PENGWIN Task2 fragment model: binary, precomputed fg/bg click heatmaps.",
        overwrite_image_reader_writer="SimpleITKIO",
    )
    print(f"[457] wrote {name} ({n_written} fragment cases)")


# --------------------------------------------------------------------------------------
# Dataset 456: anatomy (optional Phase-1 model)
# --------------------------------------------------------------------------------------
def build_anatomy(src: Path, cases: List[str], dataset_id: int, sigma: float,
                  strategy: Optional[str], link_ct: bool):
    name = f"Dataset{dataset_id:03d}_PENGWINanat"
    out = Path(nnUNet_raw) / name
    imagestr, labelstr = out / "imagesTr", out / "labelsTr"
    for d in (imagestr, labelstr):
        maybe_mkdir_p(str(d))
    clicks_root = src / CLICKS_DIRNAME

    for cid in cases:
        lab, ref = read_arr(src / cid / "label.mha")
        # remap 0-200 -> 0-4 anatomy
        anat = np.zeros_like(lab, dtype=np.uint8)
        for aid, (lo, hi) in ANATOMY_RANGES.items():
            anat[(lab >= lo) & (lab <= hi)] = aid

        strat = strategy or STRATEGIES[np.random.randint(len(STRATEGIES))]
        cj = load_clicks(clicks_root, strat, cid)
        per_anat = parse_pengwin_clicks(cj) if cj is not None else {}

        copy_or_link_ct(src / cid / "image.mha", imagestr / f"{cid}_0000{FILE_ENDING}", link_ct)
        for aid in (1, 2, 3, 4):
            heat = render_points_gauss(lab.shape, per_anat.get(aid, []), sigma)
            write_like(heat, ref, imagestr / f"{cid}_000{aid}{FILE_ENDING}", np.float32)
        write_like(anat, ref, labelstr / f"{cid}{FILE_ENDING}", np.uint8)
        print(f"[456] {cid}: strategy={strat}")

    generate_dataset_json(
        str(out),
        {0: "CT", 1: "Sacrum clicks", 2: "Left hip clicks", 3: "Right hip clicks", 4: "Femur clicks"},
        labels={"background": 0, "sacrum": 1, "left hip": 2, "right hip": 3, "femur": 4},
        num_training_cases=len(cases), file_ending=FILE_ENDING, dataset_name=name,
        description="PENGWIN Task2 anatomy model: 5-class semantic + 4 anatomy click heatmaps.",
        overwrite_image_reader_writer="SimpleITKIO",
    )
    print(f"[456] wrote {name} ({len(cases)} cases)")


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="PENGWIN 'Extracted' root")
    ap.add_argument("--anatomy", action="store_true")
    ap.add_argument("--frag-precomp", action="store_true")
    ap.add_argument("--frag-otf", action="store_true")
    ap.add_argument("--id-anatomy", type=int, default=456)
    ap.add_argument("--id-frag-precomp", type=int, default=457)
    ap.add_argument("--id-frag-otf", type=int, default=458)
    ap.add_argument("--sigma", type=float, default=1.0, help="Gaussian sigma for click heatmaps")
    ap.add_argument("--strategy", default=None, choices=list(STRATEGIES),
                    help="Fix the click strategy (default: random per case)")
    ap.add_argument("--copy-ct", action="store_true",
                    help="Copy CT files instead of symlinking (symlink is the default to save disk)")
    ap.add_argument("--limit", type=int, default=None, help="Only process the first N cases (debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    src = Path(args.src)
    cases = list_cases(src)
    if args.limit:
        cases = cases[: args.limit]
    print(f"Found {len(cases)} cases in {src}")
    if not (args.anatomy or args.frag_precomp or args.frag_otf):
        ap.error("select at least one of --anatomy / --frag-precomp / --frag-otf")

    link = not args.copy_ct
    if args.frag_otf:
        build_frag_otf(src, cases, args.id_frag_otf, link)
    if args.frag_precomp:
        build_frag_precomp(src, cases, args.id_frag_precomp, args.sigma, args.strategy, link)
    if args.anatomy:
        build_anatomy(src, cases, args.id_anatomy, args.sigma, args.strategy, link)


if __name__ == "__main__":
    main()
