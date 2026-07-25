"""
Simulate the challenge submission setting on a fold's validation cases.

For each val case (from nnU-Net's ``splits_final.json``) and each click strategy, feed the
**raw CT + the provided clicks JSON** through the exact same inference pipeline the Grand
Challenge container uses, then score the resulting label map against the GT with
:mod:`pengwin_inference.metrics`. This is an offline stand-in for the hidden leaderboard.

Example:
    python -m pengwin_inference.evaluate_fold \
        --dataset 458 --fold 0 --click-layout nninteractive \
        --model $nnUNet_results/Dataset458_PENGWINfragOTF/trialsTrainerPengwinFragNNI__nnUNetResEncUNetLPlans_noResampling__3d_fullres_ps192 \
        --raw /home/j683r/local-work-temporary/ingest/Pengwin/Extracted \
        --out eval_fold0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json, join

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.training.dataloading.pengwin_clicks import STRATEGIES
from pengwin_inference.predict import FragmentPredictor, AnatomyPredictor
from pengwin_inference.inference import run_case, run_case_combined
from pengwin_inference import metrics as M

CLICKS_DIRNAME = "PENGWIN26_task2_train_clicks"


def val_ids_for_fold(dataset_id: int, fold: int):
    ds = maybe_convert_to_dataset_name(dataset_id)
    splits = load_json(join(nnUNet_preprocessed, ds, "splits_final.json"))
    return splits[fold]["val"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=int, required=True, help="nnU-Net dataset id (e.g. 458)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--model", required=True, help="trained fragment model folder (nnU-Net results dir)")
    ap.add_argument("--raw", required=True, help="PENGWIN 'Extracted' root (image.mha/label.mha per case)")
    ap.add_argument("--clicks-root", default=None, help="default: <raw>/PENGWIN26_task2_train_clicks")
    ap.add_argument("--click-layout", default="pair", choices=("pair", "nninteractive", "strategy"))
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES),
                    help="click strategies to evaluate (each CT+strategy is one challenge case)")
    ap.add_argument("--combine-strategies", action="store_true",
                    help="feed ALL --strategies at once (multiple clicks per fragment), one case per CT")
    ap.add_argument("--assembly", default="overwrite",
                    choices=("overwrite", "argmax", "ownership", "watershed", "smaller", "seeded"),
                    help="overlap resolution: overwrite (last-writer), argmax (prob competition), "
                         "ownership (nearest-click Voronoi), watershed (click-seeded, splits at "
                         "fracture neck), smaller (smaller fragment wins contested voxels)")
    ap.add_argument("--anatomy-model", default=None, help="optional anatomy model folder")
    ap.add_argument("--routing", action="store_true")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth",
                    help="checkpoint file to load, e.g. checkpoint_best.pth")
    ap.add_argument("--point-radius", type=float, default=4.0,
                    help="click ball radius for the nninteractive layout (train value 4; smaller = sharper)")
    ap.add_argument("--point-width", type=float, default=1.0,
                    help="click Gaussian sigma for the pair layout")
    ap.add_argument("--roi-mult", type=float, default=0.0,
                    help="0 (default) = full-volume sliding window (no accuracy risk). Set e.g. 1.5 "
                         "to predict only a roi_mult*patch_size window around each click for extra "
                         "speed (safe as long as fragments fit the window; verify parity first).")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--refine", type=int, default=0,
                    help="iterative-refinement passes (nninteractive layout only): feed the "
                         "previous prediction into the initSeg channel and re-predict")
    ap.add_argument("--save-preds", action="store_true", help="also write predicted .mha files")
    ap.add_argument("--limit", type=int, default=None, help="only evaluate the first N val cases (quick check)")
    ap.add_argument("--profile-memory", action="store_true",
                    help="report peak host RAM (RSS) and peak GPU VRAM per case + overall")
    ap.add_argument("--out", default="eval_fold", help="output directory for metrics")
    args = ap.parse_args()

    def host_rss_gb():
        # ru_maxrss is peak RSS; KB on Linux, bytes on macOS
        import sys
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024 ** 2) if sys.platform != "darwin" else rss / (1024 ** 3)

    raw = Path(args.raw)
    clicks_root = Path(args.clicks_root) if args.clicks_root else raw / CLICKS_DIRNAME
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roi_mult = args.roi_mult if args.roi_mult and args.roi_mult > 0 else None
    frag = FragmentPredictor(args.model, fold=args.fold, click_layout=args.click_layout,
                             device=device, use_mirroring=args.tta, checkpoint_name=args.checkpoint,
                             roi_mult=roi_mult, point_radius=args.point_radius, point_width=args.point_width,
                             refine_iters=args.refine)
    anat = (AnatomyPredictor(args.anatomy_model, fold=args.fold, device=device,
                             use_mirroring=args.tta, checkpoint_name=args.checkpoint)
            if args.anatomy_model else None)

    val_ids = val_ids_for_fold(args.dataset, args.fold)
    if args.limit:
        val_ids = val_ids[: args.limit]
    print(f"fold {args.fold}: {len(val_ids)} val cases x {len(args.strategies)} strategies")

    def clk_path(cid, strat):
        return clicks_root / strat / cid / "peripelvic-fragment-clicks.json"

    def eval_one(cid, gt, spacing_xyz, label, pred):
        m = M.score_case(gt, pred, spacing_xyz, iou_thr=args.iou_thr)
        m.update({"case": cid, "strategy": label})
        per_case.append(m)
        rows.append({k: m[k] for k in ("case", "strategy", "fracture_dice", "instance_f1",
                                       "instance_precision", "instance_recall", "hd95", "assd",
                                       "tp", "fp", "fn")})
        mem = ""
        if args.profile_memory:
            vram = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
            mem = f"  | RAM {host_rss_gb():.1f}GB peak, VRAM {vram:.1f}GB peak"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()  # so the next line is that case's peak
        print(f"  {cid}/{label}: Dice {m['fracture_dice']:.3f} F1 {m['instance_f1']:.3f} "
              f"HD95 {m['hd95']:.1f} (tp{m['tp']}/fp{m['fp']}/fn{m['fn']})"
              + (f"  OUT-OF-RANGE {m['out_of_range_pred_labels']}" if m['out_of_range_pred_labels'] else "")
              + mem)

    rows, per_case = [], []
    for cid in val_ids:
        ct = raw / cid / "image.mha"
        gt_img = sitk.ReadImage(str(raw / cid / "label.mha"))
        gt = sitk.GetArrayFromImage(gt_img)
        spacing_xyz = gt_img.GetSpacing()

        if args.combine_strategies:
            clks = [str(clk_path(cid, s)) for s in args.strategies if clk_path(cid, s).exists()]
            if not clks:
                print(f"  [skip] {cid}: no clicks"); continue
            out_mha = str(out / "preds" / f"{cid}_combined.mha") if args.save_preds else None
            pred, _ = run_case_combined(str(ct), clks, out_mha, frag, anat, args.routing, args.assembly)
            eval_one(cid, gt, spacing_xyz, "combined", pred)
        else:
            for strat in args.strategies:
                clk = clk_path(cid, strat)
                if not clk.exists():
                    print(f"  [skip] {cid}/{strat}: no clicks"); continue
                out_mha = str(out / "preds" / f"{cid}_{strat}.mha") if args.save_preds else None
                pred, _ = run_case(str(ct), str(clk), out_mha, frag, anat, args.routing, args.assembly)
                eval_one(cid, gt, spacing_xyz, strat, pred)

    agg = M.aggregate(per_case)
    json.dump({"aggregate": agg, "per_case": per_case}, open(out / "metrics.json", "w"), indent=2)
    with open(out / "per_case.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("\n=== AGGREGATE ===")
    for k, v in agg.items():
        print(f"  {k}: {v}")
    if args.profile_memory:
        vram = torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"\n=== PEAK MEMORY (whole process) ===")
        print(f"  host RAM (RSS): {host_rss_gb():.2f} GB   [phase limit 16 GB]")
        print(f"  GPU VRAM      : {vram:.2f} GB   [T4 limit 16 GB]")
    print(f"\nwrote {out/'metrics.json'} and {out/'per_case.csv'}")


if __name__ == "__main__":
    main()
