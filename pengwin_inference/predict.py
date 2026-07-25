"""
Model wrappers for PENGWIN Task 2 inference.

The fragment model is click-conditioned: clicks are injected as raw heatmap channels
*after* the CT has been normalized/resampled by the model's preprocessing -- exactly the
layout the trainers (`trialsTrainerPengwinFrag` / `...NNI`) saw during training. We therefore
preprocess the CT once per case, then render the click channels on the resampled grid for
each fragment (mapping the original (z,y,x) click coordinates through nnU-Net's
crop+resample via :func:`preprocess_point`).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.training.dataloading.utils import preprocess_point
from nnunetv2.training.dataloading.nnInteractive_clicks import PointInteraction_stub
from nnunetv2.training.dataloading.pengwin_clicks import render_points_gauss, zyx_to_xyz, Coord
from nnunetv2.utilities.label_handling.label_handling import LabelManager

# The fragment network ALWAYS outputs 2 channels (binary), regardless of how many instance
# labels the training dataset declared, so logits must be converted with a binary label
# manager (not the dataset's, which may declare N+1 fragment slots).
_BINARY_LABEL_MANAGER = LabelManager({"background": 0, "foreground": 1}, regions_class_order=None)


def _build_predictor(model_folder: str, fold, device: torch.device, use_mirroring: bool,
                     checkpoint_name: str = "checkpoint_final.pth") -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=True, device=device, verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,),
                                                   checkpoint_name=checkpoint_name)
    return predictor


def restore_labelmap_to_original(seg_grid: np.ndarray, plans_manager, configuration_manager,
                                 properties: dict) -> np.ndarray:
    """Map an integer label map from the model's preprocessed grid back to the original image
    space (revert resample -> uncrop -> transpose). Mirrors the shape-restoration in
    ``convert_predicted_logits_to_segmentation_with_correct_shape`` but for an int seg, so it
    can be run ONCE on the assembled instance map instead of per fragment. Uses the seg
    resampler (nearest / identity for the noResampling plans)."""
    from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
    spacing_transposed = [properties['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == len(properties['shape_after_cropping_and_before_resampling']) \
        else [spacing_transposed[0], *configuration_manager.spacing]
    seg = configuration_manager.resampling_fn_seg(
        seg_grid[None].astype(np.float32),
        properties['shape_after_cropping_and_before_resampling'],
        current_spacing,
        [properties['spacing'][i] for i in plans_manager.transpose_forward],
        is_seg=True)[0]
    seg = np.rint(np.asarray(seg)).astype(np.uint16)
    out = np.zeros(properties['shape_before_cropping'], dtype=np.uint16)
    out = insert_crop_into_image(out, seg, properties['bbox_used_for_cropping'])
    if isinstance(out, torch.Tensor):
        out = out.cpu().numpy()
    return out.transpose(plans_manager.transpose_backward)


def _preprocess_ct(predictor: nnUNetPredictor, ct_path: str) -> Tuple[np.ndarray, dict]:
    """Normalize + resample the CT through the model's own preprocessing. Returns the
    preprocessed data (1, Z', Y', X') and the properties dict (with crop bbox + shapes)."""
    pp = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = pp.run_case([ct_path], None, predictor.plans_manager,
                                      predictor.configuration_manager, predictor.dataset_json)
    return data, properties


class FragmentPredictor:
    """Binary, click-conditioned fragment model (ablations B / C)."""

    def __init__(self, model_folder: str = None, fold=0, click_layout: str = "pair",
                 point_width: float = 1.0, point_radius: float = 4.0,
                 device: torch.device = torch.device("cuda"), use_mirroring: bool = False,
                 predictor: Optional[nnUNetPredictor] = None,
                 checkpoint_name: str = "checkpoint_final.pth",
                 roi_mult: Optional[float] = None, n_strat: int = 4,
                 refine_iters: int = 0):
        assert click_layout in ("pair", "nninteractive", "strategy")
        self.n_strat = n_strat  # for the "strategy" layout: one ⊕/⊖ channel pair per strategy
        # refine_iters: extra passes that feed the previous binary prediction back into the
        # nnInteractive initSeg channel (ch 0). 0 = single-shot. Only valid for the
        # "nninteractive" layout (the only one with an initSeg channel).
        self.refine_iters = refine_iters
        # roi_mult: if set, predict only a window of size roi_mult * patch_size centred on the
        # click instead of sliding over the whole volume -- a large speedup, since a fragment
        # is local. None -> full-volume sliding window.
        self.roi_mult = roi_mult
        # ``predictor`` lets callers inject a manually-initialized predictor (e.g. for the
        # raw nnInteractive checkpoint whose stub trainer is not importable via the folder API).
        self.predictor = predictor if predictor is not None else _build_predictor(
            model_folder, fold, device, use_mirroring, checkpoint_name)
        # The network head is binary (2 ch) even though the training dataset may declare many
        # instance-label slots. Force the predictor's label manager to binary so the sliding
        # window allocates a 2-channel logits buffer (avoids a 'got 31, expected 2' crash and
        # a ~15x memory blow-up).
        self.predictor.label_manager = _BINARY_LABEL_MANAGER
        self.click_layout = click_layout
        self.point_width = point_width
        self.point_radius = point_radius

    def preprocess_ct(self, ct_path: str):
        data, properties = _preprocess_ct(self.predictor, ct_path)
        return torch.from_numpy(data).float(), properties

    def _click_channels(self, shape, fg_zyx_grid: List[Coord], bg_zyx_grid: List[Coord],
                        strategy_idx: int = 0) -> torch.Tensor:
        if self.click_layout == "pair":
            fg = render_points_gauss(shape, fg_zyx_grid, self.point_width)
            bg = render_points_gauss(shape, bg_zyx_grid, self.point_width)
            return torch.from_numpy(np.stack([fg, bg], 0)).float()
        pi = PointInteraction_stub(point_radius=self.point_radius, use_distance_transform=True)
        if self.click_layout == "strategy":
            # 2*n_strat channels: ⊕ in slot [strategy_idx], ⊖ in slot [n_strat+strategy_idx]
            ch = torch.zeros((2 * self.n_strat, *shape), dtype=torch.float32)
            fi, bi = strategy_idx, self.n_strat + strategy_idx
            for fc in fg_zyx_grid:
                ch[fi] = pi.place_point(tuple(int(v) for v in fc), ch[fi], binarize=False)
            for bc in bg_zyx_grid:
                ch[bi] = pi.place_point(tuple(int(v) for v in bc), ch[bi], binarize=False)
            return ch
        # nninteractive: 7 interaction channels, point⊕=slot 3, point⊖=slot 4
        ch = torch.zeros((7, *shape), dtype=torch.float32)
        for fc in fg_zyx_grid:
            ch[3] = pi.place_point(tuple(int(v) for v in fc), ch[3], binarize=False)
        for bc in bg_zyx_grid:
            ch[4] = pi.place_point(tuple(int(v) for v in bc), ch[4], binarize=False)
        return ch

    def predict_fragment(self, ct_data: torch.Tensor, properties: dict,
                         fg_clicks_zyx: List[Coord], bg_clicks_zyx: List[Coord],
                         return_prob: bool = False, strategy_idx: int = 0):
        """Predict the fragment marked by ``fg_clicks_zyx`` (one or more positive clicks on the
        SAME fragment) with the other fragments' clicks as negatives.

        Returns ``(out_grid, fg_grid_click)`` in the model's *preprocessed grid* space (NOT
        original) -- ``out_grid`` is the binary mask (uint8), or the foreground probability
        (float32) if ``return_prob``. Staying in grid space avoids the expensive per-fragment
        resample-back; callers assemble in grid space and restore to original ONCE via
        :func:`restore_labelmap_to_original`.
        """
        grid_shape = tuple(ct_data.shape[1:])
        # map original (z,y,x) clicks onto the resampled grid (preprocess_point expects [x,y,z])
        fg_grid = [tuple(preprocess_point(zyx_to_xyz(c), properties, grid_shape)) for c in fg_clicks_zyx]
        bg_grid = [tuple(preprocess_point(zyx_to_xyz(c), properties, grid_shape)) for c in bg_clicks_zyx]

        def _fg(logits):  # foreground prob (float32) or binary (uint8)
            v = torch.softmax(logits, 0)[1] if return_prob else (logits[1] > logits[0])
            return v.cpu().numpy().astype(np.float32 if return_prob else np.uint8)

        refine = self.refine_iters
        if refine and self.click_layout != "nninteractive":
            raise ValueError("refine_iters needs the 'nninteractive' layout (initSeg channel 0)")

        def _run(ct_win, clicks):
            """Forward pass(es) over one window; between passes write the current binary mask
            into the initSeg channel (ch 0) so the net can correct itself. Returns final logits."""
            logits = None
            for it in range(refine + 1):
                logits = self.predictor.predict_sliding_window_return_logits(
                    torch.cat((ct_win, clicks), dim=0))
                if it < refine:  # feed prediction back as initSeg for the next pass
                    clicks[0] = (logits[1] > logits[0]).float().cpu()
            return logits

        out_grid = np.zeros(grid_shape, dtype=np.float32 if return_prob else np.uint8)
        if self.roi_mult is None:
            clicks = self._click_channels(grid_shape, fg_grid, bg_grid, strategy_idx)
            out_grid[:] = _fg(_run(ct_data, clicks))
        else:
            # crop a window of roi_mult * patch_size centred on the (first) positive click
            patch = self.predictor.configuration_manager.patch_size
            roi = [int(round(p * self.roi_mult)) for p in patch]
            starts = [min(max(0, c - r // 2), max(0, s - r)) for c, r, s in zip(fg_grid[0], roi, grid_shape)]
            ends = [min(s, st + r) for st, r, s in zip(starts, roi, grid_shape)]
            sl = tuple(slice(st, en) for st, en in zip(starts, ends))
            win_shape = tuple(en - st for st, en in zip(starts, ends))
            inside = lambda p: all(st <= pi < en for pi, st, en in zip(p, starts, ends))
            to_win = lambda p: tuple(int(pi) - st for pi, st in zip(p, starts))
            fg_win = [to_win(p) for p in fg_grid if inside(p)]
            bg_win = [to_win(p) for p in bg_grid if inside(p)]
            clicks = self._click_channels(win_shape, fg_win, bg_win, strategy_idx)
            out_grid[sl] = _fg(_run(ct_data[(slice(None), *sl)], clicks))
        return out_grid, fg_grid[0]


class AnatomyPredictor:
    """Optional 5-class anatomy model (Phase 1). Input = CT + 4 anatomy click heatmaps."""

    def __init__(self, model_folder: str, fold=0, point_width: float = 1.0,
                 device: torch.device = torch.device("cuda"), use_mirroring: bool = False,
                 checkpoint_name: str = "checkpoint_final.pth"):
        self.predictor = _build_predictor(model_folder, fold, device, use_mirroring, checkpoint_name)
        self.point_width = point_width

    def predict(self, ct_path: str, per_anatomy_clicks: Dict[int, List[Coord]]):
        """Return a 5-class anatomy map (0-4) in original space + the properties dict."""
        data, properties = _preprocess_ct(self.predictor, ct_path)
        ct = torch.from_numpy(data).float()
        grid_shape = tuple(ct.shape[1:])
        chans = [ct]
        for aid in (1, 2, 3, 4):
            pts = [tuple(preprocess_point(zyx_to_xyz(c), properties, grid_shape))
                   for c in per_anatomy_clicks.get(aid, [])]
            chans.append(torch.from_numpy(render_points_gauss(grid_shape, pts, self.point_width))[None])
        net_in = torch.cat(chans, dim=0)
        logits = self.predictor.predict_sliding_window_return_logits(net_in).cpu()
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, self.predictor.plans_manager, self.predictor.configuration_manager,
            self.predictor.label_manager, properties)
        return np.asarray(seg).astype(np.uint8), properties
