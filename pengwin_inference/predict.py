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


def _build_predictor(model_folder: str, fold, device: torch.device, use_mirroring: bool) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=use_mirroring,
        perform_everything_on_device=True, device=device, verbose=False,
        verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_folder, use_folds=(fold,),
                                                   checkpoint_name="checkpoint_final.pth")
    return predictor


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
                 predictor: Optional[nnUNetPredictor] = None):
        assert click_layout in ("pair", "nninteractive")
        # ``predictor`` lets callers inject a manually-initialized predictor (e.g. for the
        # raw nnInteractive checkpoint whose stub trainer is not importable via the folder API).
        self.predictor = predictor if predictor is not None else _build_predictor(
            model_folder, fold, device, use_mirroring)
        self.click_layout = click_layout
        self.point_width = point_width
        self.point_radius = point_radius

    def preprocess_ct(self, ct_path: str):
        data, properties = _preprocess_ct(self.predictor, ct_path)
        return torch.from_numpy(data).float(), properties

    def _click_channels(self, shape, fg_zyx_grid: Optional[Coord], bg_zyx_grid: List[Coord]) -> torch.Tensor:
        if self.click_layout == "pair":
            fg = render_points_gauss(shape, [] if fg_zyx_grid is None else [fg_zyx_grid], self.point_width)
            bg = render_points_gauss(shape, bg_zyx_grid, self.point_width)
            return torch.from_numpy(np.stack([fg, bg], 0)).float()
        ch = torch.zeros((7, *shape), dtype=torch.float32)
        pi = PointInteraction_stub(point_radius=self.point_radius, use_distance_transform=True)
        if fg_zyx_grid is not None:
            ch[3] = pi.place_point(tuple(int(v) for v in fg_zyx_grid), ch[3], binarize=False)
        for bc in bg_zyx_grid:
            ch[4] = pi.place_point(tuple(int(v) for v in bc), ch[4], binarize=False)
        return ch

    def predict_fragment(self, ct_data: torch.Tensor, properties: dict,
                         fg_click_zyx: Coord, bg_clicks_zyx: List[Coord]) -> np.ndarray:
        """Predict the binary mask (in *original* image space) for the fragment marked by
        ``fg_click_zyx`` with the other fragments' clicks as negatives. Clicks are original
        (z, y, x) coordinates."""
        grid_shape = tuple(ct_data.shape[1:])
        # map original (z,y,x) clicks onto the resampled grid (preprocess_point expects [x,y,z])
        fg_grid = tuple(preprocess_point(zyx_to_xyz(fg_click_zyx), properties, grid_shape))
        bg_grid = [tuple(preprocess_point(zyx_to_xyz(c), properties, grid_shape)) for c in bg_clicks_zyx]

        clicks = self._click_channels(grid_shape, fg_grid, bg_grid)
        net_in = torch.cat((ct_data, clicks), dim=0)
        logits = self.predictor.predict_sliding_window_return_logits(net_in).cpu()
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, self.predictor.plans_manager, self.predictor.configuration_manager,
            _BINARY_LABEL_MANAGER, properties)
        return (np.asarray(seg) > 0).astype(np.uint8), fg_grid


class AnatomyPredictor:
    """Optional 5-class anatomy model (Phase 1). Input = CT + 4 anatomy click heatmaps."""

    def __init__(self, model_folder: str, fold=0, point_width: float = 1.0,
                 device: torch.device = torch.device("cuda"), use_mirroring: bool = False):
        self.predictor = _build_predictor(model_folder, fold, device, use_mirroring)
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
