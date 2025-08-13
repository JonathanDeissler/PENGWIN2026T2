import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP


from batchviewer import view_batch
from tqdm import tqdm
import json
import csv
from nnunetv2.evaluation.challenge_eval_util import process_case, group_jsons_by_prefix, aggregate_group_metrics
from nnunetv2.training.dataloading.nnInteractive_clicks import build_point
from nnunetv2.training.dataloading.utils import restructure_clicks, sparse_to_dense_point_gauss, \
    generated_sparse_to_dense_point_nnInteractive, sparse_to_dense_point_nnInteractive, place_precomputed_clicks, \
    select_interactions_based_on_epochs
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, evaluate_simple_entry_point
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.misalign import Misalign2
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.data_loader_clicks import nnUNetDataLoaderClicks, nnUNetDataLoaderClicksGenerated, \
    nnUNetDataLoaderClicksGeneratedHelper, nnUNetDataLoaderClicksGeneratedDebug, \
    nnUNetDataLoaderClicksGeneratedAdvancedGeneration, nnUNetDataLoaderClicksGeneratedNoPlace
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2, nnUNetDatasetHelperSeg
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, FocalTversky_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss, TverskyLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler, PytorchCompliantPolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.optim.lr_scheduler import SequentialLR, LinearLR
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler  # nnU-Net's PolyLR


class trialsTrainer(nnUNetTrainer):
    os.environ["STEM"] = "trials"
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []

        # transforms.append(
        #     Misalign2(
        #         im_channels_2_misalign=(0,),
        #         asynchron=False,
        #
        #         squeezing_xyz=(0, 0, 0),
        #         p_squeeze=0.0,
        #
        #         rotation_sag_cor_ax=(5, 5, 5),
        #         rad_or_deg="deg",
        #         p_rotation=0.1,
        #
        #         shift_zyx=(0, 2, 2),
        #         p_shift=0.1,
        #     )
        # )

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)


    def _build_loss(self):
            # set smooth to 0
            if self.label_manager.has_regions:
                loss = DC_and_BCE_loss({},
                                       {'batch_dice': self.configuration_manager.batch_dice,
                                        'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                       use_ignore_label=self.label_manager.ignore_label is not None,
                                       dice_class=MemoryEfficientSoftDiceLoss)
            else:
                loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                       'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                      ignore_label=self.label_manager.ignore_label,
                                      dice_class=MemoryEfficientSoftDiceLoss)

            if self.enable_deep_supervision:
                deep_supervision_scales = self._get_deep_supervision_scales()

                # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
                weights[-1] = 0

                # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
                weights = weights / weights.sum()
                # now wrap the loss
                loss = DeepSupervisionWrapper(loss, weights)
            return loss


    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        architecture_class_name = "nnunetv2.architecture.ResidualEncoderUNetPoints.ResidualEncoderUNetPoints"

        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels + 2,  # positive / negative clicks
                                                        num_output_channels, enable_deep_supervision)


    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val


    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = nnUNetDatasetBlosc2(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)


        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicks(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoaderClicks(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [(i[:,:1].to(self.device, non_blocking=True), i[:,1:].to(self.device, non_blocking=True)) for i in target]
            target, target_organs = zip(*target)

        else:
            #raise NotImplementedError()
            target = target.to(self.device, non_blocking=True)

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0, 0].detach().cpu().numpy(), name='data')
        # viewer.add_labels(target[0][0,0].detach().cpu().numpy().astype(np.uint8), name='target')
        # viewer.add_labels(target_organs[0][0,0].detach().cpu().numpy().astype(np.uint8), name='target_organs')
        # napari.run()


        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, )
            # del data
            #
            # import IPython;IPython.embed()
            # from BatchViewer import view_batch
            # view_batch(data[1], target[0][1], output[0][1])

            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [(i[:,:1].to(self.device, non_blocking=True), i[:,1:].to(self.device, non_blocking=True)) for i in target]
            target, target_organs = zip(*target)
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data,)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


    def perform_actual_validation(self, save_probabilities: bool = False):
        for i in range (2):

            if i == 0:
                print("performing val without clicks")
                use_clicks = False
            else:
                print("performing val with clicks")
                use_clicks = True

            self.set_deep_supervision_enabled(False)
            self.network.eval()

            if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
                self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                       "encounter crashes in validation then this is because torch.compile forgets "
                                       "to trigger a recompilation of the model with deep supervision disabled. "
                                       "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                       "validation with --val (exactly the same as before) and then it will work. "
                                       "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                       "forward pass (where compile is triggered) already has deep supervision disabled. "
                                       "This is exactly what we need in perform_actual_validation")

            predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                        perform_everything_on_device=True, device=self.device, verbose=False,
                                        verbose_preprocessing=False, allow_tqdm=False)
            predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                            self.dataset_json, self.__class__.__name__,
                                            self.inference_allowed_mirroring_axes)

            with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
                worker_list = [i for i in segmentation_export_pool._pool]
                if use_clicks:
                    validation_output_folder = join(self.output_folder, 'validation')
                else:
                    validation_output_folder = join(self.output_folder, 'validation_no_clicks')
                maybe_mkdir_p(validation_output_folder)

                # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
                # the validation keys across the workers.
                _, val_keys = self.do_split()
                if self.is_ddp:
                    last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                    val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                    # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                    # different

                dataset_val = nnUNetDatasetBlosc2(self.preprocessed_dataset_folder, val_keys,
                                            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                            )

                next_stages = self.configuration_manager.next_stage_names

                if next_stages is not None:
                    _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]

                results = []

                for i, k in enumerate(dataset_val.get_dataset_identifiers()):
                    proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                               allowed_num_queued=2)
                    while not proceed:
                        sleep(0.1)
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                   allowed_num_queued=2)

                    self.print_to_log_file(f"predicting {k}")
                    data, seg, seg_org, properties, clicks = dataset_val.load_case_with_clicks(k)
                    shape = data.shape[1:]

                    if use_clicks:
                        clicks = restructure_clicks(clicks)
                        pos_clicks, neg_clicks = sparse_to_dense_point_gauss(clicks["points"], shape, properties, sigma=3)

                    else:
                        pos_clicks = np.zeros(shape, dtype=np.float32)
                        neg_clicks = np.zeros(shape, dtype=np.float32)

                    clicks_stacked = np.vstack((np.expand_dims(pos_clicks, axis=0), np.expand_dims(neg_clicks, axis=0)))
                    clicks_stacked = torch.from_numpy(clicks_stacked).float()
                    data = torch.from_numpy(np.asarray(data)).float()
                    data = torch.cat((data, clicks_stacked), dim=0)

                    if self.is_cascaded:
                        data = np.vstack((data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                            output_dtype=data.dtype)))
                    with warnings.catch_warnings():
                        # ignore 'The given NumPy array is not writable' warning
                        warnings.simplefilter("ignore")
                        if type(data) is torch.Tensor:
                            pass
                        else:
                            data = torch.from_numpy(data)

                    self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
                    output_filename_truncated = join(validation_output_folder, k)

                    prediction = predictor.predict_sliding_window_return_logits(data)
                    prediction = prediction.cpu()

                    # this needs to go into background processes
                    results.append(
                        segmentation_export_pool.starmap_async(
                            export_prediction_from_logits, (
                                (prediction, properties, self.configuration_manager, self.plans_manager,
                                 self.dataset_json, output_filename_truncated, save_probabilities),
                            )
                        )
                    )
                    # for debug purposes
                    # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                    #              output_filename_truncated, save_probabilities)

                    # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                    if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                        dist.barrier()

                _ = [r.get() for r in results]

            if self.is_ddp:
                dist.barrier()

            if self.local_rank == 0:
                metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
                                                    validation_output_folder,
                                                    join(validation_output_folder, 'summary.json'),
                                                    self.plans_manager.image_reader_writer_class(),
                                                    self.dataset_json["file_ending"],
                                                    self.label_manager.foreground_regions if self.label_manager.has_regions else
                                                    self.label_manager.foreground_labels,
                                                    self.label_manager.ignore_label, chill=True,
                                                    num_processes=default_num_processes * dist.get_world_size() if
                                                    self.is_ddp else default_num_processes)
                self.print_to_log_file("Validation complete", also_print_to_console=True)
                self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
                                       also_print_to_console=True)

            self.set_deep_supervision_enabled(True)
            compute_gaussian.cache_clear()


class trialsTrainerClickGen(nnUNetTrainer):
    os.environ["STEM"] = "trials"

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False
        self.point_width =1.5
        # self.num_iterations_per_epoch = 5
        # self.num_val_iterations_per_epoch = 5

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        architecture_class_name = "nnunetv2.architecture.ResidualEncoderUNetPoints.ResidualEncoderUNetPoints"

        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels + 2,  # positive / negative clicks
                                                        num_output_channels, enable_deep_supervision)

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys, val_keys = self.do_split()

        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = self.dataset_class(self.preprocessed_dataset_folder, tr_keys,
                                        folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
        return dataset_tr, dataset_val

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = nnUNetDatasetBlosc2(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicksGenerated(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)
        dl_val = nnUNetDataLoaderClicksGenerated(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [(i[:, :1].to(self.device, non_blocking=True), i[:, 1:].to(self.device, non_blocking=True)) for i
                      in target]
            target, target_organs = zip(*target)

        else:
            # raise NotImplementedError()
            target = target.to(self.device, non_blocking=True)

        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(data[0, 0].detach().cpu().numpy(), name='data')
        # viewer.add_labels(target[0][0,0].detach().cpu().numpy().astype(np.uint8), name='target')
        # viewer.add_labels(target_organs[0][0,0].detach().cpu().numpy().astype(np.uint8), name='target_organs')
        # napari.run()

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, )

            # del data
            l = self.loss(output, target)
            # print('train for key', batch['keys'],l.item())
            #
            # # import IPython;IPython.embed()
            #
            # from batchviewer import view_batch
            # view_batch(data[1], target[1], torch.softmax(output[1], dim = 0))

            # import napari
            # viewer = napari.Viewer()
            #
            # viewer.add_image(data[1][0].detach().cpu().numpy(), name='CT')
            # viewer.add_labels(target[1][0].detach().cpu().numpy(), name='segmentation')
            # viewer.add_image(data[1][1].detach().cpu().numpy(), name='positive clicks da')
            # viewer.add_image(data[1][2].detach().cpu().numpy(), name='negative clicks da')
            # output_converted = torch.softmax(output[1], 0).detach().cpu().numpy()
            # viewer.add_image(output_converted[0], name='output_background')
            # viewer.add_image(output_converted[1], name='output_tumor')
            # viewer.add_image(output_converted[2], name='output_liver')
            # # viewer.add_image(torch.softmax(output[0][1],0).detach().cpu().numpy(), name='output_tumor')
            # # viewer.add_image(torch.softmax(output[0][1],0).detach().cpu().numpy(), name='output_liver')
            # napari.run()

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [(i[:, :1].to(self.device, non_blocking=True), i[:, 1:].to(self.device, non_blocking=True)) for i
                      in target]
            target, target_organs = zip(*target)
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, )
            del data
            l = self.loss(output, target)
            # print('val for key', batch['keys'],l.item())
            # if "venous_221" in batch['keys']:
            #     from batchviewer import view_batch
            #     view_batch(data[1], target[1], torch.softmax(output[1], dim=0))
            # # import IPython;IPython.embed()
            # from batchviewer import view_batch
            # view_batch(data[1], target[1], torch.softmax(output[1], dim=0))

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    # def perform_actual_validation_old(self, save_probabilities: bool = False):
    #     for i in range(2):
    #         if i == 0:
    #             continue
    #             print("performing val without clicks")
    #             use_clicks = False
    #         else:
    #             print("performing val with clicks")
    #             use_clicks = True
    #
    #         self.set_deep_supervision_enabled(False)
    #         self.network.eval()
    #
    #         if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
    #             self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
    #                                    "encounter crashes in validation then this is because torch.compile forgets "
    #                                    "to trigger a recompilation of the model with deep supervision disabled. "
    #                                    "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
    #                                    "validation with --val (exactly the same as before) and then it will work. "
    #                                    "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
    #                                    "forward pass (where compile is triggered) already has deep supervision disabled. "
    #                                    "This is exactly what we need in perform_actual_validation")
    #
    #         predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
    #                                     perform_everything_on_device=True, device=self.device, verbose=False,
    #                                     verbose_preprocessing=False, allow_tqdm=False)
    #         predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
    #                                         self.dataset_json, self.__class__.__name__,
    #                                         self.inference_allowed_mirroring_axes)
    #
    #         with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
    #             worker_list = [i for i in segmentation_export_pool._pool]
    #             if use_clicks:
    #                 validation_output_folder = join(self.output_folder, 'validation')
    #             else:
    #                 validation_output_folder = join(self.output_folder, 'validation_no_clicks')
    #             maybe_mkdir_p(validation_output_folder)
    #
    #             # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
    #             # the validation keys across the workers.
    #             _, val_keys = self.do_split()
    #             if self.is_ddp:
    #                 last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1
    #
    #                 val_keys = val_keys[self.local_rank:: dist.get_world_size()]
    #                 # we cannot just have barriers all over the place because the number of keys each GPU receives can be
    #                 # different
    #
    #             dataset_val = nnUNetDatasetBlosc2(self.preprocessed_dataset_folder, val_keys,
    #                                               folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
    #                                               )
    #
    #             next_stages = self.configuration_manager.next_stage_names
    #
    #             if next_stages is not None:
    #                 _ = [maybe_mkdir_p(join(self.output_folder_base, 'predicted_next_stage', n)) for n in next_stages]
    #
    #             results = []
    #
    #             for i, k in enumerate(dataset_val.get_dataset_identifiers()):
    #                 proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
    #                                                            allowed_num_queued=2)
    #                 while not proceed:
    #                     sleep(0.1)
    #                     proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
    #                                                                allowed_num_queued=2)
    #
    #                 self.print_to_log_file(f"predicting {k}")
    #                 data, seg, seg_org, properties, clicks = dataset_val.load_case_with_clicks(k)
    #                 shape = data.shape[1:]
    #
    #
    #
    #                 if use_clicks:
    #                     clicks = restructure_clicks(clicks)
    #                     pos_clicks, neg_clicks = sparse_to_dense_point_nnInteractive(clicks["points"], shape,properties,
    #                                                                          sigma=self.point_width)
    #
    #                 else:
    #                     pos_clicks = np.zeros(shape, dtype=np.float32)
    #                     neg_clicks = np.zeros(shape, dtype=np.float32)
    #
    #
    #                 clicks_stacked = np.vstack((np.expand_dims(pos_clicks, axis=0), np.expand_dims(neg_clicks, axis=0)))
    #                 clicks_stacked = torch.from_numpy(clicks_stacked).float()
    #                 data = torch.from_numpy(np.asarray(data)).float()
    #                 data = torch.cat((data, clicks_stacked), dim=0)
    #
    #                 # import napari
    #                 # viewer = napari.Viewer()
    #                 # viewer.add_image(data[0].numpy(), name='CT')
    #                 # viewer.add_image(data[1], name='pos')
    #                 # viewer.add_image(data[2], name='neg')
    #                 # napari.run()
    #
    #                 if self.is_cascaded:
    #                     data = np.vstack(
    #                         (data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
    #                                                            output_dtype=data.dtype)))
    #                 with warnings.catch_warnings():
    #                     # ignore 'The given NumPy array is not writable' warning
    #                     warnings.simplefilter("ignore")
    #                     if type(data) is torch.Tensor:
    #                         pass
    #                     else:
    #                         data = torch.from_numpy(data)
    #
    #                 self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')
    #                 output_filename_truncated = join(validation_output_folder, k)
    #
    #                 prediction = predictor.predict_sliding_window_return_logits(data)
    #                 prediction = prediction.cpu()
    #
    #                 # this needs to go into background processes
    #                 results.append(
    #                     segmentation_export_pool.starmap_async(
    #                         export_prediction_from_logits, (
    #                             (prediction, properties, self.configuration_manager, self.plans_manager,
    #                              self.dataset_json, output_filename_truncated, save_probabilities),
    #                         )
    #                     )
    #                 )
    #                 # for debug purposes
    #                 # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
    #                 #              output_filename_truncated, save_probabilities)
    #
    #                 # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
    #                 if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
    #                     dist.barrier()
    #
    #             _ = [r.get() for r in results]
    #
    #         if self.is_ddp:
    #             dist.barrier()
    #
    #         if self.local_rank == 0:
    #             metrics = compute_metrics_on_folder(join(self.preprocessed_dataset_folder_base, 'gt_segmentations'),
    #                                                 validation_output_folder,
    #                                                 join(validation_output_folder, 'summary.json'),
    #                                                 self.plans_manager.image_reader_writer_class(),
    #                                                 self.dataset_json["file_ending"],
    #                                                 self.label_manager.foreground_regions if self.label_manager.has_regions else
    #                                                 self.label_manager.foreground_labels,
    #                                                 self.label_manager.ignore_label, chill=True,
    #                                                 num_processes=default_num_processes * dist.get_world_size() if
    #                                                 self.is_ddp else default_num_processes)
    #             self.print_to_log_file("Validation complete", also_print_to_console=True)
    #             self.print_to_log_file("Mean Validation Dice: ", (metrics['foreground_mean']["Dice"]),
    #                                    also_print_to_console=True)
    #
    #         self.set_deep_supervision_enabled(True)
    #         compute_gaussian.cache_clear()

    def perform_actual_validation(self, save_probabilities: bool = False):
            for num_clicks in [0,3,7,10]:

                if num_clicks == 0:
                    print("performing val without clicks")
                    use_clicks = False
                else:
                    print("performing val with clicks")
                    use_clicks = True

                self.set_deep_supervision_enabled(False)
                self.network.eval()

                if self.is_ddp and self.batch_size == 1 and self.enable_deep_supervision and self._do_i_compile():
                    self.print_to_log_file("WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                                           "encounter crashes in validation then this is because torch.compile forgets "
                                           "to trigger a recompilation of the model with deep supervision disabled. "
                                           "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                                           "validation with --val (exactly the same as before) and then it will work. "
                                           "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                                           "forward pass (where compile is triggered) already has deep supervision disabled. "
                                           "This is exactly what we need in perform_actual_validation")

                predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                            perform_everything_on_device=True, device=self.device, verbose=False,
                                            verbose_preprocessing=False, allow_tqdm=False)
                predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                                self.dataset_json, self.__class__.__name__,
                                                self.inference_allowed_mirroring_axes)

                validation_output_folder = join(self.output_folder, 'validation_challenge')

                maybe_mkdir_p(validation_output_folder)
                # del self.network, predictor
                # continue
                with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
                    worker_list = [i for i in segmentation_export_pool._pool]



                    # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
                    # the validation keys across the workers.
                    _, val_keys = self.do_split()
                    if self.is_ddp:
                        last_barrier_at_idx = len(val_keys) // dist.get_world_size() - 1

                        val_keys = val_keys[self.local_rank:: dist.get_world_size()]
                        # we cannot just have barriers all over the place because the number of keys each GPU receives can be
                        # different

                    dataset_val = nnUNetDatasetBlosc2(self.preprocessed_dataset_folder, val_keys,
                                                      folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
                                                      )

                    results = []

                    for i, k in enumerate(dataset_val.get_dataset_identifiers()):
                        proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                   allowed_num_queued=2)
                        while not proceed:
                            sleep(0.1)
                            proceed = not check_workers_alive_and_busy(segmentation_export_pool, worker_list, results,
                                                                       allowed_num_queued=2)
                        maybe_mkdir_p(join(validation_output_folder, k))
                        self.print_to_log_file(f"predicting {k}")
                        data, seg, seg_org, properties, clicks = dataset_val.load_case_with_clicks(k)
                        shape = data.shape[1:]



                        if use_clicks:
                            # select num_clicks random clicks
                            selected_clicks = {'background': [], 'lesion': []}
                            if clicks is not None and len(clicks["lesion"]) >= num_clicks:
                                selected_clicks["lesion"] = clicks["lesion"][:num_clicks]
                            if clicks is not None and len(clicks["background"]) >= num_clicks:
                                selected_clicks["background"] = clicks["background"][:num_clicks]

                            clicks = restructure_clicks(selected_clicks)



                            pos_clicks, neg_clicks = sparse_to_dense_point_nnInteractive(clicks["points"], shape,properties,
                                                                                 sigma=self.point_width)

                        else:
                            pos_clicks = np.zeros(shape, dtype=np.float32)
                            neg_clicks = np.zeros(shape, dtype=np.float32)


                        clicks_stacked = np.vstack((np.expand_dims(pos_clicks, axis=0), np.expand_dims(neg_clicks, axis=0)))
                        clicks_stacked = torch.from_numpy(clicks_stacked).float()
                        data = torch.from_numpy(np.asarray(data)).float()
                        data = torch.cat((data, clicks_stacked), dim=0)

                        # import napari
                        # viewer = napari.Viewer()
                        # viewer.add_image(data[0].numpy(), name='CT')
                        # viewer.add_image(data[1], name='pos')
                        # viewer.add_image(data[2], name='neg')
                        # napari.run()

                        if self.is_cascaded:
                            data = np.vstack(
                                (data, convert_labelmap_to_one_hot(seg[-1], self.label_manager.foreground_labels,
                                                                   output_dtype=data.dtype)))
                        with warnings.catch_warnings():
                            # ignore 'The given NumPy array is not writable' warning
                            warnings.simplefilter("ignore")
                            if type(data) is torch.Tensor:
                                pass
                            else:
                                data = torch.from_numpy(data)

                        self.print_to_log_file(f'{k}, shape {data.shape}, rank {self.local_rank}')

                        #augment filename given num clicks
                        k_clicks = f"{k}_{num_clicks}"
                        output_filename_truncated = join(validation_output_folder,k, k_clicks)

                        prediction = predictor.predict_sliding_window_return_logits(data)
                        prediction = prediction.cpu()

                        # this needs to go into background processes
                        results.append(
                            segmentation_export_pool.starmap_async(
                                export_prediction_from_logits, (
                                    (prediction, properties, self.configuration_manager, self.plans_manager,
                                     self.dataset_json, output_filename_truncated, save_probabilities),
                                )
                            )
                        )
                        # for debug purposes
                        # export_prediction(prediction_for_export, properties, self.configuration, self.plans, self.dataset_json,
                        #              output_filename_truncated, save_probabilities)

                        # if we don't barrier from time to time we will get nccl timeouts for large datasets. Yuck.
                        if self.is_ddp and i < last_barrier_at_idx and (i + 1) % 20 == 0:
                            dist.barrier()

                    _ = [r.get() for r in results]

                if self.is_ddp:
                    dist.barrier()

            if self.local_rank == 0:

                # predicted_masks_path = sys.argv[2]
                groundtruth_masks_path = join(self.preprocessed_dataset_folder_base, 'gt_segmentations')
                metrics_csv_path = join(self.output_folder, 'metrics_csv')

                os.makedirs(metrics_csv_path, exist_ok=True)

                cases_csv_path = os.path.join(metrics_csv_path, "per_case_metrics.csv")
                full_csv_path = os.path.join(metrics_csv_path, "metrics.csv")

                predicted_cases = os.listdir(validation_output_folder)
                print(f"{len(predicted_cases)} predicted cases!")


                orders = [os.path.splitext(os.path.basename(f))[0] for f in predicted_cases]

                with open(cases_csv_path, mode='w', newline='') as writer_file:
                    writer = csv.writer(writer_file)

                    writer.writerow(
                        ["Image_gt", "Image_pm", "DSC@AUC", "ASD@AUC", "MSD@AUC", "DSC@Final", "ASD@Final",
                         "MSD@Final"])  # write header row

                    for i, predicted_case in enumerate(os.listdir(validation_output_folder)):
                        image_hash = orders[i]
                        predicted_case = os.path.join(validation_output_folder, predicted_case)
                        prediction_paths = [
                            os.path.join(predicted_case, f)
                            for f in os.listdir(predicted_case)
                            if f.endswith('.nii.gz')
                        ]

                        output_dir = os.path.join(metrics_csv_path, 'val_metrics')

                        # Bundle paths into a list of tuples
                        input_data = [(path, groundtruth_masks_path, output_dir) for path in
                                      sorted(prediction_paths)]

                        for case in tqdm(input_data, desc=f'Evaluating case {os.path.basename(predicted_case)}'):
                            process_case(case)

                        grouped_jsons = group_jsons_by_prefix(output_dir, os.path.basename(predicted_case))
                        output_path = output_dir.replace('val_metrics', 'interactive_metrics')
                        os.makedirs(output_path, exist_ok=True)
                        print(f'\n  Case: {os.path.basename(predicted_case)} ({len(grouped_jsons)} files)')
                        # assert len(grouped_jsons) == 11
                        agg = aggregate_group_metrics(grouped_jsons)
                        output_json = os.path.join(output_path, f'{os.path.basename(predicted_case)}.json')
                        with open(output_json, 'w') as f:
                            json.dump(agg, f, indent=2)
                        gt_name = f"{image_hash}"
                        pm_name = f"pred-{image_hash}"
                        # Iterate through each image name and write the corresponding dice scores
                        writer.writerow(
                            [gt_name, pm_name, agg['dice']['AUC'], agg['assd']['AUC'], agg['msd']['AUC'],
                             agg['dice']['Final'], agg['assd']['Final'], agg['msd']['Final']])
                        print(f"[✓] Saved: {output_json}")

                final_metric_jsons = [os.path.join(output_path, f) for f in os.listdir(output_path)]
                assert len(final_metric_jsons) == len(predicted_cases)
                mean_metrics = {
                    "dice": {
                        "AUC": 0,
                        "Final": 0,
                    },
                    "assd": {
                        "AUC": 0,
                        "Final": 0
                    },
                    "msd": {
                        "AUC": 0,
                        "Final": 0
                    }
                }
                for final_metric_json in final_metric_jsons:
                    with open(final_metric_json, 'r') as f:
                        final_metric = json.load(f)
                        for key in mean_metrics:
                            for subkey in mean_metrics[key]:
                                mean_metrics[key][subkey] += final_metric[key][subkey]

                # Now average
                num_files = len(final_metric_jsons)
                for key in mean_metrics:
                    for subkey in mean_metrics[key]:
                        mean_metrics[key][subkey] /= num_files

                final_json = os.path.join(metrics_csv_path, 'final_interactive_metrics.json')
                with open(final_json, 'w') as f:
                    json.dump(mean_metrics, f, indent=2)

                # Open the CSV file for writing
                with open(full_csv_path, mode='w', newline='') as file:
                    # Create a CSV writer object
                    writer = csv.writer(file)
                    # Write the headers
                    writer.writerow(['Metric', 'Value'])
                    # Write the metrics
                    writer.writerow(
                        [f"DSC@AUC", mean_metrics['dice']['AUC']])  # Writing the value directly without formatting
                    writer.writerow(
                        [f"ASD@AUC", mean_metrics['assd']['AUC']])  # Writing the value directly without formatting
                    writer.writerow(
                        [f"MSD@AUC", mean_metrics['msd']['AUC']])  # Writing the value directly without formatting
                    writer.writerow([f"DSC@Final", mean_metrics['dice'][
                        'Final']])  # Writing the value directly without formatting
                    writer.writerow([f"ASD@Final", mean_metrics['assd'][
                        'Final']])  # Writing the value directly without formatting
                    writer.writerow([f"MSD@Final",
                                     mean_metrics['msd']['Final']])  # Writing the value directly without formatting

                print(f"[✓] Saved: {final_json}")

            self.set_deep_supervision_enabled(True)
            compute_gaussian.cache_clear()

class trialsTrainer1ep(trialsTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False

class trialsTrainerClickGenPW3(trialsTrainerClickGen):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 3

class trialsTrainerClickGenPW2(trialsTrainerClickGen):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 2

class trialsTrainerClickGenRemLastClass(trialsTrainerClickGen):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 2
        self.dataset_class = nnUNetDatasetHelperSeg

    def get_dataloaders(self):

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicksGeneratedHelper(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)
        dl_val = nnUNetDataLoaderClicksGeneratedHelper(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=min(max(6, allowed_num_processes // 2),10), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # def train_step(self, batch: dict) -> dict:

class trialsTrainerClickGenAdvanced(trialsTrainerClickGen):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 2
        self.dataset_class = nnUNetDatasetHelperSeg

    def get_dataloaders(self):

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicksGeneratedAdvancedGeneration(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)
        dl_val = nnUNetDataLoaderClicksGeneratedAdvancedGeneration(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=min(max(6, allowed_num_processes // 2),10), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


class trialsTrainerClickGenPointScheduling(trialsTrainerClickGenAdvanced):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 2
        self.dataset_class = nnUNetDatasetHelperSeg
        self.standard_click_simulation_probability = 0.8
        self.increase_every = None
        self.precomputed_point =  build_point(tuple((self.point_width,self.point_width,self.point_width)), use_distance_transform=True, binarize=False).to(self.device)
        self.sampling_alpha = 0.2
        # self.num_iterations_per_epoch = 2
        # self.num_val_iterations_per_epoch = 2

    def get_dataloaders(self):

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicksGeneratedNoPlace(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width, standard_click_simulation_probability=self.standard_click_simulation_probability, sampling_alpha=self.sampling_alpha)
        dl_val = nnUNetDataLoaderClicksGeneratedNoPlace(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width, standard_click_simulation_probability=self.standard_click_simulation_probability, sampling_alpha=self.sampling_alpha)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=min(max(6, allowed_num_processes // 2),10), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val


    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        interactions = batch['interactions']

        data = data.to(self.device, non_blocking=True)
        all_data = []
        for b in range(len(data)):
            pos_clicks, neg_clicks = place_precomputed_clicks(interactions[b], self.precomputed_point, data.shape[2:],
                                                              self.device)
            all_data.append(torch.cat((data[b], pos_clicks, neg_clicks), dim=0))

        data = torch.stack(all_data, dim=0)
        if isinstance(target, list):
            target = [(i[:, :1].to(self.device, non_blocking=True), i[:, 1:].to(self.device, non_blocking=True)) for i
                      in target]
            target, target_organs = zip(*target)
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, )
            del data
            l = self.loss(output, target)
            # print('val for key', batch['keys'],l.item())
            # if "venous_221" in batch['keys']:
            #     from batchviewer import view_batch
            #     view_batch(data[1], target[1], torch.softmax(output[1], dim=0))
            # # import IPython;IPython.embed()
            # from batchviewer import view_batch
            # view_batch(data[1], target[1], torch.softmax(output[1], dim=0))

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        interactions = batch['interactions']
        interactions = select_interactions_based_on_epochs(interactions=interactions, current_epoch=self.current_epoch,num_epochs=self.num_epochs, increase_every=self.increase_every)

        data = data.to(self.device, non_blocking=True)
        all_data = []
        for b in range(len(data)):

            pos_clicks , neg_clicks = place_precomputed_clicks(interactions[b], self.precomputed_point, data.shape[2:], self.device)
            all_data.append(torch.cat((data[b], pos_clicks, neg_clicks), dim=0))

        data = torch.stack(all_data, dim=0)

        if isinstance(target, list):
            target = [(i[:, :1].to(self.device, non_blocking=True), i[:, 1:].to(self.device, non_blocking=True)) for i
                      in target]
            target, target_organs = zip(*target)

        else:
            # raise NotImplementedError()
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data, )

            del data
            l = self.loss(output, target)


        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}



class trialsTrainerClickGenPointSchedulingPW1(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 1
        self.precomputed_point = build_point(tuple((self.point_width, self.point_width, self.point_width)),
                                             use_distance_transform=True, binarize=False).to(self.device)

class trialsTrainerClickGenPointSchedulingPW3(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.point_width = 3
        self.precomputed_point = build_point(tuple((self.point_width, self.point_width, self.point_width)),
                                             use_distance_transform=True, binarize=False).to(self.device)

class trialsTrainerClickGenPointSchedulingAdvancedClickGen60(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.standard_click_simulation_probability = 0.4


class trialsTrainerClickGenPointSchedulingAdvancedClickGen40(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.standard_click_simulation_probability = 0.6

class trialsTrainerClickGenPointSchedulingIncreseEvery100(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.increase_every = 100  # increase every 100 epochs

class trialsTrainerClickGenPointSchedulingIncreseEvery50(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.increase_every = 50  # increase every 50 epochs

class trialsTrainerClickGenPointSchedulingSampling01(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.sampling_alpha = 0.1  # sampling alpha 0.1

class trialsTrainerClickGenPointSchedulingSampling04(trialsTrainerClickGenPointScheduling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.sampling_alpha = 0.4  # sampling alpha 0.1

class trialsTrainerDebug(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)


    def get_dataloaders(self):

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        # tr_transforms = self.get_training_transforms(
        #     patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
        #     use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
        #     is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
        #     regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
        #     ignore_label=self.label_manager.ignore_label)
        #
        # # validation pipeline
        # val_transforms = self.get_validation_transforms(deep_supervision_scales,
        #                                                 is_cascaded=self.is_cascaded,
        #                                                 foreground_labels=self.label_manager.foreground_labels,
        #                                                 regions=self.label_manager.foreground_regions if
        #                                                 self.label_manager.has_regions else None,
        #                                                 ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoaderClicksGeneratedDebug(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                       probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)
        dl_val = nnUNetDataLoaderClicksGeneratedDebug(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                        probabilistic_oversampling=self.probabilistic_oversampling, point_width=self.point_width)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=min(max(6, allowed_num_processes // 2),10), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        architecture_class_name = "nnunetv2.architecture.ResidualEncoderUNetPoints.ResidualEncoderUNetPoints"

        return nnUNetTrainer.build_network_architecture(architecture_class_name,
                                                        arch_init_kwargs,
                                                        arch_init_kwargs_req_import,
                                                        num_input_channels +2 ,  # positive / negative clicks
                                                        num_output_channels, enable_deep_supervision)



class trialsTrainerClickGenOnlyTumorLRWarmup(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )

        warmup_epochs = 50
        total_epochs = self.num_epochs  # 1000

        # After creating optimizer
        for group in optimizer.param_groups:
            group['initial_lr'] = self.initial_lr  # hard-code base

        # Warmup: linear from 1e-3 * initial_lr to initial_lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Poly decay: from initial_lr → 0 across the remaining epochs
        poly_scheduler = PytorchCompliantPolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            max_steps=total_epochs - warmup_epochs
        )

        # Chain them together
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, poly_scheduler],
            milestones=[warmup_epochs]
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

class trialsTrainerClickGenOnlyTumorSlowWarmup(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            momentum=0.99,
            nesterov=True
        )

        warmup_epochs = 50
        total_epochs = self.num_epochs  # 1000

        for group in optimizer.param_groups:
            group['initial_lr'] = self.initial_lr  # hard-code base
        # Warmup: linear from 1e-3 * initial_lr to initial_lr


        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        # Poly decay: from initial_lr → 0 across the remaining epochs
        poly_scheduler = PytorchCompliantPolyLRScheduler(
            optimizer,
            initial_lr=self.initial_lr,
            max_steps=total_epochs - warmup_epochs
        )

        # Chain them together
        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, poly_scheduler],
            milestones=[warmup_epochs]
        )
        return optimizer, lr_scheduler

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step()
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


class trialsTrainerClickGenOnlyTumorOversample05(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.oversample_foreground_percent = 0.5

class trialsTrainerClickGenOnlyTumorFocalTverskyLoss(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = FocalTversky_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class trialsTrainerClickGenOnlyTumorFocalTverskyLossPaperBest(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.num_epochs = 1000
        # self.num_iterations_per_epoch = 5

    def _build_loss(self):
        # set smooth to 0
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss({},
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, 'smooth': 0, 'ddp': self.is_ddp},
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss)
        else:
            loss = FocalTversky_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                                   'smooth': 0, 'do_bg': False, 'ddp': self.is_ddp, 'gamma': 1.3, 'alpha': 0.7, 'beta': 0.3}, {}, weight_ce=1, weight_dice=1,
                                  ignore_label=self.label_manager.ignore_label,
                                  dice_class=MemoryEfficientSoftDiceLoss)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

class trialsTrainerClickGenOnlyTumorOversample05TverskyLoss(trialsTrainerClickGenOnlyTumorFocalTverskyLossPaperBest):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.oversample_foreground_percent = 0.5


class trialsTrainerClickGenOnlyTumorOversample95(trialsTrainerClickGenRemLastClass):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.oversample_foreground_percent = 0.95


class trialsTrainerClickGenOnlyTumorOversample95TverskyLoss(trialsTrainerClickGenOnlyTumorFocalTverskyLossPaperBest):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.oversample_foreground_percent = 0.95

class trialsTrainerClickGenOnlyTumorOversample1TverskyLoss(
    trialsTrainerClickGenOnlyTumorFocalTverskyLossPaperBest):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.oversample_foreground_percent = 1