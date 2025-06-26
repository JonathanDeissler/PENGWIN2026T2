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

#from batchviewer import view_batch
from nnunetv2.training.dataloading.utils import restructure_clicks, sparse_to_dense_point_gauss
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder, evaluate_simple_entry_point
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.misalign import Misalign2
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.data_loader_clicks import nnUNetDataLoaderClicks
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class trialsTrainerBasicTransform(nnUNetTrainer):
    os.environ["STEM"] = "trials"
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False


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

class trialsTrainerBasicTransform1ep(trialsTrainerBasicTransform):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.initial_lr = 1e-3
        self.enable_deep_supervision = False