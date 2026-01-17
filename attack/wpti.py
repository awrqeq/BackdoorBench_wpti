"""
WPTI attack integration for BackdoorBench.
"""

import os
import sys
import yaml

sys.path = ["./"] + sys.path

import argparse
import logging
from copy import deepcopy

import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from attack.prototype import NormalCase
from utils.aggregate_block.bd_attack_generate import bd_attack_label_trans_generate
from utils.backdoor_generate_poison_index import generate_poison_index_from_label_transform
from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2, dataset_wrapper_with_transform
from utils.save_load_attack import save_attack_result
from utils.aggregate_block.model_trainer_generate import generate_cls_model
from utils.aggregate_block.train_settings_generate import argparser_opt_scheduler, argparser_criterion
from utils.trainer_cls import BackdoorModelTrainer
from utils.wpti import (
    FrequencyStats,
    DWTTagger,
    FrequencyParams,
    build_pca_trigger,
    collect_vectors,
    compute_top_wpd_nodes,
    get_bands_for_dataset,
)
from utils.aggregate_block.dataset_and_transform_generate import get_input_shape
from dataset.GTSRB import GTSRB
from dataset.Tiny import TinyImageNet


def add_common_attack_args(parser):
    parser.add_argument("--attack", type=str)
    parser.add_argument("--attack_target", type=int, help="target class in all2one attack")
    parser.add_argument("--attack_label_trans", type=str, help="which type of label modification in backdoor attack")
    parser.add_argument(
        "--clean_label",
        type=int,
        default=0,
        help=(
            "Clean-label setting for all2one: only poison samples whose original label already equals attack_target, "
            "so labels are not changed in training. "
            "Note: poisoning ratio is still defined as #poison / #all samples."
        ),
    )
    parser.add_argument("--pratio", type=float, help="poison rate")
    parser.add_argument(
        "--save_bd_dataset",
        type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
        default=False,
        help="Whether to save poisoned images to disk during preprocessing (default: false, keep in memory).",
    )
    return parser


class WPTI(NormalCase):
    def __init__(self):
        super(WPTI).__init__()

    def set_bd_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = add_common_attack_args(parser)
        parser.add_argument(
            "--save_attack_result",
            type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            default=True,
            help="Whether to save attack_result.pt (default: true).",
        )
        parser.add_argument("--wpti_stats_path", type=str, help="path to WPTI stats (pkl)")
        parser.add_argument("--beta", type=float, help="trigger strength")
        parser.add_argument(
            "--wpti_beta_mode",
            type=str,
            default="fixed",
            help="Trigger strength mode: fixed or per_sample_minvar_std.",
        )
        parser.add_argument(
            "--wpti_mask_train",
            type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            default=False,
            help="Whether to randomly mask part of trigger during training (default: false).",
        )
        parser.add_argument(
            "--wpti_mask_ratio",
            type=float,
            default=0.0,
            help="Ratio of trigger coefficients to mask when --wpti_mask_train is enabled.",
        )
        parser.add_argument(
            "--wpti_vhd_concat",
            type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            default=False,
            help="Use V/H/D subbands (LH1, HL1, HH1) concatenation for PCA trigger.",
        )
        parser.add_argument(
            "--wpti_minvar_topk_train_random",
            type=int,
            default=0,
            help="If >0, train uses random direction among top-k minimum-variance PCA directions; test uses their mean.",
        )
        parser.add_argument(
            "--pca_direction_mode",
            type=str,
            default=None,
            help="Override PCA direction mode (e.g., minvar, maxvar, random, minvar4mean).",
        )
        parser.add_argument(
            "--use_cached_stats",
            type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            default=False,
            help="Whether to load stats from --wpti_stats_path (default: false, recompute each run).",
        )
        parser.add_argument(
            "--save_stats",
            type=lambda x: str(x).lower() in ("1", "true", "yes", "y"),
            default=False,
            help="Whether to save computed stats to --wpti_stats_path.",
        )
        parser.add_argument(
            "--bd_yaml_path",
            type=str,
            default="./config/attack/wpti/default.yaml",
            help="WPTI yaml with default args",
        )
        return parser

    def add_bd_yaml_to_args(self, args):
        with open(args.bd_yaml_path, "r") as f:
            mix_defaults = yaml.safe_load(f)
        mix_defaults.update({k: v for k, v in args.__dict__.items() if v is not None})
        if mix_defaults.get("pca_direction_mode") is not None:
            if "pca" not in mix_defaults or mix_defaults["pca"] is None:
                mix_defaults["pca"] = {}
            mix_defaults["pca"]["direction_mode"] = mix_defaults["pca_direction_mode"]
        args.__dict__ = mix_defaults

    def stage1_non_training_data_prepare(self):
        logging.info("stage1 start")
        assert "args" in self.__dict__
        args = self.args

        (
            train_dataset_without_transform,
            train_img_transform,
            train_label_transform,
            test_dataset_without_transform,
            test_img_transform,
            test_label_transform,
            clean_train_dataset_with_transform,
            clean_train_dataset_targets,
            clean_test_dataset_with_transform,
            clean_test_dataset_targets,
        ) = self.benign_prepare()

        stats = self._get_or_compute_stats(args)
        params = FrequencyParams(
            stats=stats,
            dataset_name=args.dataset,
            wavelet=stats.wavelet,
            freq_repr=stats.freq_repr,
        )
        beta_mode = getattr(args, "wpti_beta_mode", "fixed")
        train_mask = bool(getattr(args, "wpti_mask_train", False))
        mask_ratio = float(getattr(args, "wpti_mask_ratio", 0.0))
        topk = int(getattr(args, "wpti_minvar_topk_train_random", 0) or 0)
        if topk > 0:
            train_w_mode = "minvar_topk_random"
            test_w_mode = "minvar_topk_mean"
        else:
            train_w_mode = "fixed"
            test_w_mode = "fixed"
        tagger_train = DWTTagger(
            params,
            beta=float(args.beta),
            beta_mode=beta_mode,
            w_mode=train_w_mode,
            w_topk=topk if topk > 0 else 3,
            w_seed=getattr(args, "random_seed", None),
            mask_ratio=mask_ratio if train_mask else 0.0,
            mask_seed=getattr(args, "random_seed", None),
        )
        tagger_test = DWTTagger(
            params,
            beta=float(args.beta),
            beta_mode=beta_mode,
            w_mode=test_w_mode,
            w_topk=topk if topk > 0 else 3,
            mask_ratio=0.0,
        )
        to_tensor = ToTensor()
        to_pil = ToPILImage()
        resize = None
        # 预先统一尺寸以匹配 PCA 统计使用的输入大小
        if args.input_height and args.input_width:
            from torchvision import transforms
            resize = transforms.Resize((args.input_height, args.input_width))

        def wpti_image_transform_train(img, target=None, image_serial_id=None):
            if resize is not None:
                img = resize(img)
            img_t = to_tensor(img)
            marked = tagger_train.apply(img_t)
            return to_pil(marked)

        def wpti_image_transform_test(img, target=None, image_serial_id=None):
            if resize is not None:
                img = resize(img)
            img_t = to_tensor(img)
            marked = tagger_test.apply(img_t)
            return to_pil(marked)

        train_bd_img_transform = wpti_image_transform_train
        test_bd_img_transform = wpti_image_transform_test

        bd_label_transform = bd_attack_label_trans_generate(args)

        train_poison_index = generate_poison_index_from_label_transform(
            clean_train_dataset_targets,
            label_transform=bd_label_transform,
            train=True,
            pratio=args.pratio if "pratio" in args.__dict__ else None,
            p_num=args.p_num if "p_num" in args.__dict__ else None,
            clean_label=bool(getattr(args, "clean_label", 0)),
        )

        torch.save(train_poison_index, args.save_path + "/train_poison_index_list.pickle")

        bd_train_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(train_dataset_without_transform),
            poison_indicator=train_poison_index,
            bd_image_pre_transform=train_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=(f"{args.save_path}/bd_train_dataset" if getattr(args, "save_bd_dataset", False) else None),
        )

        bd_train_dataset_with_transform = dataset_wrapper_with_transform(
            bd_train_dataset,
            train_img_transform,
            train_label_transform,
        )

        test_poison_index = generate_poison_index_from_label_transform(
            clean_test_dataset_targets,
            label_transform=bd_label_transform,
            train=False,
        )

        bd_test_dataset = prepro_cls_DatasetBD_v2(
            deepcopy(test_dataset_without_transform),
            poison_indicator=test_poison_index,
            bd_image_pre_transform=test_bd_img_transform,
            bd_label_pre_transform=bd_label_transform,
            save_folder_path=(f"{args.save_path}/bd_test_dataset" if getattr(args, "save_bd_dataset", False) else None),
        )

        bd_test_dataset.subset(np.where(test_poison_index == 1)[0])

        bd_test_dataset_with_transform = dataset_wrapper_with_transform(
            bd_test_dataset,
            test_img_transform,
            test_label_transform,
        )

        self.stage1_results = (
            clean_train_dataset_with_transform,
            clean_test_dataset_with_transform,
            bd_train_dataset_with_transform,
            bd_test_dataset_with_transform,
        )

    @staticmethod
    def _resolve_dataset_root(dataset_path: str, dataset_name: str) -> str:
        # 如果 dataset_path 已经是一个带 train/val 的根目录（如 imagenette2-320），直接返回
        if os.path.isdir(os.path.join(dataset_path, "train")) and os.path.isdir(os.path.join(dataset_path, "val")):
            return dataset_path
        base = os.path.basename(os.path.normpath(dataset_path))
        if base == dataset_name:
            return dataset_path
        return os.path.join(dataset_path, dataset_name)

    def _build_pca_loader(self, args) -> DataLoader:
        dataset_name = str(args.dataset).lower()
        dataset_path = self._resolve_dataset_root(getattr(args, "dataset_path", "./data"), dataset_name)
        img_h, img_w, _ = get_input_shape(dataset_name)
        batch_size = int(getattr(args, "batch_size", 128))
        num_workers = int(getattr(args, "num_workers", 4))

        tf = transforms.Compose([transforms.Resize((img_h, img_w)), transforms.ToTensor()])
        if dataset_name == "cifar10":
            base_ds = datasets.CIFAR10(root=dataset_path, train=True, download=True, transform=tf)
        elif dataset_name == "cifar100":
            base_ds = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=tf)
        elif dataset_name == "gtsrb":
            base_ds = GTSRB(dataset_path, train=True, transform=tf)
        elif dataset_name == "tiny":
            base_ds = TinyImageNet(dataset_path, split="train", download=True, transform=tf)
        elif dataset_name == "imagenette":
            base_ds = datasets.ImageFolder(root=os.path.join(dataset_path, "train"), transform=tf)
        else:
            raise ValueError(f"Unsupported dataset for WPTI stats: {dataset_name}")

        return DataLoader(
            base_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _compute_stats(self, args) -> FrequencyStats:
        pca_cfg = getattr(args, "pca", {}) or {}
        dataset_name = str(getattr(args, "dataset", "cifar10")).lower()
        target_class = int(getattr(args, "attack_target", 0))
        wavelet = pca_cfg.get("wavelet", "haar")
        wpd_mode = pca_cfg.get("wpd_mode", "symmetric")
        level = int(pca_cfg.get("level", 1))
        decomp = str(pca_cfg.get("decomp", "wpd")).lower()
        top_energy_nodes = int(pca_cfg.get("top_energy_nodes", 1))
        exclude_lowpass = bool(pca_cfg.get("exclude_lowpass_nodes", True))
        energy_select = str(pca_cfg.get("wpd_energy_select", "max"))
        bands_override = pca_cfg.get("bands_override", None)
        wpd_nodes_override = pca_cfg.get("wpd_nodes_override", None)
        max_per_class = int(pca_cfg.get("pca_sample_images", 2000))
        direction_mode = str(pca_cfg.get("direction_mode", "minvar"))
        tail_dim = pca_cfg.get("tail_dim", 1)

        if bool(getattr(args, "wpti_vhd_concat", False)):
            decomp = "wavelet"
            level = 1
            bands_override = ("LH1", "HL1", "HH1")

        loader = self._build_pca_loader(args)

        bands = None
        wpd_nodes = None
        if decomp == "wpd":
            if wpd_nodes_override:
                wpd_nodes = tuple(wpd_nodes_override)
            else:
                wpd_nodes = compute_top_wpd_nodes(
                    loader,
                    wavelet=wavelet,
                    level=level,
                    top_k=top_energy_nodes,
                    max_images=max_per_class,
                    exclude_lowpass=exclude_lowpass,
                    wpd_mode=wpd_mode,
                    energy_select=energy_select,
                )
            logging.info(f"[WPD] {dataset_name} nodes={wpd_nodes}")
        else:
            bands = bands_override if bands_override else None
            if bands is None:
                bands = get_bands_for_dataset(dataset_name, level=level)

        vectors, labels = collect_vectors(
            loader,
            dataset_name=dataset_name,
            bands=bands,
            wavelet=wavelet,
            level=level,
            use_wpd=(decomp == "wpd"),
            wpd_nodes=wpd_nodes,
            wpd_mode=wpd_mode,
            max_images_per_class=max_per_class,
            device="cpu",
            seed=int(getattr(args, "random_seed", 0)),
        )

        stats = build_pca_trigger(
            vectors=vectors,
            labels=labels,
            target_class=target_class,
            tail_dim=tail_dim,
            seed=int(getattr(args, "random_seed", 0)),
            dataset_name=dataset_name,
            wavelet=wavelet,
            bands=bands,
            wpd_nodes=wpd_nodes,
            wpd_mode=wpd_mode if decomp == "wpd" else None,
            level=level,
            freq_repr="wpd" if decomp == "wpd" else "wavelet",
            direction_mode=direction_mode,
        )
        return stats

    def _get_or_compute_stats(self, args) -> FrequencyStats:
        stats_path = getattr(args, "wpti_stats_path", None)
        use_cached = bool(getattr(args, "use_cached_stats", False))
        save_stats = bool(getattr(args, "save_stats", False))

        if use_cached and stats_path and os.path.exists(stats_path):
            return FrequencyStats.load(stats_path)

        stats = self._compute_stats(args)
        if save_stats and stats_path:
            try:
                stats.save(stats_path)
                logging.info(f"Saved WPTI stats to: {stats_path}")
            except Exception as e:
                logging.warning(f"Failed to save WPTI stats to {stats_path}: {e}")
        return stats

    def stage2_training(self):
        logging.info("stage2 start")
        assert "args" in self.__dict__
        args = self.args

        (
            clean_train_dataset_with_transform,
            clean_test_dataset_with_transform,
            bd_train_dataset_with_transform,
            bd_test_dataset_with_transform,
        ) = self.stage1_results

        self.net = generate_cls_model(
            model_name=args.model,
            num_classes=args.num_classes,
            image_size=args.img_size[0],
        )

        self.device = torch.device(
            (
                f"cuda:{[int(i) for i in args.device[5:].split(',')][0]}"
                if "," in args.device
                else args.device
            )
            if torch.cuda.is_available()
            else "cpu"
        )

        if "," in args.device:
            self.net = torch.nn.DataParallel(
                self.net,
                device_ids=[int(i) for i in args.device[5:].split(",")],
            )

        trainer = BackdoorModelTrainer(self.net)
        criterion = argparser_criterion(args)
        optimizer, scheduler = argparser_opt_scheduler(self.net, args)

        from torch.utils.data.dataloader import DataLoader

        trainer.train_with_test_each_epoch_on_mix(
            DataLoader(
                bd_train_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            DataLoader(
                clean_test_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            DataLoader(
                bd_test_dataset_with_transform,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=args.pin_memory,
                num_workers=args.num_workers,
            ),
            args.epochs,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            frequency_save=args.frequency_save,
            save_folder_path=args.save_path,
            save_prefix="attack",
            amp=args.amp,
            prefetch=args.prefetch,
            prefetch_transform_attr_name="ori_image_transform_in_loading",
            non_blocking=args.non_blocking,
        )

        save_attack_result(
            model_name=args.model,
            num_classes=args.num_classes,
            model=trainer.model.cpu().state_dict(),
            data_path=args.dataset_path,
            img_size=args.img_size,
            clean_data=args.dataset,
            bd_train=bd_train_dataset_with_transform,
            bd_test=bd_test_dataset_with_transform,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    attack = WPTI()
    parser = argparse.ArgumentParser(description=sys.argv[0])
    parser = attack.set_args(parser)
    parser = attack.set_bd_args(parser)
    args = parser.parse_args()
    logging.debug("Use bd yaml defaults first.")
    attack.add_bd_yaml_to_args(args)
    attack.add_yaml_to_args(args)
    args = attack.process_args(args)
    attack.prepare(args)
    attack.stage1_non_training_data_prepare()
    attack.stage2_training()
