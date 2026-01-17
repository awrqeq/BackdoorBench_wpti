from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .color_utils import rgb_to_ycbcr
from .dwt_utils import dwt_decompose, extract_subbands, wpd_decompose


@dataclass
class FrequencyStats:
    mu: np.ndarray
    cov: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray
    w: np.ndarray
    freq_std: np.ndarray
    dataset_name: str
    vector_length: int
    k_tail: int
    tail_range: Tuple[int, int] | None
    wavelet: str
    bands: Tuple[str, ...]
    wpd_nodes: Tuple[str, ...] | None
    wpd_mode: str | None
    level: int = 1
    freq_repr: str = "wavelet"
    direction_mode: str = "minvar"

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "mu": self.mu,
            "cov": self.cov,
            "eigvals": self.eigvals,
            "eigvecs": self.eigvecs,
            "w": self.w,
            "freq_std": self.freq_std,
            "dataset_name": self.dataset_name,
            "vector_length": self.vector_length,
            "k_tail": self.k_tail,
            "tail_range": self.tail_range,
            "wavelet": self.wavelet,
            "bands": self.bands,
            "wpd_nodes": self.wpd_nodes,
            "wpd_mode": self.wpd_mode,
            "level": self.level,
            "freq_repr": self.freq_repr,
            "direction_mode": self.direction_mode,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @staticmethod
    def load(path: str | Path) -> "FrequencyStats":
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        return FrequencyStats(
            mu=data["mu"],
            cov=data["cov"],
            eigvals=data["eigvals"],
            eigvecs=data["eigvecs"],
            w=data["w"],
            freq_std=data.get("freq_std", np.ones_like(data["w"])),
            dataset_name=data.get("dataset_name", "cifar10"),
            vector_length=int(data.get("vector_length", len(data["w"]))),
            k_tail=int(data.get("k_tail", 4)),
            tail_range=tuple(data["tail_range"]) if "tail_range" in data else None,
            wavelet=data.get("wavelet", "haar"),
            bands=tuple(data.get("bands", ("LH1", "HL1"))),
            wpd_nodes=tuple(data.get("wpd_nodes")) if data.get("wpd_nodes") is not None else None,
            wpd_mode=data.get("wpd_mode", None),
            level=int(data.get("level", _infer_level_from_bands(data.get("bands")))),
            freq_repr=data.get("freq_repr", "wavelet"),
            direction_mode=data.get("direction_mode", "minvar"),
        )


def _infer_level_from_bands(bands_data) -> int:
    if bands_data is None:
        return 1
    bands = tuple(bands_data)
    for b in bands:
        if "2" in str(b):
            return 2
    return 1


def get_bands_for_dataset(
    name: str, level: int | None = None, override: Tuple[str, ...] | None = None
) -> Tuple[str, ...]:
    if override is not None:
        return tuple(override)
    if level is None:
        level = 1
    name = name.lower()
    if name in ("cifar10", "gtsrb"):
        return ("LH1", "HL1") if level == 1 else ("LH2", "HL2")
    if name == "imagenette":
        return ("LH2", "HL2")
    raise ValueError(f"Unsupported dataset for bands: {name}")


def compute_top_wpd_nodes(
    loader: DataLoader,
    wavelet: str,
    level: int,
    top_k: int = 1,
    max_images: int | None = None,
    exclude_lowpass: bool = True,
    wpd_mode: str = "symmetric",
    energy_select: str = "max",
) -> Tuple[str, ...]:
    """
    Select WPD nodes at the deepest level by average energy.

    - energy_select="max": choose highest-energy nodes (original behavior)
    - energy_select="min": choose lowest-energy nodes
    """
    energy_sum: dict[str, float] = {}
    count = 0
    for images, _ in loader:
        for img in images:
            y_only = rgb_to_ycbcr(img)[0:1]
            coeffs = wpd_decompose(y_only, wavelet=wavelet, level=level, mode=wpd_mode)
            for path, coeff in coeffs.items():
                if exclude_lowpass and path == "a" * level:
                    continue
                e = float((coeff.float() ** 2).mean().item())
                energy_sum[path] = energy_sum.get(path, 0.0) + e
            count += 1
            if max_images is not None and count >= max_images:
                break
        if max_images is not None and count >= max_images:
            break
    if count == 0:
        return tuple()
    energy_avg = {k: v / count for k, v in energy_sum.items()}
    energy_select = (energy_select or "max").lower()
    if energy_select not in ("max", "min"):
        raise ValueError(f"Unsupported energy_select: {energy_select}")
    sorted_nodes = sorted(energy_avg.items(), key=lambda kv: kv[1], reverse=(energy_select == "max"))
    top_nodes = [p for p, _ in sorted_nodes[:top_k]]
    return tuple(top_nodes)


def collect_vectors(
    loader: DataLoader,
    dataset_name: str,
    bands: Tuple[str, ...] | None = None,
    wavelet: str = "haar",
    level: int = 1,
    use_wpd: bool = False,
    wpd_nodes: Tuple[str, ...] | None = None,
    wpd_mode: str = "symmetric",
    max_images_per_class: int = 2000,
    device: torch.device | str = "cpu",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    device = torch.device(device)

    class_to_count = {}
    if hasattr(loader.dataset, "classes"):
        class_to_count = {i: 0 for i in range(len(loader.dataset.classes))}

    collected: List[torch.Tensor] = []
    labels_out: List[int] = []
    total_expected = max_images_per_class * max(1, len(class_to_count) or 10)
    desc = "Collecting WPD vectors" if use_wpd else "Collecting DWT vectors"
    # 当 stdout/stderr 重定向到文件（例如 batch log）时，关闭 tqdm 进度条避免刷屏
    disable_pbar = not sys.stderr.isatty()
    with tqdm(desc=desc, total=total_expected, disable=disable_pbar, dynamic_ncols=True, leave=False) as pbar:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.tolist()
            for img, y in zip(images, labels):
                if y not in class_to_count:
                    class_to_count[y] = 0
                if class_to_count[y] >= max_images_per_class:
                    continue
                y_only = rgb_to_ycbcr(img)[0:1]
                if use_wpd:
                    if wpd_nodes is None:
                        raise ValueError("wpd_nodes must be provided when use_wpd=True")
                    coeffs = wpd_decompose(y_only, wavelet=wavelet, level=level, mode=wpd_mode)
                    flats = []
                    for node in wpd_nodes:
                        if node not in coeffs:
                            raise ValueError(f"WPD node {node} not found (available: {list(coeffs.keys())})")
                        flats.append(coeffs[node].reshape(-1))
                    flat = torch.cat(flats, dim=0)
                else:
                    if bands is None:
                        raise ValueError("bands must be provided for wavelet collection")
                    coeffs = dwt_decompose(y_only, wavelet=wavelet, level=level)
                    subbands = extract_subbands(coeffs, bands=bands)
                    flat = subbands.reshape(-1)
                collected.append(flat.cpu())
                labels_out.append(int(y))
                class_to_count[y] += 1
                pbar.update(1)
                if all(v >= max_images_per_class for v in class_to_count.values()):
                    vectors = torch.stack(collected, dim=0)
                    return vectors.double().numpy(), np.array(labels_out, dtype=np.int64)
    if not collected:
        return np.empty((0, 0), dtype=np.float64), np.array([], dtype=np.int64)
    vectors = torch.stack(collected, dim=0)
    return vectors.double().numpy(), np.array(labels_out, dtype=np.int64)


def build_pca_trigger(
    vectors: np.ndarray,
    labels: np.ndarray,
    target_class: int,
    tail_dim: int = 64,
    seed: int = 42,
    dataset_name: str = "cifar10",
    wavelet: str = "haar",
    bands: Tuple[str, ...] | None = None,
    wpd_nodes: Tuple[str, ...] | None = None,
    wpd_mode: str | None = None,
    level: int = 1,
    freq_repr: str = "wavelet",
    raw_mean: np.ndarray | None = None,
    raw_std: np.ndarray | None = None,
    direction_mode: str = "minvar",
) -> FrequencyStats:
    """
    PCA uses all classes; trigger direction is controlled by direction_mode.
    """
    mu_norm = np.mean(vectors, axis=0)
    centered = vectors - mu_norm
    cov = (centered.T @ centered) / centered.shape[0]
    eigvals, eigvecs = np.linalg.eigh(cov)

    if isinstance(tail_dim, (list, tuple)) and len(tail_dim) >= 2:
        tail_start = int(tail_dim[0])
        tail_end = int(tail_dim[1])
    else:
        tail_start = 0
        tail_end = int(tail_dim)

    direction_mode = (direction_mode or "minvar").lower()
    if direction_mode in ("minvar", "min_var", "minvariance"):
        # 在全空间内选“方差最小”方向：最小特征值对应特征向量
        w = eigvecs[:, 0].astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("maxvar", "max_var", "maxvariance"):
        # 在全空间内选“方差最大”方向：最大特征值对应特征向量
        w = eigvecs[:, -1].astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("minvar4mean", "min_var4_mean", "minvar4", "min_var4"):
        # 最小的4个方差方向取平均作为全局触发器
        k = min(4, eigvecs.shape[1])
        w = np.mean(eigvecs[:, :k], axis=1).astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("random", "rand", "random_full", "rand_full"):
        # 在全空间内随机采样一个单位向量（与 w 维度一致），再归一化
        rng = np.random.default_rng(int(seed))
        w = rng.standard_normal(eigvecs.shape[0]).astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("random_tail", "rand_tail"):
        # 在 tail 子空间内按 seed 随机混合方向：w = U_tail @ a
        rng = np.random.default_rng(int(seed))
        u_tail = eigvecs[:, tail_start:tail_end]
        k_tail = max(1, int(u_tail.shape[1]))
        a = rng.standard_normal(k_tail).astype(np.float64)
        w = (u_tail @ a).astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("midvar", "mid_var", "midvariance", "middle"):
        # 选“中间方差”方向：中位特征值对应特征向量
        mid = int(eigvecs.shape[1] // 2)
        w = eigvecs[:, mid].astype(np.float64)
        w = w / (np.linalg.norm(w) + 1e-12)
    elif direction_mode in ("basis", "basis0", "e0", "std_basis"):
        # 固定基方向：w = (1,0,0,...)
        w = np.zeros((eigvecs.shape[0],), dtype=np.float64)
        w[0] = 1.0
    else:
        raise ValueError(f"Unsupported direction_mode: {direction_mode}")

    if bands is None and freq_repr == "wavelet":
        bands = get_bands_for_dataset(dataset_name, level=level)

    k_tail = max(1, int(tail_end - tail_start))
    stats = FrequencyStats(
        mu=mu_norm if raw_mean is None else raw_mean,
        cov=cov,
        eigvals=eigvals,
        eigvecs=eigvecs,
        w=w,
        freq_std=np.ones_like(w) if raw_std is None else raw_std,
        dataset_name=dataset_name,
        vector_length=int(vectors.shape[1]),
        k_tail=k_tail,
        tail_range=(tail_start, tail_end),
        wavelet=wavelet,
        bands=bands if bands is not None else (),
        wpd_nodes=wpd_nodes,
        wpd_mode=wpd_mode,
        level=int(level),
        freq_repr=freq_repr,
        direction_mode=direction_mode,
    )
    return stats
