from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pywt
import torch

from .color_utils import rgb_to_ycbcr, ycbcr_to_rgb
from .dwt_utils import dwt_decompose, wpd_decompose, wpd_reconstruct
from .pca_utils import FrequencyStats, get_bands_for_dataset


@dataclass
class FrequencyParams:
    stats: FrequencyStats
    dataset_name: str
    wavelet: str
    freq_repr: str = "wavelet"


class DWTTagger:
    """
    Apply WPTI trigger on Y channel using WPD (or DWT if configured).
    """

    def __init__(
        self,
        params: FrequencyParams,
        beta: float,
        beta_mode: str = "fixed",
        w_mode: str = "fixed",
        w_topk: int = 3,
        w_seed: int | None = None,
        mask_ratio: float = 0.0,
        mask_seed: int | None = None,
    ):
        self.params = params
        self.beta = float(beta)
        self.beta_mode = (beta_mode or "fixed").lower()
        self.w_mode = (w_mode or "fixed").lower()
        self.w_topk = int(w_topk)
        if self.w_topk <= 0:
            self.w_topk = 1
        self.w_generator = None
        if w_seed is not None:
            self.w_generator = torch.Generator()
            self.w_generator.manual_seed(int(w_seed))
        self.mask_ratio = float(mask_ratio)
        if self.mask_ratio < 0.0 or self.mask_ratio > 1.0:
            raise ValueError(f"mask_ratio must be in [0,1], got {self.mask_ratio}")
        self.mask_generator = None
        if mask_seed is not None:
            self.mask_generator = torch.Generator()
            self.mask_generator.manual_seed(int(mask_seed))
        self.w = torch.from_numpy(params.stats.w.astype(np.float64))
        self.eigvecs = torch.from_numpy(params.stats.eigvecs.astype(np.float64))
        self.freq_std = torch.from_numpy(params.stats.freq_std.astype(np.float64))
        self.mu = torch.from_numpy(params.stats.mu.astype(np.float64))
        self.level = getattr(params.stats, "level", 1)
        if params.stats.freq_repr == "wavelet":
            # wavelet模式需要 bands；若未指定则按数据集默认生成
            self.bands = params.stats.bands or get_bands_for_dataset(params.dataset_name, level=self.level)
        else:
            # wpd 模式不依赖 bands，保持为空即可
            self.bands = params.stats.bands
        self.wpd_nodes = params.stats.wpd_nodes
        self.wpd_mode = params.stats.wpd_mode or "symmetric"
        self.wavelet = params.wavelet

    def _vectorize_channel(self, c_coeff: Tuple) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
        band_map = {}
        ll = c_coeff[0]
        details = c_coeff[1:]
        levels = len(details)
        for i, (lh, hl, hh) in enumerate(details, start=1):
            level = levels - i + 1
            band_map[f"LH{level}"] = torch.from_numpy(lh)
            band_map[f"HL{level}"] = torch.from_numpy(hl)
            band_map[f"HH{level}"] = torch.from_numpy(hh)

        slices = []
        flats = []
        for b in self.bands:
            if b not in band_map:
                raise ValueError(f"Band {b} not found. Available: {list(band_map.keys())}")
            sb = band_map[b].to(torch.float64)
            slices.append((b, sb))
            flats.append(sb.reshape(-1))
        return torch.cat(flats, dim=0), slices

    def _reshape_back(self, vector: torch.Tensor, slices: List[Tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        out = {}
        offset = 0
        for name, sb in slices:
            size = sb.numel()
            out[name] = vector[offset : offset + size].reshape(sb.shape)
            offset += size
        return out

    def _resolve_beta(self, vector: torch.Tensor) -> torch.Tensor:
        if self.beta_mode == "fixed":
            return torch.tensor(self.beta, dtype=vector.dtype, device=vector.device)
        if self.beta_mode in ("per_sample_minvar_std", "per_sample_minvar"):
            vector_centered = vector - self.mu.to(vector.device)
            weighted = vector_centered * self.w.to(vector.device)
            std = torch.std(weighted, unbiased=False)
            return std * self.beta
        raise ValueError(f"Unsupported beta_mode: {self.beta_mode}")

    def _resolve_w(self, device: torch.device) -> torch.Tensor:
        if self.w_mode in ("fixed", "fixed_w"):
            return self.w.to(device)
        if self.w_mode in ("minvar_topk_random", "minvar_k_random"):
            k = min(self.w_topk, int(self.eigvecs.shape[1]))
            idx = int(torch.randint(0, k, (1,), generator=self.w_generator).item())
            w = self.eigvecs[:, idx].to(device)
            return w / (torch.linalg.norm(w) + 1e-12)
        if self.w_mode in ("minvar_topk_mean", "minvar_k_mean", "minvar_topk_avg", "minvar_k_avg"):
            k = min(self.w_topk, int(self.eigvecs.shape[1]))
            w = torch.mean(self.eigvecs[:, :k].to(device), dim=1)
            return w / (torch.linalg.norm(w) + 1e-12)
        raise ValueError(f"Unsupported w_mode: {self.w_mode}")

    def _mask_w(self, w: torch.Tensor) -> torch.Tensor:
        if self.mask_ratio <= 0.0:
            return w
        mask = torch.rand(
            w.shape,
            device=w.device,
            dtype=torch.float64,
            generator=self.mask_generator,
        ) > self.mask_ratio
        return w * mask.to(torch.float64)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply trigger to a single image tensor in [0,1].
        """
        img = torch.clamp(img, 0.0, 1.0).to(torch.float32)
        ycbcr = rgb_to_ycbcr(img)
        y_channel = ycbcr[0:1]
        use_wpd = (self.params.freq_repr == "wpd") or (self.wpd_nodes is not None)

        if use_wpd:
            if not self.wpd_nodes:
                raise ValueError("wpd_nodes is required for WPD trigger application.")
            coeffs = wpd_decompose(y_channel, wavelet=self.wavelet, level=self.level, mode=self.wpd_mode)
            shapes = []
            flats = []
            for node in self.wpd_nodes:
                if node not in coeffs:
                    raise ValueError(f"WPD node {node} not found. Available: {list(coeffs.keys())}")
                c = coeffs[node]
                shapes.append((node, c.shape))
                flats.append(c.reshape(-1))
            vector = torch.cat(flats, dim=0).to(torch.float64)
            if vector.numel() != self.w.numel():
                raise ValueError("Vector length mismatch, please recompute PCA stats.")
            beta = self._resolve_beta(vector)
            w_eff = self._mask_w(self._resolve_w(vector.device))
            vector_new = vector + beta * w_eff

            new_leaves = {k: v.clone() for k, v in coeffs.items()}
            offset = 0
            for node, shape in shapes:
                size = int(torch.tensor(shape).prod().item())
                slice_v = vector_new[offset : offset + size].reshape(shape)
                offset += size
                new_leaves[node] = slice_v
            rec_y = wpd_reconstruct(
                new_leaves,
                wavelet=self.wavelet,
                level=self.level,
                mode=self.wpd_mode,
                target_shape=(y_channel.shape[1], y_channel.shape[2]),
            ).to(torch.float32)
        else:
            coeffs = dwt_decompose(y_channel, wavelet=self.wavelet, level=self.level)

            per_channel_vectors = []
            per_channel_slices = []
            for c_coeff in coeffs:
                vec, slices = self._vectorize_channel(c_coeff)
                per_channel_vectors.append(vec)
                per_channel_slices.append(slices)

            vector = torch.cat(per_channel_vectors, dim=0).to(torch.float64)
            if vector.numel() != self.w.numel():
                raise ValueError("Vector length mismatch, please recompute PCA stats.")

            beta = self._resolve_beta(vector)
            w_eff = self._mask_w(self._resolve_w(vector.device))
            vector_new = vector + beta * w_eff

            rebuilt_coeffs = []
            offset = 0
            for slices in per_channel_slices:
                size = sum(s[1].numel() for s in slices)
                v_slice = vector_new[offset : offset + size]
                offset += size
                band_map = self._reshape_back(v_slice, slices)

                coeff_list = list(coeffs[0])
                ll_top = coeff_list[0]
                details = list(coeff_list[1:])
                levels = len(details)
                new_details = []
                for i, (lh_np, hl_np, hh_np) in enumerate(details, start=1):
                    level = levels - i + 1
                    lh = torch.from_numpy(lh_np)
                    hl = torch.from_numpy(hl_np)
                    hh = torch.from_numpy(hh_np)
                    if f"LH{level}" in band_map:
                        lh = band_map[f"LH{level}"]
                    if f"HL{level}" in band_map:
                        hl = band_map[f"HL{level}"]
                    if f"HH{level}" in band_map:
                        hh = band_map[f"HH{level}"]
                    new_details.append(
                        (
                            lh.cpu().numpy() if isinstance(lh, torch.Tensor) else lh,
                            hl.cpu().numpy() if isinstance(hl, torch.Tensor) else hl,
                            hh.cpu().numpy() if isinstance(hh, torch.Tensor) else hh,
                        )
                    )

                rebuilt_coeffs.append(tuple([ll_top] + new_details))

            rec_channels = []
            for c_coeff in rebuilt_coeffs:
                rec = pywt.waverec2(c_coeff, wavelet=self.wavelet)
                rec_channels.append(torch.from_numpy(rec).to(torch.float32))
            rec_y = rec_channels[0]
        ycbcr = ycbcr.clone()
        ycbcr[0] = rec_y
        rec_rgb = ycbcr_to_rgb(ycbcr)
        return torch.clamp(rec_rgb, 0.0, 1.0)
