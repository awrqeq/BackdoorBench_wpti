from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pywt
import torch


def dwt_decompose(img: torch.Tensor, wavelet: str = "haar", level: int = 2) -> Tuple[tuple, ...]:
    """
    DWT for a single image (C,H,W). Return wavedec2 coeffs per channel.
    """
    assert img.dim() == 3, "expect (C,H,W)"
    coeffs = []
    for c in range(img.shape[0]):
        arr = img[c].cpu().numpy()
        coeff = pywt.wavedec2(arr, wavelet=wavelet, level=level)
        coeffs.append(coeff)
    return tuple(coeffs)


def extract_subbands(coeffs, bands: Tuple[str, ...]) -> torch.Tensor:
    """
    Extract specified subbands from wavedec2 output.
    """
    out = []
    for c_coeff in coeffs:
        details = c_coeff[1:]
        levels = len(details)
        band_map = {}
        for i, (lh, hl, hh) in enumerate(details, start=1):
            level = levels - i + 1
            band_map[f"LH{level}"] = lh
            band_map[f"HL{level}"] = hl
            band_map[f"HH{level}"] = hh
        for b in bands:
            if b not in band_map:
                raise ValueError(f"Band {b} not found in coeffs (available: {list(band_map.keys())})")
            out.append(torch.from_numpy(band_map[b]))
    if not out:
        return torch.empty(0)
    return torch.stack(out, dim=0)


def wpd_decompose(
    img: torch.Tensor, wavelet: str = "haar", level: int = 2, mode: str = "symmetric"
) -> Dict[str, torch.Tensor]:
    """
    Wavelet packet decomposition for a single-channel image (1,H,W).
    """
    assert img.dim() == 3 and img.shape[0] == 1, "expect (1,H,W) single channel"
    arr = img[0].cpu().numpy()
    wp = pywt.WaveletPacket2D(data=arr, wavelet=wavelet, mode=mode, maxlevel=level)
    leaves = wp.get_level(level, order="natural")
    return {node.path: torch.from_numpy(node.data) for node in leaves}


def wpd_reconstruct(
    leaves: Dict[str, torch.Tensor],
    wavelet: str = "haar",
    level: int = 2,
    mode: str = "symmetric",
    target_shape: Tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    Reconstruct a single-channel image from wavelet packet leaves.
    """
    # 关键：如果 data=None，pywt 无法知道原始图像的尺寸/对齐方式，重建出来的能量可能偏到角落。
    # 这里用零初始化并显式指定 target_shape，确保与原图对齐一致。
    if target_shape is not None:
        h, w = target_shape
        wp = pywt.WaveletPacket2D(data=np.zeros((h, w), dtype=np.float32), wavelet=wavelet, mode=mode, maxlevel=level)
    else:
        wp = pywt.WaveletPacket2D(data=None, wavelet=wavelet, mode=mode, maxlevel=level)
    for path, coeff in leaves.items():
        wp[path] = coeff.cpu().numpy()
    rec = wp.reconstruct(update=False)
    rec_t = torch.from_numpy(rec).unsqueeze(0).to(torch.float32)
    if target_shape is not None:
        h, w = target_shape
        # 防御性处理：如果重建尺寸略有偏差，采用居中裁剪而不是左上角裁剪，避免“角落峰值”假象。
        hh, ww = int(rec_t.shape[-2]), int(rec_t.shape[-1])
        if hh != h or ww != w:
            top = max(0, (hh - h) // 2)
            left = max(0, (ww - w) // 2)
            rec_t = rec_t[..., top : top + h, left : left + w]
    return rec_t
