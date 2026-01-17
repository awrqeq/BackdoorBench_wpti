from __future__ import annotations

import torch


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """Convert an RGB image (C,H,W) in [0,1] to YCbCr (BT.601)."""
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError("rgb_to_ycbcr expects shape (3,H,W)")
    img = img.to(torch.float32)
    r, g, b = img[0], img[1], img[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return torch.stack((y, cb, cr), dim=0)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """Convert a YCbCr image (C,H,W) back to RGB (values unclamped)."""
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError("ycbcr_to_rgb expects shape (3,H,W)")
    img = img.to(torch.float32)
    y, cb, cr = img[0], img[1] - 0.5, img[2] - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack((r, g, b), dim=0)
