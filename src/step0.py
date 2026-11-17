#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path
import numpy as np
import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
OUT_SIZE = 512

def imread_unicode(path):
    data = np.fromfile(str(path), np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)   # 保留可能存在的 alpha

def imwrite_unicode(path, img):
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(str(path))
    return ok

def gray_world(bgr, mask=None, strength=0.6):
    """灰世界白平衡，可调强度；有 mask 时只用前景像素估计色偏。"""
    if strength <= 0:
        return bgr
    img = bgr.astype(np.float32)
    if mask is not None and mask.sum() > 0:
        means = img[mask > 0].reshape(-1, 3).mean(axis=0)
    else:
        means = img.reshape(-1, 3).mean(axis=0)
    gray = means.mean()
    full_scale = gray / (means + 1e-6)
    scale = (1 - strength) + strength * full_scale     # 按强度在“不变”和“全力”之间插值
    return np.clip(img * scale, 0, 255).astype(np.uint8)

def square_resize(bgr, alpha, size):
    """保持长宽比 → 补成正方形 → 缩放到 size。有 alpha 则先按花朵裁剪居中。"""
    if alpha is not None:
        ys, xs = np.where(alpha > 10)
        if len(xs) > 0:
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            bgr, alpha = bgr[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
    h, w = bgr.shape[:2]
    side = max(h, w)
    oy, ox = (side - h) // 2, (side - w) // 2
    pad_bgr = np.zeros((side, side, 3), np.uint8)
    pad_bgr[oy:oy + h, ox:ox + w] = bgr
    pad_bgr = cv2.resize(pad_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if alpha is not None:
        pad_a = np.zeros((side, side), np.uint8)
        pad_a[oy:oy + h, ox:ox + w] = alpha
        pad_a = cv2.resize(pad_a, (size, size), interpolation=cv2.INTER_AREA)
        out = cv2.cvtColor(pad_bgr, cv2.COLOR_BGR2BGRA)
        out[:, :, 3] = pad_a
        return out
    return pad_bgr

def process_one(in_path, out_path, size, strength):
    img = imread_unicode(in_path)
    if img is None:
        print("  [跳过] 读不出:", in_path); return False
    if img.ndim == 3 and img.shape[2] == 4:        # 带透明通道
        bgr, alpha = img[:, :, :3].copy(), img[:, :, 3].copy()
    elif img.ndim == 2:                            # 灰度
        bgr, alpha = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None
    else:                                          # 普通 RGB
        bgr, alpha = img.copy(), None
    mask = (alpha > 10).astype(np.uint8) if alpha is not None else None
    bgr = gray_world(bgr, mask, strength)
    out = square_resize(bgr, alpha, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imwrite_unicode(out_path, out)

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    size = int(sys.argv[3]) if len(sys.argv) > 3 else OUT_SIZE
    strength = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"找到 {len(files)} 张图，目标 {size}x{size}，白平衡强度 {strength}")
    done = 0
    for p in files:
        rel = p.relative_to(in_dir).with_suffix(".png")
        if process_one(p, out_dir / rel, size, strength):
            done += 1; print("  [ok]", rel)
    print(f"完成 {done}/{len(files)} → {out_dir}")

if __name__ == "__main__":
    main()