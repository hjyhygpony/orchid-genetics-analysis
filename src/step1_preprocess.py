#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""白平衡，抠图 ，统一尺寸 """
import sys
import os
from pathlib import Path
import numpy as np
import cv2

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT_SIZE = 512  # 统一边长

# 白平衡
def gray_world_white_balance(bgr):
    img = bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)  # B,G,R 各自均值
    gray = means.mean()
    scale = gray / (means + 1e-6)
    img = img * scale
    return np.clip(img, 0, 255).astype(np.uint8)

# 抠图
def cutout_rembg(bgr):
    from rembg import remove
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = remove(rgb)  # RGBA
    if out.shape[2] == 4:
        return out[:, :, 3]
    return np.full(bgr.shape[:2], 255, np.uint8)

def cutout_grabcut(bgr):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    # 距离边缘 8% 的矩形作为“大概是前景”的初始框
    m = int(min(h, w) * 0.08)
    rect = (m, m, w - 2 * m, h - 2 * m)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, np.uint8)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    # 清理小噪点
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, k)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, k)
    return alpha

def get_alpha(bgr):
    try:
        import rembg  # noqa
        return cutout_rembg(bgr)
    except Exception:
        return cutout_grabcut(bgr)

# 裁剪 + 补正方形 + 缩放
def crop_pad_resize(bgr, alpha, size=OUT_SIZE):
    ys, xs = np.where(alpha > 10)
    if len(xs) == 0:
        x0, y0, x1, y1 = 0, 0, bgr.shape[1], bgr.shape[0]
    else:
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
    crop_bgr = bgr[y0:y1, x0:x1]
    crop_a = alpha[y0:y1, x0:x1]
    h, w = crop_a.shape
    side = max(h, w)
    pad_bgr = np.zeros((side, side, 3), np.uint8)
    pad_a = np.zeros((side, side), np.uint8)
    oy, ox = (side - h) // 2, (side - w) // 2
    pad_bgr[oy:oy + h, ox:ox + w] = crop_bgr
    pad_a[oy:oy + h, ox:ox + w] = crop_a
    pad_bgr = cv2.resize(pad_bgr, (size, size), interpolation=cv2.INTER_AREA)
    pad_a = cv2.resize(pad_a, (size, size), interpolation=cv2.INTER_AREA)
    bgra = cv2.cvtColor(pad_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = pad_a
    return bgra

def solidify(alpha, thresh=10):
    fg = (alpha > thresh).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return alpha
    c = max(cnts, key=cv2.contourArea)          # 最大外轮廓
    out = np.zeros_like(alpha)
    cv2.drawContours(out, [c], -1, 255, cv2.FILLED)   # 整块填实心
    return out

def process_one(in_path, out_path):
    data = np.fromfile(str(in_path), dtype=np.uint8)  # 支持中文路径
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  [跳过] 读不出来: {in_path}")
        return False
    bgr = gray_world_white_balance(bgr)
    alpha = get_alpha(bgr)
    alpha = solidify(alpha)
    bgra = crop_pad_resize(bgr, alpha)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", bgra)
    if ok:
        buf.tofile(str(out_path))
    return ok

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    in_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"找到 {len(files)} 张图片，开始处理...")
    done = 0
    for p in files:
        rel = p.relative_to(in_dir).with_suffix(".png")
        out_p = out_dir / rel
        if process_one(p, out_p):
            done += 1
            print(f"  [ok] {rel}")
    print(f"完成：{done}/{len(files)} → {out_dir}")

if __name__ == "__main__":
    main()
