#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动提取特征 + 生成纹理标注模板"""
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
import pandas as pd
from sklearn.cluster import KMeans

IMG_EXT = {".png"}

def variety_from_filename(path: Path) -> str:
    """编号 = 品种名"""
    stem = path.stem
    stem = re.sub(r"[_\-\s]*\d+$", "", stem).strip()
    return stem

# 颜色
def dominant_color_bgr(bgr, alpha, k=4):
    """在前景像素上 KMeans，返回占比最大簇的 BGR 颜色。"""
    pix = bgr[alpha > 30].reshape(-1, 3).astype(np.float32)
    if len(pix) < 50:
        return np.array([0, 0, 0], np.float32)
    if len(pix) > 20000:  # 采样加速
        idx = np.random.choice(len(pix), 20000, replace=False)
        pix = pix[idx]
    k = min(k, max(1, len(pix) // 10))
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(pix)
    counts = np.bincount(km.labels_, minlength=k)
    return km.cluster_centers_[counts.argmax()]

def bgr_to_hsv_lab(bgr_color):
    px = np.uint8([[bgr_color]])
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]   # H:0-179 S,V:0-255
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]   # L:0-255 a,b:0-255(128中心)
    return hsv, lab

def suggest_color_category(hsv):
    """按 HSV 粗分主色类别"""
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 50:
        return "深色/近黑"
    if s < 35:
        return "白/近白" if v > 170 else "灰白"
    # 有彩色，按色相分
    if h < 8 or h >= 170:
        return "红"
    if h < 20:
        return "橙/桃"
    if h < 38:
        return "黄"
    if h < 85:
        return "绿"
    if h < 130:
        return "蓝/青"
    if h < 150:
        return "紫"
    return "紫红/粉"  # 150-170，朵丽蝶最常见

# 形状
def shape_features(alpha):
    m = (alpha > 30).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return dict(aspect_ratio=np.nan, solidity=np.nan, circularity=np.nan)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 1:
        return dict(aspect_ratio=np.nan, solidity=np.nan, circularity=np.nan)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    long_, short_ = max(w, h), min(w, h)
    aspect = long_ / short_ if short_ > 0 else np.nan
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else np.nan
    perim = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / (perim * perim) if perim > 0 else np.nan
    return dict(aspect_ratio=round(aspect, 3),
                solidity=round(solidity, 3),
                circularity=round(circ, 3))

def extract_one(png_path):
    data = np.fromfile(str(png_path), dtype=np.uint8)
    bgra = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.shape[2] < 4:
        return None
    bgr, alpha = bgra[:, :, :3], bgra[:, :, 3]
    dom = dominant_color_bgr(bgr, alpha)
    hsv, lab = bgr_to_hsv_lab(dom)
    feat = dict(
        main_R=int(dom[2]), main_G=int(dom[1]), main_B=int(dom[0]),
        main_H=int(hsv[0]), main_S=int(hsv[1]), main_V=int(hsv[2]),
        main_L=int(lab[0]), main_a=int(lab[1]), main_b=int(lab[2]),
        color_category_suggest=suggest_color_category(hsv),
    )
    feat.update(shape_features(alpha))
    return feat

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("features")
    out_dir.mkdir(parents=True, exist_ok=True)

    pngs = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXT]
    print(f"找到 {len(pngs)} 张处理后的图片")

    per_variety = defaultdict(list)
    for p in pngs:
        f = extract_one(p)
        if f:
            per_variety[variety_from_filename(p)].append(f)
            print(f"  [ok] {p.name} -> {variety_from_filename(p)}")

    num_cols = ["main_R","main_G","main_B","main_H","main_S","main_V",
                "main_L","main_a","main_b",
                "aspect_ratio","solidity","circularity"]
    rows = []
    for v, feats in sorted(per_variety.items()):
        row = {"品种名（英文）": v, "照片数": len(feats)}
        for c in num_cols:
            vals = [f[c] for f in feats if f.get(c) is not None and not pd.isna(f.get(c))]
            row[c] = round(float(np.mean(vals)), 3) if vals else np.nan
        cats = [f["color_category_suggest"] for f in feats]
        row["color_category_suggest"] = Counter(cats).most_common(1)[0][0]
        rows.append(row)

    auto = pd.DataFrame(rows)
    auto_path = out_dir / "auto_features.csv"
    auto.to_csv(auto_path, index=False, encoding="utf-8-sig")
    print(f"\n自动特征已保存: {auto_path}  （{len(auto)} 个品种）")

    # 纹理标注模板
    tmpl = pd.DataFrame({
        "品种名（英文）": auto["品种名（英文）"],
        "color_category_suggest": auto["color_category_suggest"],
        "main_color_final": "",   # 看图确认后的主色类别
        "edge_color": "",         # 边缘色：如 白边/深红边/无明显边
        "lip_color": "",          # 唇瓣颜色：如 黄/紫红/白
        "has_spots": "",          # 有无斑点/喷点  1=有 0=无
        "has_stripes": "",        # 有无条纹/脉纹  1=有 0=无
        "has_eyespot": "",        # 有无眼斑/色块  1=有 0=无
    })
    tmpl_path = out_dir / "texture_template.csv"
    tmpl.to_csv(tmpl_path, index=False, encoding="utf-8-sig")
    print(f"纹理标注模板已保存: {tmpl_path}")

if __name__ == "__main__":
    main()
