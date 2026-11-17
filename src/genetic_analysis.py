# -*- coding: utf-8 -*-
"""
朵丽蝶遗传规律分析

1. 统计遗传概率
2. 关联规则分析，筛选 提升度大于 1.5 的规则
3. 聚类分析，并画 Dendrogram 树状图
4. 找出重要祖先/奠基者
5. 用孤立森林找异常个体
6. 用决策树进行分类，并计算 F1 分数
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 固定列名
NAME_COL = "品种名（英文）"
FULL_NAME_COL = "完整品种名（参考）"
MOTHER_COL = "母本"
FATHER_COL = "父本"
GEN_COL = "祖先代数"
TYPE_COL = "杂交类型"
PHOTO_COUNT_COL = "照片数"

# 数值型特征：新版特征总表中自动提取的颜色、比例和形状特征
# main_* 表示主色区域，sec_* 表示第二主色区域，*_frac 表示该颜色区域占比
NUMERIC_FEATURES = [
    "main_R", "main_G", "main_B",
    "main_H", "main_S", "main_V",
    "main_L", "main_a", "main_b", "main_frac",
    "sec_R", "sec_G", "sec_B",
    "sec_H", "sec_S", "sec_V",
    "sec_L", "sec_a", "sec_b", "sec_frac",
    "aspect_ratio", "solidity", "circularity",
]

# 类别型特征：主色、第二主色、边缘色、唇瓣色、颜色建议分类等
CATEGORICAL_FEATURES = [
    "color_category_suggest",
    "sec_color_category_suggest",
    "main_color_final",
    "edge_color",
    "lip_color",
]

# 二值纹理特征：0/1
BINARY_FEATURES = [
    "has_spots",
    "has_stripes",
    "has_eyespot",
]

# 用于聚类、异常检测、分类的全部表型特征
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

# 用于遗传规律统计的重点特征
TRAIT_FEATURES = [
    "main_color_final",
    "edge_color",
    "lip_color",
    "has_spots",
    "has_stripes",
    "has_eyespot",
]

# 输出文件中的中文字段名。程序内部仍使用原始列名，保存结果时统一转换为中文。
COLUMN_CN = {
    NAME_COL: "品种名（英文）",
    FULL_NAME_COL: "完整品种名（参考）",
    MOTHER_COL: "母本",
    FATHER_COL: "父本",
    GEN_COL: "祖先代数",
    TYPE_COL: "杂交类型",
    PHOTO_COUNT_COL: "照片数",
    "main_R": "主色R值",
    "main_G": "主色G值",
    "main_B": "主色B值",
    "main_H": "主色H值",
    "main_S": "主色S值",
    "main_V": "主色V值",
    "main_L": "主色L值",
    "main_a": "主色a值",
    "main_b": "主色b值",
    "main_frac": "主色占比",
    "sec_R": "第二主色R值",
    "sec_G": "第二主色G值",
    "sec_B": "第二主色B值",
    "sec_H": "第二主色H值",
    "sec_S": "第二主色S值",
    "sec_V": "第二主色V值",
    "sec_L": "第二主色L值",
    "sec_a": "第二主色a值",
    "sec_b": "第二主色b值",
    "sec_frac": "第二主色占比",
    "aspect_ratio": "花瓣长宽比",
    "solidity": "花朵饱满度",
    "circularity": "花朵圆度",
    "color_category_suggest": "自动建议主色类别",
    "sec_color_category_suggest": "自动建议第二主色类别",
    "main_color_final": "最终主色",
    "edge_color": "边缘色",
    "lip_color": "唇瓣颜色",
    "has_spots": "是否有斑点",
    "has_stripes": "是否有条纹",
    "has_eyespot": "是否有眼斑",
    "母本_lineage": "族谱表母本",
    "父本_lineage": "族谱表父本",
    "祖先代数_lineage": "族谱表祖先代数",
    "杂交类型_lineage": "族谱表杂交类型",
    "name_key": "品种名匹配键",
    "zip_image_count": "压缩包图片数",
    "image_examples": "图片文件示例",
    "photo_count_check": "照片数核对结果",
    "cluster": "聚类编号",
    "isolation_score": "孤立森林分数",
    "parent_feature_distance": "亲本特征距离",
    "真实类别": "真实类别",
    "预测类别": "预测类别",
    "前项": "前项",
    "后项": "后项",
    "support": "支持度",
    "confidence": "置信度",
    "lift": "提升度",
}


def cn_col(col: object) -> str:
    """把内部字段名转换成中文展示名。"""
    s = str(col)
    return COLUMN_CN.get(s, s)


def to_chinese_columns(df: pd.DataFrame) -> pd.DataFrame:
    """保存结果前，把表头统一改成中文。"""
    return df.rename(columns={c: cn_col(c) for c in df.columns})


def display_feature_name(feature: object) -> str:
    """把特征名或独热编码后的特征名转成中文显示。"""
    s = str(feature)
    if s in COLUMN_CN:
        return COLUMN_CN[s]
    for raw, cn in sorted(COLUMN_CN.items(), key=lambda x: len(x[0]), reverse=True):
        prefix = raw + "_"
        if s.startswith(prefix):
            value = s[len(prefix):]
            return f"{cn}={value}"
    return s

# 工具函数
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def canon_name(name: object) -> str:
    """把品种名转成便于匹配图片文件名的格式。"""
    if pd.isna(name):
        return ""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def safe_read_csv(path: str | Path) -> pd.DataFrame:
    """自动尝试常见中文编码读取 CSV。"""
    path = Path(path)
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("无法识别 CSV 编码，请另存为 UTF-8 后重试。", b"", 0, 1, "unknown encoding")


def clean_parent_value(x: object) -> Optional[str]:
    """把父本/母本中的空值和「野生种」处理成 None。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() == "nan" or s == "野生种":
        return None
    return s


def available_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def make_ohe() -> OneHotEncoder:
    """兼容不同 sklearn 版本的 OneHotEncoder。"""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def resolve_input_path(user_path: str | Path, kind: str, patterns: Sequence[str]) -> Path:
    p = Path(user_path)

    # 绝对路径或相对路径本身存在
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    candidates = []
    if not p.is_absolute():
        candidates.append(cwd / p)
        candidates.append(script_dir / p)

    # 根据通配符自动寻找
    for base in [cwd, script_dir]:
        for pattern in patterns:
            candidates.extend(sorted(base.glob(pattern)))

    # 去重
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists() and c.is_file():
            print(f"[自动找到{kind}] {c.resolve()}")
            return c.resolve()

    msg = [
        f"找不到{kind}文件：{user_path}",
        f"当前工作目录：{cwd}",
        f"脚本所在目录：{script_dir}",
        "请把文件放到脚本同一文件夹，或运行时用完整路径指定。",
        "程序尝试过的候选路径：",
    ]
    msg.extend([f"  - {c}" for c in unique_candidates[:30]])
    raise FileNotFoundError("\n".join(msg))


# 读取三份文件并合并
def count_images_from_zip(zip_path: str | Path) -> pd.DataFrame:
    """
    读取 processed1.zip，统计每个品种对应的图片数量。
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"找不到图片压缩包：{zip_path}")

    counts: Dict[str, int] = defaultdict(int)
    examples: Dict[str, List[str]] = defaultdict(list)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            filename = info.filename
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            base = os.path.basename(filename)
            stem = os.path.splitext(base)[0]
            variety = re.sub(r"_\d+$", "", stem)
            key = canon_name(variety)
            counts[key] += 1
            if len(examples[key]) < 3:
                examples[key].append(filename)

    rows = []
    for key, count in sorted(counts.items()):
        rows.append({
            "name_key": key,
            "zip_image_count": count,
            "image_examples": "; ".join(examples[key]),
        })
    return pd.DataFrame(rows)


def load_and_merge(lineage_path: str | Path, features_path: str | Path, zip_path: str | Path, out_dir: Path) -> pd.DataFrame:
    """读取族谱、特征表、图片压缩包，并合并成分析需要的数据表。"""
    lineage = pd.read_excel(lineage_path, sheet_name=0)
    features = safe_read_csv(features_path)
    image_counts = count_images_from_zip(zip_path)

    # 清理列名
    lineage.columns = [str(c).strip() for c in lineage.columns]
    features.columns = [str(c).strip() for c in features.columns]

    # 检查关键列
    need_lineage = [NAME_COL, FULL_NAME_COL, MOTHER_COL, FATHER_COL, GEN_COL, TYPE_COL]
    need_features = [NAME_COL, MOTHER_COL, FATHER_COL, GEN_COL, TYPE_COL] + available_cols(features, MODEL_FEATURES)

    missing_l = [c for c in need_lineage if c not in lineage.columns]
    missing_f = [c for c in need_features if c not in features.columns]

    if missing_l:
        raise ValueError(f"族谱表缺少列：{missing_l}")
    if missing_f:
        raise ValueError(f"特征表缺少列：{missing_f}")

    lineage[NAME_COL] = lineage[NAME_COL].astype(str).str.strip()
    features[NAME_COL] = features[NAME_COL].astype(str).str.strip()

    # 以特征表为主，族谱表用于核对和补充
    lineage_small = lineage[need_lineage].copy()
    lineage_small = lineage_small.rename(columns={
        MOTHER_COL: "母本_lineage",
        FATHER_COL: "父本_lineage",
        GEN_COL: "祖先代数_lineage",
        TYPE_COL: "杂交类型_lineage",
    })

    df = features.merge(lineage_small, on=[NAME_COL, FULL_NAME_COL], how="left")

    # 用族谱表补齐或校正父母、代数、类型
    for raw, backup in [
        (MOTHER_COL, "母本_lineage"),
        (FATHER_COL, "父本_lineage"),
        (GEN_COL, "祖先代数_lineage"),
        (TYPE_COL, "杂交类型_lineage"),
    ]:
        if backup in df.columns:
            df[raw] = df[raw].where(df[raw].notna(), df[backup])

    # 加入图片数量核对
    df["name_key"] = df[NAME_COL].map(canon_name)
    df = df.merge(image_counts, on="name_key", how="left")
    df["zip_image_count"] = df["zip_image_count"].fillna(0).astype(int)

    if PHOTO_COUNT_COL in df.columns:
        df["photo_count_check"] = np.where(
            df[PHOTO_COUNT_COL].fillna(-1).astype(int) == df["zip_image_count"],
            "一致",
            "不一致"
        )
    else:
        df["photo_count_check"] = "无照片数字段"

    # 数值列转换
    for col in available_cols(df, NUMERIC_FEATURES + BINARY_FEATURES + [GEN_COL, PHOTO_COUNT_COL]):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 输出合并后的数据
    to_chinese_columns(df).to_csv(out_dir / "00_合并分析数据.csv", index=False, encoding="utf-8-sig")

    # 输出数据检查报告
    check = {
        "族谱表行列数": list(lineage.shape),
        "特征表行列数": list(features.shape),
        "合并后行列数": list(df.shape),
        "压缩包中匹配到的品种数": int(image_counts.shape[0]),
        "压缩包图片总数": int(image_counts["zip_image_count"].sum()) if not image_counts.empty else 0,
        "无图片品种": df.loc[df["zip_image_count"] == 0, NAME_COL].tolist(),
        "照片数不一致品种": df.loc[df["photo_count_check"] == "不一致", NAME_COL].tolist(),
        "使用的数值特征": [cn_col(c) for c in available_cols(df, NUMERIC_FEATURES)],
        "使用的类别特征": [cn_col(c) for c in available_cols(df, CATEGORICAL_FEATURES)],
        "使用的二值特征": [cn_col(c) for c in available_cols(df, BINARY_FEATURES)],
        "用于遗传统计的重点性状": [cn_col(c) for c in available_cols(df, TRAIT_FEATURES)],
    }
    (out_dir / "00_数据检查.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")

    return df


# 代际统计与遗传概率
def generation_trait_stats(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    统计每一代中颜色、斑点、条纹等性状出现频率。
    """
    rows = []
    trait_cols = available_cols(df, TRAIT_FEATURES)

    for gen, sub in df.groupby(GEN_COL):
        for col in trait_cols:
            if col in BINARY_FEATURES:
                count_1 = int((sub[col] == 1).sum())
                total = int(sub[col].notna().sum())
                rows.append({
                    "祖先代数": gen,
                    "性状列": cn_col(col),
                    "性状取值": 1,
                    "出现次数": count_1,
                    "该代总数": total,
                    "出现比例": count_1 / total if total else np.nan,
                })
            else:
                vc = sub[col].dropna().astype(str).value_counts()
                total = int(sub[col].notna().sum())
                for value, count in vc.items():
                    rows.append({
                        "祖先代数": gen,
                        "性状列": cn_col(col),
                        "性状取值": value,
                        "出现次数": int(count),
                        "该代总数": total,
                        "出现比例": count / total if total else np.nan,
                    })

    result = pd.DataFrame(rows).sort_values(["性状列", "性状取值", "祖先代数"])
    to_chinese_columns(result).to_csv(out_dir / "01_代际性状频率统计.csv", index=False, encoding="utf-8-sig")

    # 专门输出「紫红色」相关统计
    purple_rows = result[
        result["性状取值"].astype(str).str.contains("紫红", na=False)
    ].copy()
    to_chinese_columns(purple_rows).to_csv(out_dir / "01_紫红色性状代际统计.csv", index=False, encoding="utf-8-sig")

    return result


    return result


def trait_base_rates(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """计算每个性状类别的整体基率，并标记变异不足的性状。

    一个近乎恒定的性状（如 has_eyespot 几乎全部为 0），亲子"一致"
    反映的是基率而非遗传信号——因为没有变异就谈不上遗传。
    只有基率在 [0.1, 0.9] 之间的性状才有足够变异来谈传递率。
    """
    rows = []
    trait_cols = available_cols(df, TRAIT_FEATURES)

    for col in trait_cols:
        s = df[col].dropna()
        total = len(s)
        if col in BINARY_FEATURES:
            n_one = int((s == 1).sum())
            n_zero = int((s == 0).sum())
            base_rate = n_one / total if total else np.nan
            rows.append({
                "性状列": cn_col(col),
                "总样本数": total,
                "取值=1 数量": n_one,
                "取值=0 数量": n_zero,
                "基率（取1的比例）": base_rate,
                "变异是否充足": "是" if 0.1 <= base_rate <= 0.9 else "否（近乎恒定，不宜谈传递率）",
            })
        else:
            vc = s.astype(str).value_counts()
            max_rate = vc.max() / total if total else np.nan
            rows.append({
                "性状列": cn_col(col),
                "总样本数": total,
                "可能取值数": int(vc.count()),
                "最高频取值": vc.index[0],
                "最高频比例": max_rate,
                "变异是否充足": "是" if max_rate < 0.9 else "否（近乎恒定，不宜谈传递率）",
            })

    result = pd.DataFrame(rows)
    to_chinese_columns(result).to_csv(out_dir / "02_性状整体基率.csv", index=False, encoding="utf-8-sig")
    return result


def _wilson_ci(n_success: int, n_total: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson 分数置信区间，适用于小样本比例估计。"""
    import scipy.stats as st
    if n_total == 0:
        return (np.nan, np.nan)
    z = st.norm.ppf(1 - alpha / 2)
    p = n_success / n_total
    denom = 1 + z * z / n_total
    centre = (p + z * z / (2 * n_total)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n_total)) / n_total) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def parent_child_inheritance(df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    计算亲本到子代的性状传递概率。
    - 父母都具有同一性状时，计算子代是否也具有该性状
    - 至少一个亲本具有某性状时，计算子代是否具有该性状
    """
    name_to_row = {
        str(row[NAME_COL]).strip(): row
        for _, row in df.iterrows()
        if pd.notna(row[NAME_COL])
    }

    detail_rows = []
    trait_cols = available_cols(df, TRAIT_FEATURES)

    for _, child in df.iterrows():
        child_name = str(child[NAME_COL]).strip()
        mother_name = clean_parent_value(child.get(MOTHER_COL))
        father_name = clean_parent_value(child.get(FATHER_COL))

        parents = []
        if mother_name and mother_name in name_to_row:
            parents.append(("母本", mother_name, name_to_row[mother_name]))
        if father_name and father_name in name_to_row:
            parents.append(("父本", father_name, name_to_row[father_name]))

        if not parents:
            continue

        for trait in trait_cols:
            child_value = child.get(trait)
            if pd.isna(child_value):
                continue

            parent_values = []
            for parent_role, parent_name, parent_row in parents:
                pv = parent_row.get(trait)
                if pd.notna(pv):
                    parent_values.append((parent_role, parent_name, pv))

            if not parent_values:
                continue

            # 逐个亲本计算：该亲本性状是否传给子代
            for parent_role, parent_name, pv in parent_values:
                inherited = int(str(child_value) == str(pv))
                detail_rows.append({
                    "子代": child_name,
                    "母本": mother_name,
                    "父本": father_name,
                    "亲本角色": parent_role,
                    "亲本名称": parent_name,
                    "性状列": cn_col(trait),
                    "亲本性状": pv,
                    "子代性状": child_value,
                    "是否继承": inherited,
                    "判断方式": "子代性状与该亲本性状完全相同",
                })

            # 父母都存在时，额外计算「父母都一样」的传递概率
            if len(parent_values) == 2:
                pv1 = parent_values[0][2]
                pv2 = parent_values[1][2]
                if str(pv1) == str(pv2):
                    inherited = int(str(child_value) == str(pv1))
                    detail_rows.append({
                        "子代": child_name,
                        "母本": mother_name,
                        "父本": father_name,
                        "亲本角色": "父母共同",
                        "亲本名称": f"{parent_values[0][1]} + {parent_values[1][1]}",
                        "性状列": cn_col(trait),
                        "亲本性状": pv1,
                        "子代性状": child_value,
                        "是否继承": inherited,
                        "判断方式": "父母性状相同，判断子代是否保持该性状",
                    })

    detail = pd.DataFrame(detail_rows)
    to_chinese_columns(detail).to_csv(out_dir / "02_亲本子代性状继承明细.csv", index=False, encoding="utf-8-sig")

    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby(["性状列", "亲本角色", "亲本性状", "判断方式"], dropna=False)
            .agg(样本数=("是否继承", "size"), 继承次数=("是否继承", "sum"), 传递概率=("是否继承", "mean"))
            .reset_index()
            .sort_values(["传递概率", "样本数"], ascending=[False, False])
        )

        # 添加 Wilson 置信区间，并标记小样本不可靠行
        ci_lows, ci_highs, reliability = [], [], []
        for _, row in summary.iterrows():
            lo, hi = _wilson_ci(int(row["继承次数"]), int(row["样本数"]))
            ci_lows.append(lo)
            ci_highs.append(hi)
            if int(row["样本数"]) < 5:
                reliability.append("样本过小（n<5），不宜下结论")
            elif int(row["样本数"]) < 10:
                reliability.append("样本偏小（n<10），解读需谨慎")
            else:
                reliability.append("样本尚可")
        summary["CI下界(95%)"] = ci_lows
        summary["CI上界(95%)"] = ci_highs
        summary["可靠性标注"] = reliability

    to_chinese_columns(summary).to_csv(out_dir / "02_性状传递概率汇总.csv", index=False, encoding="utf-8-sig")
    return detail, summary


def numeric_parent_similarity(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """计算数值特征的亲子相似度：子代与父母均值越接近，说明越相似。"""
    name_to_row = {str(row[NAME_COL]).strip(): row for _, row in df.iterrows()}
    rows = []

    for _, child in df.iterrows():
        child_name = str(child[NAME_COL]).strip()
        mother_name = clean_parent_value(child.get(MOTHER_COL))
        father_name = clean_parent_value(child.get(FATHER_COL))

        parent_rows = []
        for p in [mother_name, father_name]:
            if p and p in name_to_row:
                parent_rows.append(name_to_row[p])

        if not parent_rows:
            continue

        for col in available_cols(df, NUMERIC_FEATURES):
            child_val = pd.to_numeric(child.get(col), errors="coerce")
            p_vals = [pd.to_numeric(pr.get(col), errors="coerce") for pr in parent_rows]
            p_vals = [v for v in p_vals if pd.notna(v)]

            if pd.isna(child_val) or not p_vals:
                continue

            parent_mean = float(np.mean(p_vals))
            abs_diff = abs(float(child_val) - parent_mean)
            rows.append({
                "子代": child_name,
                "母本": mother_name,
                "父本": father_name,
                "数值特征": cn_col(col),
                "子代值": float(child_val),
                "亲本均值": parent_mean,
                "绝对差": abs_diff,
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        summary = (
            result.groupby("数值特征")
            .agg(样本数=("绝对差", "size"), 平均绝对差=("绝对差", "mean"), 中位绝对差=("绝对差", "median"))
            .reset_index()
            .sort_values("平均绝对差")
        )
        to_chinese_columns(summary).to_csv(out_dir / "02_数值特征亲子相似度汇总.csv", index=False, encoding="utf-8-sig")

    to_chinese_columns(result).to_csv(out_dir / "02_数值特征亲子相似度明细.csv", index=False, encoding="utf-8-sig")
    return result


# 关联规则
def discretize_numeric_for_rules(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """把连续数值分成 low/mid/high，便于关联规则分析"""
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        if s.notna().sum() < 5 or s.nunique(dropna=True) <= 3:
            out[col] = s.round(3).astype(str)
        else:
            out[col] = pd.qcut(s, q=3, labels=["低", "中", "高"], duplicates="drop").astype(str)
    return out


def make_transactions(df: pd.DataFrame) -> List[Set[str]]:
    """把每个品种的一行特征转成一个item集合"""
    # 不直接使用所有RGB/HSV/Lab数值
    # 保留颜色类别、纹理有无、颜色占比和形状指标
    # 注意：剔除 color_category_suggest / sec_color_category_suggest，
    # 它们与 main_color_final 是同一属性的两次测量（机器建议 vs 人工确认），
    # 同时保留会产生"同义反复"的虚假高 Lift 规则。
    rule_numeric_cols = ["main_frac", "sec_frac", "aspect_ratio", "solidity", "circularity"]
    rule_categorical_cols = ["main_color_final", "edge_color", "lip_color"]
    use_cols = available_cols(df, rule_categorical_cols + BINARY_FEATURES + rule_numeric_cols + [TYPE_COL])
    temp = df[use_cols].copy()
    temp = discretize_numeric_for_rules(temp, rule_numeric_cols)

    transactions = []
    for _, row in temp.iterrows():
        items = set()
        for col in use_cols:
            v = row.get(col)
            if pd.isna(v):
                continue

            # 二值特征只保留「有」的状态
            if col in BINARY_FEATURES:
                if int(v) == 1:
                    label = {
                        "has_spots": "有斑点",
                        "has_stripes": "有条纹",
                        "has_eyespot": "有眼斑",
                    }.get(col, col)
                    items.add(label)
                continue

            items.add(f"{cn_col(col)}={str(v).strip()}")
        transactions.append(items)
    return transactions


def apriori_rules(df: pd.DataFrame, out_dir: Path, min_support: float = 0.08, min_confidence: float = 0.60, min_lift: float = 1.50, max_len: int = 3) -> pd.DataFrame:
    transactions = make_transactions(df)
    n = len(transactions)
    if n == 0:
        return pd.DataFrame()

    all_items = sorted(set().union(*transactions))
    support_cache: Dict[Tuple[str, ...], float] = {}

    def support(itemset: Sequence[str]) -> float:
        key = tuple(sorted(itemset))
        if key not in support_cache:
            itemset_s = set(key)
            support_cache[key] = sum(itemset_s.issubset(t) for t in transactions) / n
        return support_cache[key]

    frequent_itemsets = []
    for k in range(1, max_len + 1):
        for combo in itertools.combinations(all_items, k):
            sup = support(combo)
            if sup >= min_support:
                frequent_itemsets.append((combo, sup))

    rows = []
    for itemset, sup_ab in frequent_itemsets:
        if len(itemset) < 2:
            continue
        itemset_set = set(itemset)
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(itemset_set - set(antecedent)))
                sup_a = support(antecedent)
                sup_b = support(consequent)
                if sup_a == 0 or sup_b == 0:
                    continue
                confidence = sup_ab / sup_a
                lift = confidence / sup_b
                if confidence >= min_confidence and lift >= min_lift:
                    rows.append({
                        "前项": " + ".join(antecedent),
                        "后项": " + ".join(consequent),
                        "support": sup_ab,
                        "confidence": confidence,
                        "lift": lift,
                    })

    rules = pd.DataFrame(rows)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)
    to_chinese_columns(rules).to_csv(out_dir / "03_性状关联规则_提升度大于1_5.csv", index=False, encoding="utf-8-sig")
    return rules


# 特征矩阵、聚类、异常检测
def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, ColumnTransformer, List[str]]:
    """构建数值化特征矩阵"""
    feature_cols = available_cols(df, MODEL_FEATURES)
    numeric_cols = [c for c in feature_cols if c in NUMERIC_FEATURES + BINARY_FEATURES]
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]

    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe())
            ]), categorical_cols),
        ],
        remainder="drop"
    )

    X = transformer.fit_transform(df[feature_cols])

    feature_names = [display_feature_name(c) for c in numeric_cols]
    if categorical_cols:
        enc = transformer.named_transformers_["cat"].named_steps["onehot"]
        try:
            feature_names += [display_feature_name(x) for x in enc.get_feature_names_out(categorical_cols)]
        except Exception:
            feature_names += [f"{cn_col(c)}_编码" for c in categorical_cols]

    return X, transformer, feature_names


def cluster_analysis(df: pd.DataFrame, out_dir: Path, n_clusters: int = 4) -> pd.DataFrame:
    """层次聚类，并输出Dendrogram"""
    X, _, _ = build_feature_matrix(df)

    Z = linkage(X, method="ward")
    labels = df[NAME_COL].astype(str).tolist()

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(max(12, len(labels) * 0.45), 7))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=8)
    plt.title("朵丽蝶品种表型特征层次聚类树状图")
    plt.xlabel("品种名")
    plt.ylabel("特征距离")
    plt.tight_layout()
    plt.savefig(out_dir / "04_表型特征聚类树状图.png", dpi=300)
    plt.close()

    cluster_id = fcluster(Z, t=n_clusters, criterion="maxclust")
    result = df[[NAME_COL, GEN_COL, TYPE_COL, "main_color_final", "lip_color"]].copy()
    result["cluster"] = cluster_id
    to_chinese_columns(result).to_csv(out_dir / "04_品种聚类分组结果.csv", index=False, encoding="utf-8-sig")
    return result


def anomaly_detection(df: pd.DataFrame, out_dir: Path, contamination: float = 0.12) -> pd.DataFrame:
    """使用孤立森林检测表型异常个体，同时计算与亲本均值的距离"""
    X, _, _ = build_feature_matrix(df)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42
    )
    pred = iso.fit_predict(X)
    score = iso.decision_function(X)

    name_to_idx = {str(name).strip(): i for i, name in enumerate(df[NAME_COL])}
    parent_distance = []

    for i, row in df.iterrows():
        parent_indices = []
        for p in [clean_parent_value(row.get(MOTHER_COL)), clean_parent_value(row.get(FATHER_COL))]:
            if p and p in name_to_idx:
                parent_indices.append(name_to_idx[p])
        if not parent_indices:
            parent_distance.append(np.nan)
        else:
            parent_vec = X[parent_indices].mean(axis=0)
            parent_distance.append(float(np.linalg.norm(X[i] - parent_vec)))

    result = df[[NAME_COL, MOTHER_COL, FATHER_COL, GEN_COL, TYPE_COL, "main_color_final", "lip_color"]].copy()
    result["是否异常"] = pred == -1
    result["isolation_score"] = score
    result["parent_feature_distance"] = parent_distance
    result = result.sort_values(["是否异常", "isolation_score"], ascending=[False, True])
    to_chinese_columns(result).to_csv(out_dir / "05_异常个体检测结果.csv", index=False, encoding="utf-8-sig")
    return result


# 6. 重要祖先/奠基者分析
def build_descendant_graph(df: pd.DataFrame) -> Dict[str, Set[str]]:
    """建立 parent -> children 图。"""
    graph: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        child = str(row[NAME_COL]).strip()
        for p in [clean_parent_value(row.get(MOTHER_COL)), clean_parent_value(row.get(FATHER_COL))]:
            if p:
                graph[p].add(child)
    return graph


def get_descendants(root: str, graph: Dict[str, Set[str]]) -> Set[str]:
    """递归找某祖先的所有后代"""
    seen = set()
    q = deque(graph.get(root, []))
    while q:
        x = q.popleft()
        if x in seen:
            continue
        seen.add(x)
        q.extend(graph.get(x, []))
    return seen


def trait_match_rate(ancestor: pd.Series, descendants: pd.DataFrame) -> float:
    """计算祖先性状在后代中保持一致的平均比例"""
    rates = []
    for col in available_cols(descendants, TRAIT_FEATURES):
        av = ancestor.get(col)
        if pd.isna(av):
            continue
        values = descendants[col].dropna()
        if values.empty:
            continue
        rate = (values.astype(str) == str(av)).mean()
        rates.append(rate)
    return float(np.mean(rates)) if rates else 0.0


def founder_analysis(df: pd.DataFrame, out_dir: Path, n_permutations: int = 500) -> pd.DataFrame:
    """
    后代数量越多越重要
    祖先性状在后代中出现比例越高越重要

    注：奠基者评分 = 后代数 × 平均性状保持率。后代数完全由节点在树中
    的深度/枢纽位置决定，与遗传贡献无关；保持率又被性状基率污染。
    因此加入置乱检验：随机打乱品种性状 N 次，计算仅凭拓扑能获得的
    评分期望分布，看真实评分是否显著高于"仅凭拓扑"的预期。
    """
    graph = build_descendant_graph(df)
    by_name = {str(row[NAME_COL]).strip(): row for _, row in df.iterrows()}
    trait_cols = available_cols(df, TRAIT_FEATURES)
    rows = []

    for name, ancestor_row in by_name.items():
        desc_names = sorted([d for d in get_descendants(name, graph) if d in by_name])
        if not desc_names:
            continue
        desc_df = df[df[NAME_COL].isin(desc_names)]
        match_rate = trait_match_rate(ancestor_row, desc_df)
        score = len(desc_names) * match_rate

        # 找出保持率最高的几个性状
        stable_traits = []
        for col in trait_cols:
            av = ancestor_row.get(col)
            if pd.isna(av):
                continue
            values = desc_df[col].dropna()
            if values.empty:
                continue
            rate = (values.astype(str) == str(av)).mean()
            if rate >= 0.5:
                stable_traits.append(f"{cn_col(col)}={av}，后代保持率{rate:.2f}")

        rows.append({
            "祖先品种": name,
            "祖先代数": ancestor_row.get(GEN_COL),
            "杂交类型": ancestor_row.get(TYPE_COL),
            "后代数量": len(desc_names),
            "平均性状保持率": match_rate,
            "奠基者评分": score,
            "主要后代": "; ".join(desc_names[:12]),
            "较稳定传递性状": "; ".join(stable_traits[:8]),
        })

    result = pd.DataFrame(rows).sort_values("奠基者评分", ascending=False)

    # ---- 置乱零模型 ----
    rng = np.random.default_rng(42)
    null_scores: Dict[str, List[float]] = defaultdict(list)

    # 预先计算每个节点的"后代集"（纯拓扑，不依赖性状）
    node_descendants: Dict[str, Set[str]] = {}
    for name in by_name:
        desc_names = sorted([d for d in get_descendants(name, graph) if d in by_name])
        node_descendants[name] = set(desc_names)

    node_names_with_desc = sorted([n for n, d in node_descendants.items() if d])

    for _ in range(n_permutations):
        # 随机打乱性状：保持树拓扑不变，但把性状值随机分配给品种
        shuffled = df.copy()
        perm_idx = rng.permutation(len(shuffled))
        shuffled[trait_cols] = shuffled[trait_cols].iloc[perm_idx].values
        shuffled_by_name = {str(row[NAME_COL]).strip(): row for _, row in shuffled.iterrows()}

        for name in node_names_with_desc:
            desc_df = shuffled[shuffled[NAME_COL].isin(node_descendants[name])]
            null_rate = trait_match_rate(shuffled_by_name[name], desc_df)
            null_score = len(node_descendants[name]) * null_rate
            null_scores[name].append(null_score)

    # 计算每个真实评分的置乱 p 值和 z-score
    p_values = []
    z_scores = []
    for _, r in result.iterrows():
        name = r["祖先品种"]
        real_score = r["奠基者评分"]
        if name in null_scores and null_scores[name]:
            null_arr = np.array(null_scores[name])
            p_val = float((np.sum(null_arr >= real_score) + 1) / (len(null_arr) + 1))
            null_mean = float(np.mean(null_arr))
            null_std = float(np.std(null_arr))
            z = float((real_score - null_mean) / null_std) if null_std > 0 else 0.0
        else:
            p_val = np.nan
            z = np.nan
        p_values.append(p_val)
        z_scores.append(z)

    result["零模型p值"] = p_values
    result["零模型z分数"] = z_scores
    result["评分是否显著高于拓扑预期"] = [
        "是（p<0.05）" if p <= 0.05 else "否（仅反映拓扑位置）"
        for p in p_values
    ]

    to_chinese_columns(result).to_csv(out_dir / "06_重要祖先奠基者排名.csv", index=False, encoding="utf-8-sig")
    return result


# 决策树分类模型
def build_chinese_classification_report(y_true: pd.Series, y_pred: np.ndarray, labels: List[str]) -> str:
    precision, recall, f1_values, support_values = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision, macro_recall, macro_f1_value, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1_value, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    lines = []
    lines.append("类别\t精确率\t召回率\tF1分数\t样本数")
    for label, p, r, f1_v, s in zip(labels, precision, recall, f1_values, support_values):
        lines.append(f"{label}\t{p:.4f}\t{r:.4f}\t{f1_v:.4f}\t{int(s)}")
    lines.append(f"整体准确率\t-\t-\t{accuracy:.4f}\t{len(y_true)}")
    lines.append(f"宏平均\t{macro_precision:.4f}\t{macro_recall:.4f}\t{macro_f1_value:.4f}\t{len(y_true)}")
    lines.append(f"加权平均\t{weighted_precision:.4f}\t{weighted_recall:.4f}\t{weighted_f1_value:.4f}\t{len(y_true)}")
    return "\n".join(lines)


def decision_tree_classification(df: pd.DataFrame, out_dir: Path, target_col: str = TYPE_COL) -> Dict[str, object]:
    """
    用决策树进行可解释分类，并通过小范围自动调参提高宏平均F1分数。
    默认使用「杂交类型」作为分类目标
    """
    if target_col not in df.columns:
        raise ValueError(f"找不到分类目标列：{target_col}")

    data = df.dropna(subset=[target_col]).copy()
    y = data[target_col].astype(str)

    if y.nunique() < 2:
        raise ValueError(f"分类目标 {target_col} 只有一个类别，无法训练分类模型。")

    feature_cols = available_cols(data, MODEL_FEATURES)
    X_df = data[feature_cols].copy()

    numeric_cols = [c for c in feature_cols if c in NUMERIC_FEATURES + BINARY_FEATURES]
    categorical_cols = [c for c in feature_cols if c in CATEGORICAL_FEATURES]

    transformer = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_ohe())
            ]), categorical_cols),
        ],
        remainder="drop"
    )

    # 小样本且类别不均衡时，必须分层抽样；否则测试集中少数类可能太少，宏平均F1会明显波动
    vc = y.value_counts()
    stratify = y if vc.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=0.30,
        random_state=42,
        stratify=stratify,
    )

    # 自动调参：解决「野生种」样本少导致少数类识别差的问题
    # class_weight="balanced" 会提高少数类权重；min_samples_leaf 防止树过拟合单个样本
    candidate_params = []
    for class_weight in [None, "balanced"]:
        for max_depth in [1, 2, 3, 4, None]:
            for min_samples_leaf in [1, 2, 3]:
                candidate_params.append({
                    "class_weight": class_weight,
                    "max_depth": max_depth,
                    "min_samples_leaf": min_samples_leaf,
                    "criterion": "gini",
                })

    # 训练集内部做交叉验证，选训练集交叉验证宏平均F1最高的参数
    n_splits = min(3, int(y_train.value_counts().min()))
    cv_rows = []
    best_score = -1.0
    best_params = candidate_params[0]

    for params in candidate_params:
        clf = DecisionTreeClassifier(
            criterion=params["criterion"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight=params["class_weight"],
            random_state=42,
        )
        pipe = Pipeline([
            ("prep", transformer),
            ("clf", clf)
        ])

        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X_train, y_train, scoring="f1_macro", cv=cv)
            cv_score = float(np.mean(scores))
            cv_std = float(np.std(scores))
        else:
            cv_score = np.nan
            cv_std = np.nan

        cv_rows.append({
            "类别权重": "平衡权重" if params["class_weight"] == "balanced" else "不加权重",
            "最大深度": "不限制" if params["max_depth"] is None else params["max_depth"],
            "叶节点最少样本数": params["min_samples_leaf"],
            "划分标准": params["criterion"],
            "训练集交叉验证宏平均F1均值": cv_score,
            "训练集交叉验证宏平均F1标准差": cv_std,
        })

        # 分数相同则优先选择更稳的模型：叶节点样本数更大、深度更浅
        current_depth = 99 if params["max_depth"] is None else params["max_depth"]
        best_depth = 99 if best_params["max_depth"] is None else best_params["max_depth"]
        better = False
        if np.isnan(best_score) or cv_score > best_score:
            better = True
        elif abs(cv_score - best_score) < 1e-12:
            # 分数相同且类别不均衡时，优先选择平衡权重，避免少数类完全识别不出来
            if params["class_weight"] == "balanced" and best_params["class_weight"] != "balanced":
                better = True
            elif params["class_weight"] == best_params["class_weight"]:
                if params["min_samples_leaf"] > best_params["min_samples_leaf"]:
                    better = True
                elif params["min_samples_leaf"] == best_params["min_samples_leaf"] and current_depth < best_depth:
                    better = True

        if better:
            best_score = cv_score
            best_params = params

    cv_df = pd.DataFrame(cv_rows).sort_values(
        ["训练集交叉验证宏平均F1均值", "叶节点最少样本数"],
        ascending=[False, False]
    )
    cv_df.to_csv(out_dir / "07_决策树调参记录.csv", index=False, encoding="utf-8-sig")

    best_clf = DecisionTreeClassifier(
        criterion=best_params["criterion"],
        max_depth=best_params["max_depth"],
        min_samples_leaf=best_params["min_samples_leaf"],
        class_weight=best_params["class_weight"],
        random_state=42,
    )

    pipe = Pipeline([
        ("prep", transformer),
        ("clf", best_clf)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    labels = sorted(y.unique())
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = build_chinese_classification_report(y_test, y_pred, labels)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 再给一个全数据三折交叉验证分数，作为小样本下更稳定的参考
    if y.value_counts().min() >= 2:
        full_cv = StratifiedKFold(n_splits=min(3, int(y.value_counts().min())), shuffle=True, random_state=42)
        full_cv_scores = cross_val_score(pipe, X_df, y, scoring="f1_macro", cv=full_cv)
    else:
        full_cv_scores = np.array([np.nan])

    pd.DataFrame({
        "折号": list(range(1, len(full_cv_scores) + 1)),
        "宏平均F1分数": full_cv_scores,
    }).to_csv(out_dir / "07_决策树交叉验证结果.csv", index=False, encoding="utf-8-sig")

    with open(out_dir / "07_决策树分类结果.txt", "w", encoding="utf-8") as f:
        f.write(f"分类目标：{cn_col(target_col)}\n")
        f.write(f"类别：{labels}\n")
        f.write("最佳参数：\n")
        f.write(f"  类别权重：{'平衡权重' if best_params['class_weight'] == 'balanced' else '不加权重'}\n")
        f.write(f"  最大深度：{'不限制' if best_params['max_depth'] is None else best_params['max_depth']}\n")
        f.write(f"  叶节点最少样本数：{best_params['min_samples_leaf']}\n")
        f.write(f"  训练集交叉验证宏平均F1：{best_score:.4f}\n")
        f.write(f"测试集宏平均F1分数：{macro_f1:.4f}\n")
        f.write(f"全数据三折交叉验证宏平均F1均值：{np.nanmean(full_cv_scores):.4f}\n")
        f.write(f"全数据三折交叉验证宏平均F1标准差：{np.nanstd(full_cv_scores):.4f}\n\n")
        f.write("分类报告：\n")
        f.write(report)
        f.write("\n混淆矩阵：\n")
        f.write(np.array2string(cm))

    # 保存决策树图
    prep = pipe.named_steps["prep"]
    feature_names = [display_feature_name(c) for c in numeric_cols]
    if categorical_cols:
        enc = prep.named_transformers_["cat"].named_steps["onehot"]
        try:
            feature_names += [display_feature_name(x) for x in enc.get_feature_names_out(categorical_cols)]
        except Exception:
            feature_names += [f"{cn_col(c)}_编码" for c in categorical_cols]

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    plt.figure(figsize=(18, 9))
    plot_tree(
        pipe.named_steps["clf"],
        feature_names=feature_names,
        class_names=labels,
        filled=False,
        rounded=True,
        fontsize=8,
    )
    plt.title(f"决策树分类模型：{cn_col(target_col)}")
    plt.tight_layout()
    plt.savefig(out_dir / "07_决策树分类模型.png", dpi=300)
    plt.close()

    pred_df = pd.DataFrame({
        NAME_COL: data.loc[X_test.index, NAME_COL].values if NAME_COL in data.columns else X_test.index.astype(str),
        "真实类别": y_test.values,
        "预测类别": y_pred,
    })
    pred_df["是否预测正确"] = np.where(pred_df["真实类别"] == pred_df["预测类别"], "正确", "错误")
    to_chinese_columns(pred_df).to_csv(out_dir / "07_决策树预测明细.csv", index=False, encoding="utf-8-sig")

    return {
        "target_col": target_col,
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "best_params": best_params,
        "cv_macro_f1_mean": float(np.nanmean(full_cv_scores)),
        "cv_macro_f1_std": float(np.nanstd(full_cv_scores)),
    }


def generate_conclusion(
    df: pd.DataFrame,
    out_dir: Path,
    gen_stats: pd.DataFrame,
    inherit_summary: pd.DataFrame,
    rules: pd.DataFrame,
    clusters: pd.DataFrame,
    anomalies: pd.DataFrame,
    founders: pd.DataFrame,
    clf_result: Dict[str, object],
    base_rates: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# 朵丽蝶遗传规律分析结果说明\n")

    lines.append("## 零、重要前置声明\n")
    lines.append("1. 样本为 MituoLoveDragon 单一后代的祖先谱系（n=33），非随机抽样。")
    lines.append("   所有频率、概率和关联强度仅在本家系内部有效，不能外推到朵丽蝶总体。")
    lines.append("   育种者的长期偏好（如偏爱紫红唇瓣）可能造成选择偏差，")
    lines.append("   谱系中同一野生种在多条路径反复作为亲本出现，")
    lines.append("   将其视为独立数据点会系统性高估频率和关联强度。\n")
    lines.append("2. 祖先代数方向：代数 0 = 2023 年当代品种 MituoLoveDragon，")
    lines.append("   代数 7 = 最古老的野生种 lueddemanniana。")
    lines.append("   代数越大越古老，顺着真实时间（古→今，即第7代→第0代）读。")
    lines.append("   每个代数层是一组不同品种的静态快照，不是同一支血统被追踪的时间序列，")
    lines.append("   因此「频率随代数变化」反映的是你选了哪些祖先放入该代数层，")
    lines.append("   而非性状在遗传时间轴上的动态变化。\n")
    lines.append("3. 颜色数据来自网络实拍照片，未做色卡校准，光照/白平衡不统一，")
    lines.append("   因此 RGB/HSV/Lab 跨品种绝对值不可直接比较。")
    lines.append("   颜色差异的一部分可能来自拍摄噪声而非生物学变异。")
    lines.append("   形态指标（长宽比、饱满度、圆度）是比值，对拍摄条件相对鲁棒。")

    lines.append("\n## 一、数据基础\n")
    lines.append(f"本次分析共纳入 {len(df)} 个朵丽蝶相关品种。")
    lines.append(f"其中，杂交类型分布为：{df[TYPE_COL].value_counts().to_dict()}。")
    if "zip_image_count" in df.columns:
        lines.append(f"图片压缩包共匹配到 {int(df['zip_image_count'].sum())} 张图片，覆盖 {int((df['zip_image_count'] > 0).sum())} 个品种。")
        photo_counts = df["zip_image_count"]
        lines.append(f"各品种照片数 1–9 张不等（中位 {int(photo_counts.median())}），单张品种的特征可靠性低于多张品种。")

    lines.append("\n## 二、性状整体基率（变异是否充足）\n")
    if not base_rates.empty:
        for _, r in base_rates.iterrows():
            var_text = r.get("变异是否充足", "")
            lines.append(f"- {r['性状列']}：{var_text}（{r.to_dict()}）")
        lines.append("只有变异充足的性状才适合谈「传递率」；近乎恒定性状的亲子一致反映的是基率，非遗传信号。")

    lines.append("\n## 三、祖先代数中性状的分布（静态快照，非遗传动态）\n")
    if not gen_stats.empty:
        lines.append("各性状在不同代数层的出现比例见 `01_代际性状频率统计.csv`。")
        lines.append("深层代数（如第5–7代）有条纹的品种多，是因为带有条纹的野生奠基种")
        lines.append("（amboinensis、lueddemanniana、micholitzii、mariae）被放在了这些深层代数。")
        lines.append("顺着真实时间方向（古→今，即第7代→第0代），条纹是从野生种向现代杂交种被逐渐稀释掉的，")
        lines.append("不存在「隔代沉默后重新激活」的机制。")

    lines.append("\n## 四、亲本到子代的传递概率\n")
    lines.append("注：已添加 Wilson 95% 置信区间，n<5 的估计标注为不可靠。")
    if not inherit_summary.empty:
        n_col = "样本数" if "样本数" in inherit_summary.columns else inherit_summary.columns[0]
        reliable = inherit_summary[
            inherit_summary[n_col].astype(int) >= 5
        ].head(8) if n_col in inherit_summary.columns else pd.DataFrame()
        if not reliable.empty:
            for _, r in reliable.iterrows():
                lines.append(
                    f"- `{r['性状列']}` 中，亲本性状 `{r['亲本性状']}` 的传递概率约为 {r['传递概率']:.2f}"
                    f"（样本数 {int(r[n_col])}，95%CI [{r.get('CI下界(95%)', 'N/A')}, {r.get('CI上界(95%)', 'N/A')}]）。"
                )
        else:
            lines.append("可靠样本（n≥5）的传递概率为空，大部分估计因样本过小不可信。")
    else:
        lines.append("由于亲本缺失或亲本未进入表格，未能计算传递概率。")

    lines.append("\n## 五、关联规则（已剔除同义反复特征）\n")
    lines.append("注：已排除 color_category_suggest / sec_color_category_suggest，")
    lines.append("因其与 main_color_final 是同一属性的两次测量，会产生虚假高 Lift。")
    if not rules.empty:
        lines.append("关联规则算法挖掘得到的高提升度规则见 `03_性状关联规则_提升度大于1_5.csv`。代表性规则如下：")
        for _, r in rules.head(8).iterrows():
            lines.append(
                f"- 若 `{r['前项']}`，则更容易出现 `{r['后项']}`；置信度={r['confidence']:.2f}，提升度={r['lift']:.2f}。"
            )
    else:
        lines.append("在当前阈值下未发现 提升度大于 1.5 的强关联规则，可以适当降低最小支持度后重新运行。")

    lines.append("\n## 六、聚类分析\n")
    if not clusters.empty:
        lines.append("已根据颜色、纹理和形状特征完成层次聚类，树状图见 `04_表型特征聚类树状图.png`，分组结果见 `04_品种聚类分组结果.csv`。")
        lines.append("注：PCA 和 t-SNE 使用同一批特征做降维可视化，不能作为聚类有效性的独立验证——")
        lines.append("同一份数据当然会显示分离。聚类的合理性需用留出特征、外部标签或轮廓系数等独立指标评估。")
        cluster_counts = clusters["cluster"].value_counts().sort_index().to_dict()
        lines.append(f"各簇数量为：{cluster_counts}。")

    lines.append("\n## 七、重要祖先/奠基者（含置乱零模型检验）\n")
    lines.append("奠基者评分 = 后代数 × 平均性状保持率。后代数由节点在树中的深度/枢纽位置决定，")
    lines.append("与遗传贡献无关；保持率受性状基率污染。已用置乱检验（500次）评估真实评分是否")
    lines.append("显著高于「仅凭拓扑位置」的预期。")
    if not founders.empty:
        top = founders.iloc[0]
        lines.append(
            f"综合后代数量和性状保持率，当前最重要的祖先可能是 `{top['祖先品种']}`，后代数量为 {int(top['后代数量'])}，"
            f"奠基者评分为 {top['奠基者评分']:.2f}，置乱 p={top.get('零模型p值', 'N/A')}。"
        )
        # 列出显著高于拓扑预期的节点
        sig_nodes = founders[founders.get("评分是否显著高于拓扑预期", "") == "是（p<0.05）"]
        if not sig_nodes.empty:
            lines.append("以下祖先的评分显著高于拓扑随机预期（p<0.05）：")
            for _, r in sig_nodes.iterrows():
                lines.append(f"  - `{r['祖先品种']}`：真实评分={r['奠基者评分']:.2f}，p={r.get('零模型p值', 'N/A'):.4f}")
        else:
            lines.append("所有祖先的评分均未显著高于拓扑随机预期，说明当前「奠基者评分」主要由谱系拓扑位置驱动，")
            lines.append("而非该祖先在遗传上确有超出预期的性状传递力。")
    else:
        lines.append("未形成可排序的祖先结果。")

    lines.append("\n## 八、异常个体\n")
    lines.append("注：孤立森林使用 contamination=0.12，在小样本下会强制标出约 4 个「异常」。")
    lines.append("被标记者可能仅为照片数较少（1张）导致的特征不稳定，不一定是生物学异常。")
    abnormal = anomalies[anomalies["是否异常"] == True] if not anomalies.empty else pd.DataFrame()
    if not abnormal.empty:
        lines.append("孤立森林识别出的异常个体：")
        for _, r in abnormal.head(5).iterrows():
            dist = r["parent_feature_distance"]
            dist_text = "NA" if pd.isna(dist) else f"{dist:.3f}"
            lines.append(
                f"- `{r[NAME_COL]}`：isolation_score={r['isolation_score']:.4f}，亲本特征距离={dist_text}。"
            )
    else:
        lines.append("当前没有检测到明显异常个体。")

    lines.append("\n## 九、分类模型\n")
    if clf_result:
        lines.append(
            f"决策树以 `{cn_col(clf_result['target_col'])}` 为分类目标，宏平均F1分数={clf_result['macro_f1']:.4f}。详细分类报告见 `07_决策树分类结果.txt`。"
        )

    lines.append("\n## 十、可写入报告的简要结论\n")
    lines.append(
        "综合分析可见，在 MituoLoveDragon 的 33 个祖先构成的非随机谱系样本中，"
        "紫红色唇瓣是最稳定的跨代性状，但限于样本量和选择偏差，不能推断为总体遗传规律。"
        "深层代数有条纹的品种多是因为有条纹的野生种（amboinensis、lueddemanniana 等）被放在了深层，"
        "反映的是祖先构成而非「隔代沉默-重显」。"
        "奠基者评分受谱系拓扑位置严重混淆，经置乱检验后多数节点评分不显著。"
        "后续研究应增大独立样本、对颜色做色卡校准、并引入真正的遗传标记进行交叉验证。"
    )

    (out_dir / "分析结论说明.md").write_text("\n".join(lines), encoding="utf-8")


# 主函数
def main() -> None:
    parser = argparse.ArgumentParser(description="朵丽蝶遗传规律分析")
    parser.add_argument("--lineage", default="A族谱1.xlsx", help="族谱 Excel 文件")
    parser.add_argument("--features", default="朵丽蝶_特征总表.csv", help="特征总表 CSV 文件")
    parser.add_argument("--images_zip", default="processed1.zip", help="处理后图片压缩包")
    parser.add_argument("--out", default="分析结果", help="输出目录")
    parser.add_argument("--target", default=TYPE_COL, help="决策树分类目标列，默认 杂交类型；如果有血统列可改为 血统")
    parser.add_argument("--min_support", type=float, default=0.08, help="关联规则最小支持度")
    parser.add_argument("--min_confidence", type=float, default=0.60, help="关联规则最小置信度")
    parser.add_argument("--min_lift", type=float, default=1.50, help="关联规则最小提升度")
    parser.add_argument("--clusters", type=int, default=4, help="聚类数量")
    args = parser.parse_args()

    lineage_path = resolve_input_path(args.lineage, "族谱表", ["A族谱*.xlsx", "*族谱*.xlsx"])
    features_path = resolve_input_path(args.features, "特征总表", ["朵丽蝶_特征总表*.csv", "*特征总表*.csv"])
    images_zip_path = resolve_input_path(args.images_zip, "图片压缩包", ["processed*.zip", "*processed*.zip", "*图片*.zip"])

    out_dir = ensure_dir(args.out)

    print("正在读取并合并三份文件...")
    print(f"族谱表：{lineage_path}")
    print(f"特征表：{features_path}")
    print(f"图片包：{images_zip_path}")
    df = load_and_merge(lineage_path, features_path, images_zip_path, out_dir)

    print("正在计算性状整体基率（变异是否充足）...")
    base_rates = trait_base_rates(df, out_dir)

    print("正在统计代际性状频率...")
    gen_stats = generation_trait_stats(df, out_dir)

    print("正在计算亲本-子代遗传概率...")
    _, inherit_summary = parent_child_inheritance(df, out_dir)
    numeric_parent_similarity(df, out_dir)

    print("正在挖掘性状关联规则...")
    rules = apriori_rules(
        df,
        out_dir,
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
    )

    print("正在进行层次聚类分析...")
    clusters = cluster_analysis(df, out_dir, n_clusters=args.clusters)

    print("正在进行异常个体检测...")
    anomalies = anomaly_detection(df, out_dir)

    print("正在计算重要祖先/奠基者评分...")
    founders = founder_analysis(df, out_dir)

    print("正在训练决策树分类模型...")
    clf_result = decision_tree_classification(df, out_dir, target_col=args.target)

    print("正在生成结论说明...")
    generate_conclusion(
        df=df,
        out_dir=out_dir,
        gen_stats=gen_stats,
        inherit_summary=inherit_summary,
        rules=rules,
        clusters=clusters,
        anomalies=anomalies,
        founders=founders,
        clf_result=clf_result,
        base_rates=base_rates,
    )

    print("\n分析完成！输出文件如下：")
    for p in sorted(out_dir.iterdir()):
        print(" -", p.name)


if __name__ == "__main__":
    main()
