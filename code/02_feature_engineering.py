"""
02_feature_engineering.py
余震预测项目 - 特征工程
从每个地震序列中提取用于建模的特征
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR = r"E:\data_download"
SEQ_DIR = os.path.join(DATA_DIR, "test_eq_data")
OUTPUT_DIR = r"E:\earthquake_project\output"

CATALOG_PATH = os.path.join(DATA_DIR, "test_eq_catalog.csv")
STATS_PATH = os.path.join(OUTPUT_DIR, "test_eq_stats.csv")


# ── 工具函数 ──────────────────────────────────────────────
def haversine(lon1, lat1, lon2, lat2):
    """计算两点间的球面距离 (km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371 * 2 * atan2(sqrt(a), sqrt(1 - a))


def estimate_b_value(mags, mc=None):
    """用最大似然法估计 Gutenberg-Richter b 值"""
    if mc is None:
        mc = np.min(mags)
    mags_above = mags[mags >= mc]
    if len(mags_above) < 5:
        return np.nan
    delta_m = 0.1  # 震级精度
    mean_m = np.mean(mags_above)
    b = 1.0 / (np.log(10) * (mean_m - mc + delta_m / 2))
    return b


def estimate_mc(mags, method="maxc"):
    """估计完备震级 Mc (最大曲率法)"""
    if len(mags) < 10:
        return np.min(mags) if len(mags) > 0 else np.nan
    bins = np.arange(np.floor(mags.min() * 10) / 10,
                     np.ceil(mags.max() * 10) / 10 + 0.1, 0.1)
    counts, edges = np.histogram(mags, bins=bins)
    if len(counts) == 0:
        return np.nan
    mc_idx = np.argmax(counts)
    return round(edges[mc_idx], 1)


# ── 主流程 ────────────────────────────────────────────────
def load_catalog():
    """加载主震目录"""
    cat = pd.read_csv(CATALOG_PATH)
    cat["eq_id"] = cat.apply(
        lambda r: f"{r['Year']:04d}{r['Month']:02d}{r['Day']:02d}"
                  f"{r['Hour']:02d}{r['Minute']:02d}{r['Second']:02d}",
        axis="columns",
    )
    cat["mainshock_time"] = pd.to_datetime(
        cat[["Year", "Month", "Day", "Hour", "Minute", "Second"]]
        .rename(columns={"Year": "year", "Month": "month", "Day": "day",
                         "Hour": "hour", "Minute": "minute", "Second": "second"})
    )
    return cat


def load_sequence(eq_id):
    """加载某个地震的余震序列"""
    path = os.path.join(SEQ_DIR, f"{eq_id}_eq.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    return df


def extract_features(cat_row, seq_df):
    """从单条主震 + 余震序列中提取特征"""
    eq_id = cat_row["eq_id"]
    main_lon = cat_row["Lon"]
    main_lat = cat_row["Lat"]
    main_mag = cat_row["Mag"]
    main_depth = cat_row["Depth"]
    main_time = cat_row["mainshock_time"]

    # 排除主震本身 (第一条通常是主震或最大前震)
    aftershocks = seq_df.iloc[1:].copy()

    feat = {
        "eq_id": eq_id,
        "mainshock_mag": main_mag,
        "mainshock_depth": main_depth,
        "mainshock_lat": main_lat,
        "mainshock_lon": main_lon,
    }

    n = len(aftershocks)
    feat["aftershock_count"] = n

    if n == 0:
        return feat

    mags = aftershocks["Mag"].values.astype(float)
    depths = aftershocks["Depth"].values.astype(float)

    # ── 震级统计 ──
    feat["mag_mean"] = np.mean(mags)
    feat["mag_std"] = np.std(mags)
    feat["mag_max"] = np.max(mags)
    feat["mag_min"] = np.min(mags)
    feat["mag_median"] = np.median(mags)
    feat["mag_range"] = feat["mag_max"] - feat["mag_min"]
    feat["mag_diff_main_max"] = main_mag - feat["mag_max"]  # Bath's law 相关

    # ── 深度统计 ──
    feat["depth_mean"] = np.nanmean(depths)
    feat["depth_std"] = np.nanstd(depths)
    feat["depth_max"] = np.nanmax(depths)

    # ── 空间特征 ──
    dists = aftershocks.apply(
        lambda r: haversine(main_lon, main_lat, r["Lon"], r["Lat"]), axis=1
    ).values
    feat["dist_mean"] = np.mean(dists)
    feat["dist_std"] = np.std(dists)
    feat["dist_max"] = np.max(dists)
    feat["dist_median"] = np.median(dists)
    feat["spatial_spread"] = np.percentile(dists, 90)  # 90% 余震覆盖范围

    # ── 时间特征 ──
    dt = (aftershocks["datetime"] - main_time).dt.total_seconds() / 3600  # 小时
    dt_pos = dt[dt > 0]
    if len(dt_pos) > 0:
        feat["time_span_hours"] = dt_pos.max()
        feat["time_mean_hours"] = dt_pos.mean()
        # 前 24 小时余震数
        feat["count_24h"] = int((dt_pos <= 24).sum())
        # 前 72 小时余震数
        feat["count_72h"] = int((dt_pos <= 72).sum())
        # 前 24 小时占比
        feat["ratio_24h"] = feat["count_24h"] / n if n > 0 else 0
    else:
        feat["time_span_hours"] = np.nan
        feat["time_mean_hours"] = np.nan
        feat["count_24h"] = 0
        feat["count_72h"] = 0
        feat["ratio_24h"] = 0

    # ── 频次-震级分布特征 ──
    mc = estimate_mc(mags)
    feat["mc"] = mc
    feat["b_value"] = estimate_b_value(mags, mc)

    # 震级区间计数
    feat["n_mag_ge_4"] = int((mags >= 4.0).sum())
    feat["n_mag_ge_5"] = int((mags >= 5.0).sum())
    feat["n_mag_ge_6"] = int((mags >= 6.0).sum())

    return feat


def main():
    print("=" * 60)
    print("  余震预测 — 特征工程")
    print("=" * 60)

    # 1. 加载目录
    cat = load_catalog()
    print(f"\n加载主震目录: {len(cat)} 条地震")

    # 2. 逐条提取特征
    features = []
    for _, row in cat.iterrows():
        eq_id = row["eq_id"]
        print(f"  处理 {eq_id} (M{row['Mag']}) ...", end=" ")
        seq = load_sequence(eq_id)
        feat = extract_features(row, seq)
        features.append(feat)
        print(f"余震 {feat['aftershock_count']} 条, "
              f"b={feat.get('b_value', 'N/A')}")

    # 3. 合并并保存
    df_feat = pd.DataFrame(features)

    out_path = os.path.join(OUTPUT_DIR, "features.csv")
    df_feat.to_csv(out_path, index=False)
    print(f"\n特征矩阵已保存: {out_path}")
    print(f"  形状: {df_feat.shape}")
    print(f"\n特征列表:")
    for c in df_feat.columns:
        print(f"  - {c}")

    print("\n前 5 行预览:")
    print(df_feat.head().to_string())
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
