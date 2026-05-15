"""
01_data_exploration.py
余震预测项目 — 第一步：数据探索与清洗
  1) 清洗 USGS 全球 M6.0+ 地震数据  -> usgs_clean.csv
  2) 汇总 20 条测试地震序列统计信息   -> test_eq_stats.csv
  3) 生成 3 张 EDA 图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ──────────────────────────────────────────────
DATA_DIR = r"E:\data_download"
SEQ_DIR = os.path.join(DATA_DIR, "test_eq_data")
OUTPUT_DIR = r"E:\earthquake_project\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USGS_RAW_PATH = os.path.join(DATA_DIR, "usgs_query.csv")
CATALOG_PATH = os.path.join(DATA_DIR, "test_eq_catalog.csv")


# ═══════════════════════════════════════════════════════════
#  第 1 部分：USGS 全球地震数据清洗
# ═══════════════════════════════════════════════════════════
def clean_usgs():
    print("[1/3] 清洗 USGS 全球地震数据 ...")
    raw = pd.read_csv(USGS_RAW_PATH)
    print(f"  原始数据: {raw.shape}")

    df = raw.copy()

    # 解析时间, 提取年月
    df["time"] = pd.to_datetime(df["time"])
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month

    # 选取建模所需列, 按时间降序排列
    cols = ["time", "latitude", "longitude", "depth",
            "mag", "magType", "place", "year", "month"]
    df = df[cols].sort_values("time", ascending=False).reset_index(drop=True)

    out_path = os.path.join(OUTPUT_DIR, "usgs_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"  清洗后: {df.shape} -> {out_path}")
    return df


# ═══════════════════════════════════════════════════════════
#  第 2 部分：测试地震序列统计
# ═══════════════════════════════════════════════════════════
def build_test_eq_stats():
    print("[2/3] 汇总测试地震序列统计 ...")
    cat = pd.read_csv(CATALOG_PATH)

    rows = []
    for _, r in cat.iterrows():
        eq_id = (f"{int(r['Year']):04d}{int(r['Month']):02d}{int(r['Day']):02d}"
                 f"{int(r['Hour']):02d}{int(r['Minute']):02d}{int(r['Second']):02d}")
        seq_path = os.path.join(SEQ_DIR, f"{eq_id}_eq.csv")
        seq = pd.read_csv(seq_path, encoding="utf-8-sig")
        rows.append({
            "eq_id": eq_id,
            "mainshock_mag": r["Mag"],
            "latitude": r["Lat"],
            "longitude": r["Lon"],
            "depth": r["Depth"],
            "source": r["Source"],
            "aftershock_count": len(seq),   # 序列中全部记录数
            "year": int(r["Year"]),
        })

    stats = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "test_eq_stats.csv")
    stats.to_csv(out_path, index=False)
    print(f"  {len(stats)} 条地震 -> {out_path}")
    return stats


# ═══════════════════════════════════════════════════════════
#  第 3 部分：EDA 可视化
# ═══════════════════════════════════════════════════════════

# ---- 图 1: USGS 数据 4 面板 EDA ----
def plot_usgs_eda(df):
    print("[3/3] 绘制 EDA 图表 ...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("USGS 全球6.0级以上地震数据探索性分析（1990-2024）",
                 fontsize=14, fontweight="bold")

    # (1) 震级分布
    ax = axes[0, 0]
    ax.hist(df["mag"], bins=30, color="steelblue", edgecolor="white")
    mean_mag = df["mag"].mean()
    ax.axvline(mean_mag, color="red", ls="--", lw=1.2)
    ax.set_title("震级分布")
    ax.set_xlabel("震级（Ms）")
    ax.set_ylabel("频次")
    ax.legend([f"均值≈{mean_mag:.2f}"], loc="upper right",
              frameon=False, fontsize=9, handlelength=1.5,
              handler_map={str: matplotlib.legend_handler.HandlerBase()})
    # 用 text 添加标注更灵活
    ax.text(mean_mag + 0.05, ax.get_ylim()[1] * 0.92,
            f"均值≈{mean_mag:.2f}", color="red", fontsize=9)

    # (2) 震源深度分布
    ax = axes[0, 1]
    ax.hist(df["depth"], bins=50, color="darkorange", edgecolor="white")
    mean_dep = df["depth"].mean()
    ax.axvline(mean_dep, color="red", ls="--", lw=1.2)
    ax.set_title("震源深度分布")
    ax.set_xlabel("震源深度（km）")
    ax.set_ylabel("频次")
    ax.text(mean_dep + 1, ax.get_ylim()[1] * 0.92,
            f"均值≈{mean_dep:.1f}km", color="red", fontsize=9)

    # (3) 年度频次
    ax = axes[1, 0]
    yearly = df.groupby("year").size()
    ax.bar(yearly.index, yearly.values, color="steelblue", edgecolor="white")
    ax.set_title("年度6.0+级地震频次")
    ax.set_xlabel("年份")
    ax.set_ylabel("地震数")

    # (4) 震级 vs 深度（按年份着色）
    ax = axes[1, 1]
    sc = ax.scatter(df["depth"], df["mag"], c=df["year"],
                    cmap="viridis", s=8, alpha=0.6)
    fig.colorbar(sc, ax=ax, label="年份")
    ax.set_title("震级 vs 深度（按年份着色）")
    ax.set_xlabel("震源深度（km）")
    ax.set_ylabel("震级")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "01_usgs_eda.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---- 图 2: 全球地震分布 ----
def plot_global_distribution(df):
    fig, ax = plt.subplots(figsize=(14, 7))
    sc = ax.scatter(df["longitude"], df["latitude"],
                    c=df["mag"], cmap="YlOrRd",
                    s=(df["mag"] - 5) ** 2.5 * 3,
                    alpha=0.5, edgecolors="none")
    fig.colorbar(sc, ax=ax, label="震级（Mw）", shrink=0.6)
    ax.set_title("全球6.0+级地震分布（1990-2024）（颜色和大小表示震级）",
                 fontsize=13)
    ax.set_xlabel("经度")
    ax.set_ylabel("纬度")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "02_global_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ---- 图 3: 测试地震序列分析 ----
def plot_test_earthquakes(stats):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # 左: 余震数量柱状图
    ax = axes[0]
    x = np.arange(len(stats))
    ax.bar(x, stats["aftershock_count"], color="steelblue")
    ax.set_title("20个测试震例的余震数量", fontsize=13)
    ax.set_xlabel("震例编号")
    ax.set_ylabel("余震数量")
    ax.set_xticks(x)

    # 右: 主震震级 vs 余震数量 + 拟合
    ax = axes[1]
    M = stats["mainshock_mag"].values
    N = stats["aftershock_count"].values
    ax.scatter(M, N, s=60, color="red", zorder=3)

    # 线性拟合: log10(N) = a*M + b
    logN = np.log10(N.astype(float))
    coeffs = np.polyfit(M, logN, 1)
    a, b = coeffs
    M_fit = np.linspace(M.min() - 0.2, M.max() + 0.2, 100)
    N_fit = 10 ** (a * M_fit + b)
    ax.plot(M_fit, N_fit, "b--", lw=1.5,
            label=f"拟合：log10(N)={a:.2f}*M + {b:.2f}")

    ax.set_title("主震震级 vs 余震数量", fontsize=13)
    ax.set_xlabel("主震震级")
    ax.set_ylabel("余震数量")
    ax.legend(fontsize=10)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "03_test_earthquakes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out}")


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  余震预测项目 — 数据探索与清洗")
    print("=" * 60)

    usgs = clean_usgs()
    stats = build_test_eq_stats()

    plot_usgs_eda(usgs)
    plot_global_distribution(usgs)
    plot_test_earthquakes(stats)

    print("\n全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
