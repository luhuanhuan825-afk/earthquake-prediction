"""
03_model_training.py
余震预测项目 — 第三步：模型训练与预测
  - 样本量仅 20 条，采用留一交叉验证（LOO-CV）
  - 目标变量：log10(aftershock_count)，回归后反变换
  - 对比 4 个模型：Ridge / Lasso / RandomForest / GradientBoosting
  - 输出：评估指标 CSV、特征重要性图、预测对比图、最优模型持久化
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 路径 ──────────────────────────────────────────────────
FEAT_PATH = r"E:\earthquake_project\output\features.csv"
OUTPUT_DIR = r"E:\earthquake_project\output"
MODEL_DIR = r"E:\earthquake_project\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 用于预测的特征列（排除 ID、目标变量及全序列聚合特征以防数据泄漏）
FEATURE_COLS = [
    "mainshock_mag", "mainshock_depth", "mainshock_lat", "mainshock_lon",
    "mag_mean", "mag_std", "mag_max", "mag_min", "mag_median",
    "mag_range", "mag_diff_main_max",
    "depth_mean", "depth_std", "depth_max",
    "dist_mean", "dist_std", "dist_max", "dist_median", "spatial_spread",
    "time_span_hours", "time_mean_hours",
    "count_24h", "count_72h", "ratio_24h",
    "mc", "b_value",
]
TARGET_COL = "aftershock_count"

# ── 模型定义 ──────────────────────────────────────────────
MODELS = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Lasso(alpha=0.1, max_iter=5000)),
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_features="sqrt",
        min_samples_leaf=2, random_state=42,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=100, max_depth=3,
        learning_rate=0.1, subsample=0.8, random_state=42,
    ),
}


# ── LOO-CV 评估 ───────────────────────────────────────────
def loo_evaluate(X, y_log, model):
    """返回 LOO-CV 在 log 空间的预测值数组"""
    loo = LeaveOneOut()
    preds_log = np.zeros(len(y_log))
    import copy
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y_log[train_idx]
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        preds_log[test_idx] = m.predict(X_te)
    return preds_log


def compute_metrics(y_true, y_pred, label=""):
    """在原始尺度上计算 MAE / RMSE / R²"""
    y_true_orig = np.power(10, y_true)
    y_pred_orig = np.power(10, y_pred)
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2   = r2_score(y_true_orig, y_pred_orig)
    mae_log  = mean_absolute_error(y_true, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    r2_log   = r2_score(y_true, y_pred)
    print(f"  {label:20s}  MAE={mae:7.1f}  RMSE={rmse:8.1f}  R2={r2:+.3f}"
          f"  |  logMAE={mae_log:.3f}  logR2={r2_log:+.3f}")
    return {
        "model": label,
        "MAE": round(mae, 1), "RMSE": round(rmse, 1), "R2": round(r2, 4),
        "logMAE": round(mae_log, 4), "logRMSE": round(rmse_log, 4),
        "logR2": round(r2_log, 4),
    }


# ── 特征重要性（树模型）────────────────────────────────────
def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(idx)), imp[idx][::-1], color="steelblue")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel("特征重要性")
    ax.set_title(f"{model_name} — 特征重要性（Top15）")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"04_{model_name.lower()}_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  特征重要性图 -> {out}")


# ── 预测对比图 ────────────────────────────────────────────
def plot_predictions(df, results, best_name):
    best = next(r for r in results if r["model"] == best_name)
    y_true = np.power(10, df["log_count"].values)
    y_pred = np.power(10, df[f"pred_{best_name}"].values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左：真实 vs 预测散点
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=60, color="steelblue", zorder=3)
    lim = max(y_true.max(), y_pred.max()) * 1.1
    ax.plot([0, lim], [0, lim], "r--", lw=1.2)
    for i, row in df.iterrows():
        ax.annotate(str(row["eq_id"])[:8],
                    (y_true[i], y_pred[i]),
                    fontsize=6, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("真实余震数")
    ax.set_ylabel("预测余震数")
    ax.set_title(f"{best_name} — 真实 vs 预测（LOO-CV）")

    # 右：各模型 logMAE 对比柱状图
    ax = axes[1]
    names = [r["model"] for r in results]
    log_maes = [r["logMAE"] for r in results]
    colors = ["gold" if n == best_name else "steelblue" for n in names]
    bars = ax.bar(names, log_maes, color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_ylabel("logMAE（LOO-CV）")
    ax.set_title("各模型 logMAE 对比")
    ax.set_ylim(0, max(log_maes) * 1.3)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "05_model_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  模型对比图  -> {out}")


# ── 主流程 ────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  余震预测 — 模型训练与评估")
    print("=" * 65)

    # 1. 加载特征
    df = pd.read_csv(FEAT_PATH)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    print(f"\n有效样本: {len(df)} 条")

    X = df[FEATURE_COLS].values.astype(float)
    y_orig = df[TARGET_COL].values.astype(float)
    y_log = np.log10(y_orig)
    df["log_count"] = y_log

    # 2. LOO-CV 评估所有模型
    print(f"\nLOO-CV 评估（目标: log10(aftershock_count)）\n" + "-" * 65)
    metrics_list = []
    for name, model in MODELS.items():
        preds_log = loo_evaluate(X, y_log, model)
        df[f"pred_{name}"] = preds_log
        m = compute_metrics(y_log, preds_log, label=name)
        metrics_list.append(m)

    # 3. 确定最优模型（按 logMAE 最小）
    best_name = min(metrics_list, key=lambda x: x["logMAE"])["model"]
    print(f"\n最优模型: {best_name}")

    # 4. 用全量数据拟合最优模型并保存
    import copy
    best_model = copy.deepcopy(MODELS[best_name])
    best_model.fit(X, y_log)
    model_path = os.path.join(MODEL_DIR, f"best_{best_name.lower()}.pkl")
    joblib.dump(best_model, model_path)
    print(f"最优模型已保存: {model_path}")

    # 也保存特征列名
    feat_meta = {"feature_cols": FEATURE_COLS, "best_model": best_name}
    joblib.dump(feat_meta, os.path.join(MODEL_DIR, "feature_meta.pkl"))

    # 5. 特征重要性
    plot_feature_importance(best_model, FEATURE_COLS, best_name)

    # 6. 所有树模型的特征重要性
    for name, model in MODELS.items():
        if name != best_name and hasattr(model, "feature_importances_"):
            m_full = copy.deepcopy(model)
            m_full.fit(X, y_log)
            plot_feature_importance(m_full, FEATURE_COLS, name)

    # 7. 预测对比图
    plot_predictions(df, metrics_list, best_name)

    # 8. 保存评估指标
    metrics_df = pd.DataFrame(metrics_list)
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n评估指标已保存: {metrics_path}")
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
