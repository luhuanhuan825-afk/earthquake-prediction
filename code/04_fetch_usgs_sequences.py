"""
04_fetch_usgs_sequences.py
从 USGS FDSN API 获取历史余震序列数据

输入: output/usgs_clean.csv (4140条 M6.0+ 主震)
API : https://earthquake.usgs.gov/fdsnws/event/1/query
策略: 震后168小时内、主震周边3度范围、M2.0+ 余震
输出: E:/data_download/usgs_sequences/{eq_id}_seq.csv
"""

import os, time, warnings
import requests
import pandas as pd
from io import StringIO
from datetime import timedelta

warnings.filterwarnings("ignore")

# ── 配置 ──────────────────────────────────────────────────
USGS_PATH  = r"E:/earthquake_project/output/usgs_clean.csv"
OUT_DIR    = r"E:/data_download/usgs_sequences"
os.makedirs(OUT_DIR, exist_ok=True)

API_URL     = "https://earthquake.usgs.gov/fdsnws/event/1/query"
N_SAMPLE    = 50        # 首批测试数量
RADIUS_DEG  = 3.0       # 搜索半径（度）
MIN_MAG     = 2.0       # 余震最低震级
HOURS       = 168       # 时间窗口（小时）
DELAY_SEC   = 1.0       # 请求间隔（秒）
TIMEOUT_SEC = 45        # 单次超时
MAX_RETRIES = 3         # 失败重试次数
API_LIMIT   = 20000     # USGS 单次返回上限

# 保留的输出列（与 test_eq_data 格式对齐）
KEEP_COLS = ["time", "latitude", "longitude", "depth", "mag", "magType", "place"]


def make_eq_id(dt_utc):
    """Pandas Timestamp → YYYYMMDDHHMMSS (取整秒, UTC)"""
    ts = dt_utc.floor("s")
    return ts.strftime("%Y%m%d%H%M%S")


def fetch_one(lat, lon, start_ts, end_ts):
    """
    调用 USGS FDSN API, 返回 CSV 文本 / "" / None
      ""   → 无事件 (HTTP 204 or 空内容)
      None → 全部重试失败
    """
    params = {
        "format":       "csv",
        "starttime":    start_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "endtime":      end_ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "latitude":     f"{lat:.4f}",
        "longitude":    f"{lon:.4f}",
        "maxradius":    RADIUS_DEG,
        "minmagnitude": MIN_MAG,
        "orderby":      "time",
        "limit":        API_LIMIT,
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(API_URL, params=params, timeout=TIMEOUT_SEC)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code == 204:
                return ""
            # 4xx 一般是参数问题，不重试
            if 400 <= resp.status_code < 500:
                print(f" HTTP {resp.status_code}", end="", flush=True)
                return None
            print(f" HTTP{resp.status_code}(retry{attempt+1})", end="", flush=True)
        except requests.exceptions.Timeout:
            print(f" timeout(retry{attempt+1})", end="", flush=True)
        except requests.exceptions.ConnectionError:
            print(f" conn_err(retry{attempt+1})", end="", flush=True)
        except Exception as e:
            print(f" err:{e}(retry{attempt+1})", end="", flush=True)
        time.sleep(DELAY_SEC * (attempt + 2))
    return None


def parse_csv(csv_text):
    """解析 API 返回的 CSV, 提取关键列"""
    df = pd.read_csv(StringIO(csv_text))
    present = [c for c in KEEP_COLS if c in df.columns]
    return df[present].copy()


def main():
    print("=" * 65)
    print("  USGS 余震序列抓取 (测试 50 个震例)")
    print("=" * 65)

    # ── 1. 加载主震目录 ────────────────────────────────────
    usgs = pd.read_csv(USGS_PATH)
    usgs["time_ts"] = pd.to_datetime(usgs["time"], format="mixed", utc=True)

    # 按震级降序取前 N_SAMPLE 个（震级越大余震越丰富）
    sample = usgs.nlargest(N_SAMPLE, "mag").reset_index(drop=True)

    print(f"\n选取范围: M{sample['mag'].min():.1f}–M{sample['mag'].max():.1f}, "
          f"{sample['year'].min()}–{sample['year'].max()} 年")
    print(f"输出目录: {OUT_DIR}")
    print(f"窗口参数: ±{RADIUS_DEG}° / {HOURS}h / M≥{MIN_MAG}\n")

    # ── 2. 逐一抓取 ────────────────────────────────────────
    log = []
    t_global_start = time.time()

    for idx, row in sample.iterrows():
        seq_num  = idx + 1
        lat      = float(row["latitude"])
        lon      = float(row["longitude"])
        mag      = float(row["mag"])
        start_ts = row["time_ts"]
        end_ts   = start_ts + timedelta(hours=HOURS)
        eq_id    = make_eq_id(start_ts)
        place    = str(row["place"])[:45]

        out_path = os.path.join(OUT_DIR, f"{eq_id}_seq.csv")
        prefix   = f"[{seq_num:2d}/{N_SAMPLE}] {eq_id} M{mag:.1f}"

        # ── 跳过已有文件 ──────────────────────────────
        if os.path.exists(out_path):
            n_cached = max(0, len(open(out_path).readlines()) - 1)
            print(f"{prefix}  (cached {n_cached} rows, skip)")
            log.append({"eq_id": eq_id, "mag": mag, "lat": lat, "lon": lon,
                         "year": int(row["year"]), "n_events": n_cached,
                         "status": "cached"})
            continue

        # ── 调用 API ──────────────────────────────────
        place_safe = place.encode("gbk", errors="replace").decode("gbk")
        print(f"{prefix}  {place_safe} ...", end=" ", flush=True)
        t0      = time.time()
        csv_txt = fetch_one(lat, lon, start_ts, end_ts)
        elapsed = time.time() - t0

        if csv_txt is None:
            print(f" FAILED ({elapsed:.1f}s)")
            log.append({"eq_id": eq_id, "mag": mag, "lat": lat, "lon": lon,
                         "year": int(row["year"]), "n_events": 0, "status": "failed"})
            time.sleep(DELAY_SEC)
            continue

        if csv_txt.strip() == "" or len(csv_txt.strip().splitlines()) <= 1:
            print(f" 0 events ({elapsed:.1f}s)")
            # 写空头文件，下次跳过
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(",".join(KEEP_COLS) + "\n")
            log.append({"eq_id": eq_id, "mag": mag, "lat": lat, "lon": lon,
                         "year": int(row["year"]), "n_events": 0, "status": "empty"})
            time.sleep(DELAY_SEC)
            continue

        # ── 解析并保存 ────────────────────────────────
        try:
            df = parse_csv(csv_txt)
            n  = len(df)
            df.to_csv(out_path, index=False)

            mag_range = f"[{df['mag'].min():.1f}–{df['mag'].max():.1f}]" \
                        if "mag" in df.columns and n > 0 else ""
            truncated = " ⚠TRUNCATED" if n >= API_LIMIT - 100 else ""
            print(f" {n:5d} events {mag_range} ({elapsed:.1f}s){truncated}")

            log.append({"eq_id": eq_id, "mag": mag, "lat": lat, "lon": lon,
                         "year": int(row["year"]), "n_events": n,
                         "status": "truncated" if truncated else "ok"})
        except Exception as e:
            print(f" parse error: {e}")
            log.append({"eq_id": eq_id, "mag": mag, "lat": lat, "lon": lon,
                         "year": int(row["year"]), "n_events": 0,
                         "status": "parse_error"})

        time.sleep(DELAY_SEC)

    # ── 3. 汇总统计 ────────────────────────────────────────
    log_df = pd.DataFrame(log)
    log_path = os.path.join(OUT_DIR, "_fetch_log.csv")
    log_df.to_csv(log_path, index=False)

    total_elapsed = time.time() - t_global_start
    ok_mask = log_df["status"].isin(["ok", "cached", "truncated"])

    print("\n" + "=" * 65)
    print(f"  完成! 用时 {total_elapsed/60:.1f} 分钟")
    print(f"  成功: {ok_mask.sum():2d}/{N_SAMPLE}  "
          f"失败: {(log_df['status']=='failed').sum()}  "
          f"空序列: {(log_df['status']=='empty').sum()}")
    if ok_mask.any():
        ok_df = log_df[ok_mask]
        print(f"  余震总数: {ok_df['n_events'].sum():,}")
        print(f"  每震均值: {ok_df['n_events'].mean():.0f}  "
              f"最多: {ok_df['n_events'].max()}  最少: {ok_df['n_events'].min()}")
    print(f"  日志: {log_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
