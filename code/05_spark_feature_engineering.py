"""
05_spark_feature_engineering.py
大数据课程实训 4.2 — PySpark 特征工程（HDFS + Spark）

技术栈体现:
  ▸ Hadoop HDFS : 原始序列数据 / 特征矩阵 存储
  ▸ PySpark     : 分布式读取、DataFrame API 特征计算
  ▸ Spark SQL   : createOrReplaceTempView + SQL 查询 + 窗口函数
  ▸ Spark MLlib : VectorAssembler + StandardScaler + Pipeline
  ▸ UDF         : Haversine 距离、Gutenberg-Richter b 值、Mc 完备震级

数据流:
  本地 usgs_sequences/  →  HDFS /earthquake/usgs_sequences/
  本地 usgs_clean.csv   →  HDFS /earthquake/catalog/
  HDFS 序列 + 目录      →  PySpark 特征提取
  特征矩阵              →  HDFS /earthquake/features/ + 本地 output/
"""

import os, sys, time, subprocess, socket, warnings
import numpy as np
warnings.filterwarnings("ignore")

# Windows 中文系统强制 UTF-8 输出，避免 UnicodeEncodeError
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════════════════════════════
# 0. 环境变量（必须在 PySpark 导入之前设置）
# ══════════════════════════════════════════════════════════════════════
JAVA_HOME    = r"C:\Java\jdk11"
HADOOP_HOME  = r"E:\hadoop"
HADOOP_CONF  = r"E:\hadoop\etc\hadoop"

os.environ["JAVA_HOME"]      = JAVA_HOME
os.environ["HADOOP_HOME"]    = HADOOP_HOME
os.environ["HADOOP_CONF_DIR"]= HADOOP_CONF

# 使用系统 Spark 3.5.1（与已安装的 pyspark 3.5.1 版本一致）
# 不要删除 SPARK_HOME，否则 pyspark.resource 等子模块会找不到
os.environ["SPARK_HOME"] = r"E:\spark-3.5.1-bin-hadoop3\spark"

# 指定 Python 解释器（保持一致）
os.environ["PYSPARK_PYTHON"]        = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ══════════════════════════════════════════════════════════════════════
# 1. PySpark 导入
# ══════════════════════════════════════════════════════════════════════
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# ── 路径配置 ──────────────────────────────────────────────────────────
HDFS_ROOT     = "hdfs://localhost:9000"
HDFS_SEQ_DIR  = f"{HDFS_ROOT}/earthquake/usgs_sequences"
HDFS_CAT_PATH = f"{HDFS_ROOT}/earthquake/catalog/usgs_clean.csv"
HDFS_FEAT_DIR = f"{HDFS_ROOT}/earthquake/features"

LOCAL_SEQ_DIR  = r"E:\data_download\usgs_sequences"
LOCAL_CAT_PATH = r"E:\earthquake_project\output\usgs_clean.csv"
LOCAL_OUT_DIR  = r"E:\earthquake_project\output"
HDFS_LOG_DIR   = r"E:\hadoop\logs"
os.makedirs(LOCAL_OUT_DIR, exist_ok=True)
os.makedirs(HDFS_LOG_DIR,  exist_ok=True)

# VectorAssembler 输入特征列（不含目标变量 aftershock_count）
VECTOR_FEATURE_COLS = [
    "mainshock_mag", "mainshock_depth",
    "mag_mean", "mag_std", "mag_max", "mag_min", "mag_median",
    "mag_range", "mag_diff_main_max",
    "depth_mean", "depth_std", "depth_max",
    "dist_mean", "dist_std", "dist_max", "dist_median", "spatial_spread",
    "time_span_hours", "time_mean_hours",
    "count_24h", "count_72h", "ratio_24h",
    "b_value", "mc",
    "n_mag_ge_4", "n_mag_ge_5", "n_mag_ge_6",
]

# ══════════════════════════════════════════════════════════════════════
# 2. HDFS 守护进程管理
# ══════════════════════════════════════════════════════════════════════
HDFS_CMD = os.path.join(HADOOP_HOME, "bin", "hdfs.cmd")
_hdfs_procs = []   # 全局，供 finally 块清理


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_port(host: str, port: int, seconds: int = 90, label: str = "") -> bool:
    deadline = time.time() + seconds
    dots = 0
    while time.time() < deadline:
        if _port_open(host, port):
            print(f"\r  {label} 就绪 ({'.' * dots})           ")
            return True
        print(f"\r  等待 {label} 端口 {port}{'.' * (dots % 6 + 1)}   ", end="", flush=True)
        dots += 1
        time.sleep(3)
    print()
    return False


def _remove_lock(path: str):
    try:
        os.remove(path)
        print(f"  已清理残留 lock: {path}")
    except FileNotFoundError:
        pass


def _jps_running() -> set:
    """返回当前运行的 JVM 进程名集合（如 {'NameNode', 'DataNode'}）"""
    try:
        r = subprocess.run(
            [r"C:\Java\jdk11\bin\jps.exe"],
            capture_output=True, text=True, timeout=10
        )
        names = set()
        for line in r.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                names.add(parts[1])
        return names
    except Exception:
        return set()


def start_hdfs() -> bool:
    """
    启动 HDFS NameNode + DataNode（逐一检查，已运行则跳过）。
    返回 True 表示 HDFS 就绪，False 表示超时失败。
    """
    global _hdfs_procs

    print("\n" + "═" * 65)
    print("  Step-0  HDFS 守护进程管理")
    print("═" * 65)

    running = _jps_running()
    print(f"  当前 JVM 进程: {running or '(无)'}")

    nn_running = "NameNode" in running
    dn_running = "DataNode" in running

    if nn_running and dn_running:
        print("  NameNode + DataNode 均已运行 [OK]")
        return True

    # 清理残留 lock 文件
    if not nn_running:
        _remove_lock(rf"{HADOOP_HOME}\data\namenode\in_use.lock")
    if not dn_running:
        _remove_lock(rf"{HADOOP_HOME}\data\datanode\in_use.lock")

    env = os.environ.copy()
    env["JAVA_HOME"]   = JAVA_HOME
    env["HADOOP_HOME"] = HADOOP_HOME

    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP

    # ── 启动 NameNode（若未运行）─────────────────────────────
    if not nn_running:
        nn_log_path = os.path.join(HDFS_LOG_DIR, "namenode_stdout.log")
        print(f"\n  [HDFS] 启动 NameNode... (日志: {nn_log_path})")
        nn_log = open(nn_log_path, "w", encoding="utf-8", errors="replace")
        nn_proc = subprocess.Popen(
            [HDFS_CMD, "namenode"],
            stdout=nn_log, stderr=subprocess.STDOUT,
            env=env, creationflags=flags,
        )
        _hdfs_procs.append(nn_proc)
        if not wait_port("localhost", 9000, seconds=90, label="NameNode"):
            print("  [ERROR] NameNode 超时，日志:", nn_log_path)
            return False
        print(f"  NameNode PID: {nn_proc.pid}")
    else:
        print("  NameNode 已运行，跳过启动")

    # ── 启动 DataNode（若未运行）─────────────────────────────
    if not dn_running:
        dn_log_path = os.path.join(HDFS_LOG_DIR, "datanode_stdout.log")
        print(f"\n  [HDFS] 启动 DataNode... (日志: {dn_log_path})")
        dn_log = open(dn_log_path, "w", encoding="utf-8", errors="replace")
        dn_proc = subprocess.Popen(
            [HDFS_CMD, "datanode"],
            stdout=dn_log, stderr=subprocess.STDOUT,
            env=env, creationflags=flags,
        )
        _hdfs_procs.append(dn_proc)
        print("  等待 DataNode 注册到 NameNode", end="", flush=True)
        for _ in range(15):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" [OK]")
        print(f"  DataNode PID: {dn_proc.pid}")
    else:
        print("  DataNode 已运行，跳过启动")

    return True


def stop_hdfs():
    """终止本脚本启动的 HDFS 进程"""
    global _hdfs_procs
    for proc in _hdfs_procs:
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True
                    )
                else:
                    proc.terminate()
        except Exception as e:
            print(f"  [WARN] 停止进程异常: {e}")
    _hdfs_procs.clear()
    print("  HDFS 进程已停止")


# ══════════════════════════════════════════════════════════════════════
# 3. HDFS 文件操作工具
# ══════════════════════════════════════════════════════════════════════
def run_hdfs(args: list, check: bool = True) -> str:
    """执行 hdfs dfs 命令，返回 stdout 文本"""
    env = os.environ.copy()
    env["JAVA_HOME"]   = JAVA_HOME
    env["HADOOP_HOME"] = HADOOP_HOME
    cmd = [HDFS_CMD, "dfs"] + args
    print(f"  [HDFS] {' '.join(cmd[-4:])}")   # 只打印后几段避免路径太长
    result = subprocess.run(cmd, capture_output=True, text=True,
                            errors="replace", env=env)
    if result.stdout.strip():
        print(f"         → {result.stdout.strip()[:120]}")
    if result.returncode != 0 and check:
        print(f"  [WARN] {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def upload_to_hdfs():
    """上传 usgs_sequences/ 和 usgs_clean.csv 到 HDFS"""
    print("\n" + "═" * 65)
    print("  Step-1  上传数据到 HDFS")
    print("═" * 65)

    # 创建目录
    for d in ["/earthquake", "/earthquake/usgs_sequences",
              "/earthquake/catalog", "/earthquake/features"]:
        run_hdfs(["-mkdir", "-p", d], check=False)

    # 先获取 HDFS 上已有的文件列表，用于增量判断
    hdfs_env = {**os.environ, "JAVA_HOME": JAVA_HOME, "HADOOP_HOME": HADOOP_HOME}
    ls_result = subprocess.run(
        [HDFS_CMD, "dfs", "-ls", "/earthquake/usgs_sequences/"],
        capture_output=True, text=True, errors="replace", env=hdfs_env
    )
    existing_hdfs = set()
    for line in ls_result.stdout.splitlines():
        # 每行格式: "-rw-r--r--  1 ... /earthquake/usgs_sequences/xxx_seq.csv"
        if "_seq.csv" in line or "_fetch_log.csv" in line:
            fname = line.strip().split()[-1].split("/")[-1]
            existing_hdfs.add(fname)

    seq_files = [f for f in os.listdir(LOCAL_SEQ_DIR) if f.endswith(".csv")]
    print(f"\n  本地序列文件: {len(seq_files)} 个  HDFS已有: {len(existing_hdfs)} 个")

    uploaded = skipped = failed = 0
    for fname in seq_files:
        if fname in existing_hdfs:
            skipped += 1
            continue
        local  = os.path.join(LOCAL_SEQ_DIR, fname)
        target = f"/earthquake/usgs_sequences/{fname}"
        result = subprocess.run(
            [HDFS_CMD, "dfs", "-put", local, target],
            capture_output=True, text=True, errors="replace", env=hdfs_env
        )
        if result.returncode == 0:
            uploaded += 1
        else:
            failed += 1
            if failed <= 3:   # 只打印前几个错误
                print(f"  [WARN] 上传失败 {fname}: {result.stderr.strip()[:100]}")

    print(f"  上传: {uploaded} 个  跳过: {skipped} 个  失败: {failed} 个")

    # 上传主震目录
    run_hdfs(["-put", "-f", LOCAL_CAT_PATH,
              "/earthquake/catalog/usgs_clean.csv"], check=False)
    print("  usgs_clean.csv → HDFS /earthquake/catalog/ ✓")

    # 确认
    out = run_hdfs(["-ls", "/earthquake/usgs_sequences/"], check=False)
    n = sum(1 for l in out.splitlines() if "_seq.csv" in l)
    print(f"  HDFS 序列文件确认: {n} 个")


# ══════════════════════════════════════════════════════════════════════
# 4. 自定义 UDF
# ══════════════════════════════════════════════════════════════════════
@F.udf(returnType=DoubleType())
def haversine_udf(main_lat, main_lon, aft_lat, aft_lon):
    """球面 Haversine 距离 (km)"""
    from math import radians, sin, cos, sqrt, atan2
    if None in (main_lat, main_lon, aft_lat, aft_lon):
        return None
    R = 6371.0
    lat1, lon1 = radians(float(main_lat)), radians(float(main_lon))
    lat2, lon2 = radians(float(aft_lat)),  radians(float(aft_lon))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return float(R * 2 * atan2(sqrt(a), sqrt(1 - a)))


@F.udf(returnType=DoubleType())
def b_value_udf(mags_list):
    """Gutenberg-Richter b 值（最大似然法）"""
    if not mags_list or len(mags_list) < 5:
        return None
    import numpy as np
    mags = np.array([float(m) for m in mags_list if m is not None])
    if len(mags) < 5:
        return None
    mc       = float(np.min(mags))
    mean_m   = float(np.mean(mags[mags >= mc]))
    delta_m  = 0.1
    return float(1.0 / (np.log(10) * (mean_m - mc + delta_m / 2)))


@F.udf(returnType=DoubleType())
def mc_udf(mags_list):
    """最大曲率法估计完备震级 Mc"""
    if not mags_list or len(mags_list) < 10:
        return None
    import numpy as np
    mags = np.array([float(m) for m in mags_list if m is not None])
    bins = np.arange(np.floor(mags.min() * 10) / 10,
                     np.ceil(mags.max() * 10) / 10 + 0.1, 0.1)
    if len(bins) < 2:
        return None
    counts, edges = np.histogram(mags, bins=bins)
    return float(round(edges[int(np.argmax(counts))], 1))


# ══════════════════════════════════════════════════════════════════════
# 5. Spark Session
# ══════════════════════════════════════════════════════════════════════
def create_spark_session() -> SparkSession:
    print("\n" + "═" * 65)
    print("  Step-2  初始化 Spark Session")
    print("═" * 65)

    # PySpark 需要知道 Hadoop conf 来读取 HDFS
    spark = (
        SparkSession.builder
        .appName("EarthquakeFeatureEngineering_v4.2")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        # Hadoop HDFS 连接配置
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
        .config("spark.hadoop.dfs.replication", "1")
        # 禁用 UI 控制台进度（减少噪音输出）
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    sc = spark.sparkContext
    print(f"  PySpark 版本  : {spark.version}")
    print(f"  Master        : {sc.master}")
    print(f"  App ID        : {sc.applicationId}")
    print(f"  默认并行度    : {sc.defaultParallelism}")
    print(f"  Driver 内存   : {spark.conf.get('spark.driver.memory')}")
    print(f"  AQE 状态      : {spark.conf.get('spark.sql.adaptive.enabled')}")
    print(f"  HDFS defaultFS: {spark.conf.get('spark.hadoop.fs.defaultFS')}")
    return spark


# ══════════════════════════════════════════════════════════════════════
# 6. 从 HDFS 读取数据
# ══════════════════════════════════════════════════════════════════════
def load_data_from_hdfs(spark: SparkSession):
    print("\n" + "═" * 65)
    print("  Step-3  从 HDFS 读取数据")
    print("═" * 65)
    t0 = time.time()

    # ── 6a. 余震序列（所有 *_seq.csv）────────────────────────
    print(f"\n  读取: {HDFS_SEQ_DIR}/*_seq.csv")
    seq_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .option("mode", "DROPMALFORMED")
        .csv(f"{HDFS_SEQ_DIR}/*_seq.csv")
        .withColumn("source_file", F.input_file_name())
        .withColumn(
            "eq_id",
            F.regexp_extract(F.col("source_file"), r"(\d{14})_seq\.csv", 1)
        )
    )
    seq_count = seq_df.count()
    eq_ids    = seq_df.select("eq_id").distinct().count()
    print(f"  序列总行数  : {seq_count:,}")
    print(f"  地震序列数  : {eq_ids}")
    print(f"  分区数      : {seq_df.rdd.getNumPartitions()}")

    # ── 6b. 主震目录 ──────────────────────────────────────────
    print(f"\n  读取: {HDFS_CAT_PATH}")
    cat_raw = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(HDFS_CAT_PATH)
    )

    # 时间格式: "2024-12-21 15:30:53.399000+00:00"
    # 取前 19 字符 "2024-12-21 15:30:53"，对齐文件名精度
    cat_df = (
        cat_raw
        .withColumn(
            "mainshock_time",
            F.to_timestamp(F.substring(F.col("time").cast("string"), 1, 19),
                           "yyyy-MM-dd HH:mm:ss")
        )
        .withColumn("eq_id",
            F.date_format(F.col("mainshock_time"), "yyyyMMddHHmmss"))
        .select(
            "eq_id",
            F.col("latitude").cast(DoubleType()).alias("mainshock_lat"),
            F.col("longitude").cast(DoubleType()).alias("mainshock_lon"),
            F.col("mag").cast(DoubleType()).alias("mainshock_mag"),
            F.col("depth").cast(DoubleType()).alias("mainshock_depth"),
            "mainshock_time",
            "year",
        )
    )
    print(f"  主震目录行数: {cat_df.count():,}")
    print(f"  读取耗时    : {time.time()-t0:.1f}s")

    # ── 打印执行计划（教学重点）───────────────────────────────
    print("\n  ── seq_df 读取执行计划 (explain) ──────────────────────")
    seq_df.explain(mode="formatted")

    return seq_df, cat_df


# ══════════════════════════════════════════════════════════════════════
# 7. DataFrame API 特征提取
# ══════════════════════════════════════════════════════════════════════
def extract_features_spark(spark, seq_df, cat_df):
    print("\n" + "═" * 65)
    print("  Step-4  PySpark DataFrame API 特征提取")
    print("═" * 65)

    # ── 7a. JOIN（广播小表）──────────────────────────────────
    print("\n  [JOIN] 序列 ⋈ 主震目录 (broadcastHashJoin)...")
    joined_df = seq_df.join(F.broadcast(cat_df), on="eq_id", how="inner")
    print(f"  JOIN 结果行数: {joined_df.count():,}")

    # ── 7b. 解析时间，过滤余震（hours_since > 0）─────────────
    enriched_df = (
        joined_df
        .withColumn(
            "aft_time",
            F.to_timestamp(F.col("time"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")
        )
        .withColumn(
            "hours_since",
            (F.unix_timestamp("aft_time") -
             F.unix_timestamp("mainshock_time")) / 3600.0
        )
        .filter(F.col("hours_since") > 0)
        .withColumn(
            "dist_km",
            haversine_udf(
                F.col("mainshock_lat"), F.col("mainshock_lon"),
                F.col("latitude").cast(DoubleType()),
                F.col("longitude").cast(DoubleType())
            )
        )
        .withColumn("mag",   F.col("mag").cast(DoubleType()))
        .withColumn("depth", F.col("depth").cast(DoubleType()))
    )
    aft_count = enriched_df.count()
    print(f"  有效余震总行数: {aft_count:,}")

    # ── 7c. groupBy 聚合（DataFrame API）───────────────────
    print("\n  [AGG] 按 eq_id 聚合提取特征...")
    t_agg = time.time()

    agg_df = enriched_df.groupBy(
        "eq_id", "mainshock_mag", "mainshock_depth",
        "mainshock_lat", "mainshock_lon", "year"
    ).agg(
        F.count("*").cast(IntegerType()).alias("aftershock_count"),
        F.mean("mag").alias("mag_mean"),
        F.stddev("mag").alias("mag_std"),
        F.max("mag").alias("mag_max"),
        F.min("mag").alias("mag_min"),
        F.percentile_approx("mag", 0.5).alias("mag_median"),
        F.mean("depth").alias("depth_mean"),
        F.stddev("depth").alias("depth_std"),
        F.max("depth").alias("depth_max"),
        F.mean("dist_km").alias("dist_mean"),
        F.stddev("dist_km").alias("dist_std"),
        F.max("dist_km").alias("dist_max"),
        F.percentile_approx("dist_km", 0.5).alias("dist_median"),
        F.percentile_approx("dist_km", 0.9).alias("spatial_spread"),
        F.max("hours_since").alias("time_span_hours"),
        F.mean("hours_since").alias("time_mean_hours"),
        F.sum(F.when(F.col("hours_since") <= 24,  1).otherwise(0)).alias("count_24h"),
        F.sum(F.when(F.col("hours_since") <= 72,  1).otherwise(0)).alias("count_72h"),
        F.sum(F.when(F.col("hours_since") <= 168, 1).otherwise(0)).alias("count_168h"),
        F.sum(F.when(F.col("mag") >= 4.0, 1).otherwise(0)).alias("n_mag_ge_4"),
        F.sum(F.when(F.col("mag") >= 5.0, 1).otherwise(0)).alias("n_mag_ge_5"),
        F.sum(F.when(F.col("mag") >= 6.0, 1).otherwise(0)).alias("n_mag_ge_6"),
        F.collect_list("mag").alias("mag_list"),
    ).withColumn("mag_range",
        F.col("mag_max") - F.col("mag_min")
    ).withColumn("mag_diff_main_max",
        F.col("mainshock_mag") - F.col("mag_max")
    ).withColumn("ratio_24h",
        F.col("count_24h").cast(DoubleType()) / F.col("aftershock_count")
    ).withColumn("b_value", b_value_udf(F.col("mag_list"))
    ).withColumn("mc",      mc_udf(F.col("mag_list")))

    n_agg = agg_df.count()
    print(f"  聚合完成: {n_agg} 条  耗时: {time.time()-t_agg:.1f}s")

    # 聚合执行计划
    print("\n  ── agg_df 聚合执行计划 (explain) ──────────────────────")
    agg_df.drop("mag_list").explain(mode="formatted")

    return agg_df, enriched_df


# ══════════════════════════════════════════════════════════════════════
# 8. Spark SQL 分析
# ══════════════════════════════════════════════════════════════════════
def spark_sql_analysis(spark, agg_df):
    print("\n" + "═" * 65)
    print("  Step-5  Spark SQL 特征分析")
    print("═" * 65)

    clean_df = agg_df.drop("mag_list")
    clean_df.createOrReplaceTempView("earthquake_features")
    print("  已注册视图: earthquake_features\n")

    print("  [SQL-1] 全局统计:")
    spark.sql("""
        SELECT COUNT(*) AS total,
               ROUND(AVG(mainshock_mag),2)    AS avg_main_mag,
               ROUND(MAX(mainshock_mag),1)    AS max_main_mag,
               ROUND(AVG(aftershock_count),0) AS avg_aft_count,
               MAX(aftershock_count)          AS max_aft_count,
               ROUND(AVG(b_value),3)          AS avg_b_value
        FROM earthquake_features
    """).show(truncate=False)

    print("  [SQL-2] 按主震震级段统计:")
    spark.sql("""
        SELECT CASE WHEN mainshock_mag>=8 THEN 'M8+'
                    WHEN mainshock_mag>=7 THEN 'M7-8'
                    WHEN mainshock_mag>=6.5 THEN 'M6.5-7'
                    ELSE 'M6-6.5' END AS mag_bin,
               COUNT(*)                         AS n,
               ROUND(AVG(aftershock_count),0)   AS avg_aft,
               ROUND(MAX(aftershock_count),0)   AS max_aft,
               ROUND(AVG(b_value),3)            AS avg_b,
               ROUND(AVG(ratio_24h),3)          AS avg_24h_ratio
        FROM earthquake_features
        GROUP BY mag_bin ORDER BY mag_bin DESC
    """).show(truncate=False)

    print("  [SQL-3] 余震最多 Top-10（窗口函数 RANK）:")
    spark.sql("""
        SELECT eq_id,
               ROUND(mainshock_mag,1) AS mag, year,
               aftershock_count,
               ROUND(mag_max,1) AS max_aft_mag,
               ROUND(b_value,3) AS b_value,
               RANK() OVER (ORDER BY aftershock_count DESC) AS rank_by_count
        FROM earthquake_features
        LIMIT 10
    """).show(truncate=False)

    print("  [SQL-4] 相关系数分析:")
    spark.sql("""
        SELECT ROUND(CORR(mainshock_mag, aftershock_count),4)   AS corr_mag_count,
               ROUND(CORR(mainshock_mag, mag_max),4)            AS corr_mag_maxaft,
               ROUND(CORR(mainshock_depth, aftershock_count),4) AS corr_depth_count,
               ROUND(CORR(b_value, aftershock_count),4)         AS corr_bval_count
        FROM earthquake_features
    """).show(truncate=False)

    return clean_df


# ══════════════════════════════════════════════════════════════════════
# 9. VectorAssembler + Pipeline（Spark MLlib）
# ══════════════════════════════════════════════════════════════════════
def build_feature_vectors(spark, clean_df):
    print("\n" + "═" * 65)
    print("  Step-6  Spark MLlib VectorAssembler + StandardScaler")
    print("═" * 65)

    # 1) 所有特征列统一转 Double（避免 Integer 列 fillna 不兼容）
    filled_df = clean_df
    for c in VECTOR_FEATURE_COLS:
        filled_df = filled_df.withColumn(c, F.col(c).cast(DoubleType()))

    # 2) 用 na.fill 填充 null（对 Double 列有效）
    filled_df = filled_df.na.fill(0.0, subset=VECTOR_FEATURE_COLS)

    # 3) 用 when/isnan 替换 NaN（fillna 不处理 NaN）
    for c in VECTOR_FEATURE_COLS:
        filled_df = filled_df.withColumn(
            c, F.when(F.isnan(F.col(c)), F.lit(0.0)).otherwise(F.col(c))
        )

    n_rows = filled_df.count()
    print(f"  填充后行数     : {n_rows}")

    assembler = VectorAssembler(
        inputCols=VECTOR_FEATURE_COLS,
        outputCol="raw_features",
        handleInvalid="keep"    # keep 不跳过行，避免空 DataFrame
    )
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withMean=True, withStd=True
    )
    pipeline = Pipeline(stages=[assembler, scaler])

    print(f"  Pipeline 阶段  : {[type(s).__name__ for s in pipeline.getStages()]}")
    print(f"  输入特征维度   : {len(VECTOR_FEATURE_COLS)}")

    t0 = time.time()
    model       = pipeline.fit(filled_df)
    vec_df      = model.transform(filled_df)
    print(f"  fit+transform  : {time.time()-t0:.2f}s")

    print("\n  特征向量示例（前 3 行）:")
    vec_df.select("eq_id", "mainshock_mag", "aftershock_count", "features").show(3, truncate=80)

    print("\n  ── vectorized_df 执行计划 ──────────────────────────────")
    vec_df.select("eq_id", "features").explain(mode="formatted")

    return vec_df, model


# ══════════════════════════════════════════════════════════════════════
# 10. 保存到 HDFS + 本地
# ══════════════════════════════════════════════════════════════════════
def save_features(spark, clean_df, vec_df) -> str:
    print("\n" + "═" * 65)
    print("  Step-7  保存特征矩阵（HDFS + 本地）")
    print("═" * 65)

    # ── HDFS CSV ──────────────────────────────────────────────
    run_hdfs(["-rm", "-r", "-f", "/earthquake/features/features_csv"], check=False)
    (clean_df.coalesce(1)
             .write.option("header", "true")
             .mode("overwrite")
             .csv(f"{HDFS_FEAT_DIR}/features_csv"))
    print(f"  ✓ CSV  → HDFS {HDFS_FEAT_DIR}/features_csv/")

    # ── HDFS Parquet（列式高效存储）─────────────────────────
    run_hdfs(["-rm", "-r", "-f", "/earthquake/features/features_parquet"], check=False)
    (clean_df.write.mode("overwrite")
             .parquet(f"{HDFS_FEAT_DIR}/features_parquet"))
    print(f"  ✓ Parquet → HDFS {HDFS_FEAT_DIR}/features_parquet/")

    # ── 从 HDFS 读回验证 ─────────────────────────────────────
    verify_df = spark.read.parquet(f"{HDFS_FEAT_DIR}/features_parquet")
    print(f"\n  HDFS 读回验证: {verify_df.count()} 行")
    verify_df.printSchema()

    # ── 导出本地 CSV ──────────────────────────────────────────
    local_path = os.path.join(LOCAL_OUT_DIR, "spark_features.csv")
    pdf = clean_df.toPandas()
    pdf.to_csv(local_path, index=False)
    print(f"\n  ✓ 本地 CSV: {local_path}")
    print(f"  形状: {pdf.shape[0]} 行 × {pdf.shape[1]} 列")

    # HDFS 目录汇总
    print("\n  HDFS 存储目录:")
    run_hdfs(["-ls", "/earthquake/"], check=False)

    return local_path


# ══════════════════════════════════════════════════════════════════════
# 11. Spark 统计信息
# ══════════════════════════════════════════════════════════════════════
def print_spark_statistics(spark):
    print("\n" + "═" * 65)
    print("  Step-8  Spark 任务统计")
    print("═" * 65)

    sc = spark.sparkContext
    print(f"\n  App 名称   : {sc.appName}")
    print(f"  App ID     : {sc.applicationId}")
    print(f"  并行度     : {sc.defaultParallelism}")
    print(f"  Spark UI   : http://localhost:4040")

    conf_keys = [
        "spark.driver.memory", "spark.executor.memory",
        "spark.sql.shuffle.partitions",
        "spark.sql.adaptive.enabled",
        "spark.hadoop.fs.defaultFS",
    ]
    print("\n  ── Spark 配置 ──")
    for k in conf_keys:
        try:
            print(f"  {k:<48} = {spark.conf.get(k)}")
        except Exception:
            pass

    print("\n  ── HDFS 存储统计 ──")
    for p in ["/earthquake/usgs_sequences",
              "/earthquake/catalog",
              "/earthquake/features"]:
        out = run_hdfs(["-count", p], check=False)
        print(f"  {p}: {out}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    wall_start = time.time()

    print("╔" + "═" * 63 + "╗")
    print("║  大数据课程实训 4.2 — PySpark 特征工程（HDFS + Spark）  ║")
    print("╚" + "═" * 63 + "╝")
    print(f"  Python  : {sys.version.split()[0]}")
    try:
        import pyspark
        print(f"  PySpark : {pyspark.__version__}")
    except ImportError:
        print("  [ERROR] 请先安装: pip install pyspark")
        sys.exit(1)

    spark = None
    try:
        # Step-0: HDFS 启动
        if not start_hdfs():
            sys.exit(1)

        # Step-1: 上传数据
        upload_to_hdfs()

        # Step-2: Spark Session
        spark = create_spark_session()

        # Step-3: 读取 HDFS
        seq_df, cat_df = load_data_from_hdfs(spark)

        # Step-4: 特征提取
        agg_df, enriched_df = extract_features_spark(spark, seq_df, cat_df)

        # Step-5: Spark SQL
        clean_df = spark_sql_analysis(spark, agg_df)

        # Step-6: MLlib 向量化
        vec_df, pipeline_model = build_feature_vectors(spark, clean_df)

        # Step-7: 保存
        local_feat_path = save_features(spark, clean_df, vec_df)

        # Step-8: 统计
        print_spark_statistics(spark)

        # ── 完成汇总 ─────────────────────────────────────────
        elapsed = time.time() - wall_start
        print("\n╔" + "═" * 63 + "╗")
        print("║                 4.2 特征工程完成！                     ║")
        print("╚" + "═" * 63 + "╝")
        print(f"  总耗时           : {elapsed:.0f}s ({elapsed/60:.1f} min)")
        print(f"  本地特征文件     : {local_feat_path}")
        print(f"  HDFS 特征路径    : {HDFS_FEAT_DIR}")
        print("\n  ✓ 大数据技术栈使用汇总:")
        print("  · HDFS: -put / -ls / -count / -rm / 读写 Parquet+CSV")
        print("  · PySpark: SparkSession / DataFrame API / RDD.getNumPartitions")
        print("  · Spark SQL: createOrReplaceTempView / RANK() OVER / CORR()")
        print("  · Spark MLlib: VectorAssembler / StandardScaler / Pipeline")
        print("  · UDF: haversine_udf / b_value_udf / mc_udf")
        print("  · explain(): seq_df / agg_df / vec_df 三处执行计划")
        print("  · broadcastHashJoin: F.broadcast() 小表优化")
        print("  · AQE: spark.sql.adaptive.enabled=true")

    finally:
        if spark:
            spark.stop()
            print("\n  Spark Session 已关闭")
        stop_hdfs()


if __name__ == "__main__":
    main()
