import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from util import api_call_with_retry, init_tushare, iter_date_strings, shift_date_str


load_dotenv()


DEFAULT_INDEX_CODE_MAP = {
    "300": "399006.SZ",
    "301": "399006.SZ",
    "000": "399001.SZ",
    "001": "399001.SZ",
    "002": "399001.SZ",
    "003": "399001.SZ",
    "600": "000001.SH",
    "601": "000001.SH",
    "603": "000001.SH",
    "605": "000001.SH",
    "688": "000001.SH",
    "510": "000001.SH",
    "511": "000001.SH",
    "512": "000001.SH",
    "513": "000001.SH",
    "515": "000001.SH",
    "516": "000001.SH",
    "517": "000001.SH",
    "518": "000001.SH",
    "588": "000001.SH",
    "159": "399001.SZ",
}


def safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    if np.isscalar(denominator):
        denominator = pd.Series(float(denominator), index=numerator.index, dtype=float)
    denominator = denominator.replace(0, np.nan)
    return numerator.astype(float).div(denominator.astype(float))


def calculate_obv(close: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """计算 OBV 系列指标：返回包含 obv 和 ma_obv 均线的 DataFrame"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (vol * direction).cumsum()
    return pd.DataFrame({"obv": obv})


def get_obv_status(obv: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    """返回 (obv_status(布尔序列), 归一化趋势系数(Pearson Correlation)序列 [-1, 1])"""
    # 构造一条单调递增的时间刻度线 [0, 1, 2, ..., N-1]
    time_index = pd.Series(np.arange(len(obv)), index=obv.index)
    
    # 滚动计算 OBV 与时间的皮尔逊相关系数，结果完美落在 [-1, 1]
    # 皮尔逊相关系数 > 0，在数学上严格等价于最小二乘法斜率 > 0
    trend_score = obv.rolling(window=window).corr(time_index).fillna(0.0)
    
    obv_status = (trend_score > 0).astype(int)
    return obv_status, trend_score


def rolling_ma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rolling_llv(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def rolling_hhv(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=1).mean()


def rolling_avedev(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )


def ref(series: pd.Series, periods: int = 1) -> pd.Series:
    return series.shift(periods)


def barscount(series: pd.Series) -> pd.Series:
    counts = np.arange(1, len(series) + 1, dtype=float)
    counts[pd.isna(series.to_numpy())] = np.nan
    return pd.Series(counts, index=series.index)


def sma_tdx(series: pd.Series, window: int, weight: int = 1) -> pd.Series:
    result = []
    prev = np.nan
    for value in series.astype(float).to_numpy():
        if np.isnan(value):
            result.append(prev)
            continue
        if np.isnan(prev):
            prev = value
        else:
            prev = (weight * value + (window - weight) * prev) / window
        result.append(prev)
    return pd.Series(result, index=series.index, dtype=float)


def count_condition(condition: pd.Series, window: int) -> pd.Series:
    numeric = condition.fillna(False).astype(int)
    return numeric.rolling(window=window, min_periods=1).sum()


def backset(condition: pd.Series, periods: int) -> pd.Series:
    values = condition.fillna(False).astype(bool).to_numpy()
    result = np.zeros(len(values), dtype=bool)
    for idx, is_true in enumerate(values):
        if not is_true:
            continue
        start = max(0, idx - periods + 1)
        result[start : idx + 1] = True
    return pd.Series(result, index=condition.index)


def cross(series_a: pd.Series, series_b: pd.Series | float) -> pd.Series:
    if np.isscalar(series_b):
        series_b = pd.Series(series_b, index=series_a.index, dtype=float)
    else:
        series_b = series_b.reindex(series_a.index)
    prev_a = ref(series_a, 1)
    prev_b = ref(series_b, 1)
    return (series_a > series_b) & (prev_a <= prev_b)


def cross_down(series_a: pd.Series, series_b: pd.Series | float) -> pd.Series:
    if np.isscalar(series_b):
        series_b = pd.Series(series_b, index=series_a.index, dtype=float)
    else:
        series_b = series_b.reindex(series_a.index)
    prev_a = ref(series_a, 1)
    prev_b = ref(series_b, 1)
    return (series_a < series_b) & (prev_a >= prev_b)


def zig_pivots(series: pd.Series, percent: float) -> tuple[pd.Series, pd.Series]:
    values = series.astype(float).to_numpy()
    peak_mask = np.zeros(len(values), dtype=bool)
    trough_mask = np.zeros(len(values), dtype=bool)

    valid_idx = np.where(~np.isnan(values))[0]
    if len(valid_idx) == 0:
        return pd.Series(peak_mask, index=series.index), pd.Series(trough_mask, index=series.index)

    threshold = percent / 100.0
    first = int(valid_idx[0])
    candidate_idx = first
    candidate_price = values[first]
    trend = 0

    # 在未确定趋势阶段，同时跟踪最高价和最低价的位置
    highest_idx = first
    highest_price = values[first]
    lowest_idx = first
    lowest_price = values[first]

    for idx in valid_idx[1:]:
        price = values[idx]

        if trend == 0:
            # 更新极值（使用 >= / <= 保证取最后一个极值点作为转折）
            if price >= highest_price:
                highest_idx = idx
                highest_price = price
            if price <= lowest_price:
                lowest_idx = idx
                lowest_price = price

            # 检查是否确认方向：从最低点上涨超过阈值 → 最低点为谷
            if lowest_price > 0 and (price - lowest_price) / lowest_price >= threshold:
                trough_mask[lowest_idx] = True
                trend = 1
                candidate_idx = idx
                candidate_price = price
            # 从最高点下跌超过阈值 → 最高点为峰
            elif highest_price > 0 and (highest_price - price) / highest_price >= threshold:
                peak_mask[highest_idx] = True
                trend = -1
                candidate_idx = idx
                candidate_price = price
            continue

        if trend == 1:
            if price >= candidate_price:
                candidate_idx = idx
                candidate_price = price
                continue
            drawdown = (candidate_price - price) / candidate_price
            if drawdown >= threshold:
                peak_mask[candidate_idx] = True
                trend = -1
                candidate_idx = idx
                candidate_price = price
        else:
            if price <= candidate_price:
                candidate_idx = idx
                candidate_price = price
                continue
            rebound = (price - candidate_price) / candidate_price
            if rebound >= threshold:
                trough_mask[candidate_idx] = True
                trend = 1
                candidate_idx = idx
                candidate_price = price

    # 末尾候选点标记为峰/谷（与 THS 行为一致：PEAKBARS/TROUGHBARS 包含
    # ZIG 末尾连接到当前价格的最后一个转折点）
    if trend == 1:
        peak_mask[candidate_idx] = True
    elif trend == -1:
        trough_mask[candidate_idx] = True

    return pd.Series(peak_mask, index=series.index), pd.Series(trough_mask, index=series.index)


def bars_since_recent_pivot(pivot_mask: pd.Series, occurrence: int = 1) -> pd.Series:
    values = pivot_mask.fillna(False).astype(bool).to_numpy()
    result = np.full(len(values), np.nan, dtype=float)
    pivot_positions: list[int] = []

    for idx, is_pivot in enumerate(values):
        if is_pivot:
            pivot_positions.append(idx)
        if len(pivot_positions) >= occurrence:
            target_idx = pivot_positions[-occurrence]
            result[idx] = idx - target_idx

    return pd.Series(result, index=pivot_mask.index)


def peakbars_close(series: pd.Series, percent: float, occurrence: int = 1) -> pd.Series:
    peak_mask, _ = zig_pivots(series, percent)
    return bars_since_recent_pivot(peak_mask, occurrence=occurrence)


def troughbars_close(series: pd.Series, percent: float, occurrence: int = 1) -> pd.Series:
    _, trough_mask = zig_pivots(series, percent)
    return bars_since_recent_pivot(trough_mask, occurrence=occurrence)


def get_default_end_date() -> str:
    """17:00 之前取前一个交易日，17:00 之后取当天日期。"""
    now = datetime.now()
    if now.hour < 17:
        target = now - timedelta(days=1)
    else:
        target = now
    return target.strftime("%Y%m%d")


def get_default_backtest_dates() -> tuple[str, str]:
    end_date = get_default_end_date()
    start_dt = datetime.strptime(end_date, "%Y%m%d") - timedelta(days=365)
    return start_dt.strftime("%Y%m%d"), end_date


def prewarm_all_caches(pro, data_start_date: str, end_date: str) -> None:
    """在批量扫描前一次性预热所有按日缓存，避免逐股票触发重复 API 调用。"""
    print(f"预热缓存 ({data_start_date} ~ {end_date}) ...")
    ensure_stock_day_cache(pro, data_start_date, end_date)
    print("  ✔ daily 缓存完成")
    ensure_adj_factor_day_cache(pro, data_start_date, end_date)
    print("  ✔ adj_factor 缓存完成")
    ensure_float_share_day_cache(pro, data_start_date, end_date)
    print("  ✔ daily_basic (float_share) 缓存完成")
    load_day_caches_to_memory(data_start_date, end_date)
    print("  ✔ 内存索引加载完成")
    print("缓存预热完成\n")


# ---- 批量扫描用的内存缓存（按 ts_code 索引） ----
_mem_stock_by_code: dict[str, pd.DataFrame] = {}
_mem_adj_by_code: dict[str, pd.DataFrame] = {}
_mem_float_by_code: dict[str, pd.DataFrame] = {}
_mem_cache_range: tuple[str, str] | None = None


def _mem_range_covers(start_date: str, end_date: str) -> bool:
    return (
        _mem_cache_range is not None
        and _mem_cache_range[0] <= start_date
        and _mem_cache_range[1] >= end_date
    )


def load_day_caches_to_memory(start_date: str, end_date: str) -> None:
    """将所有按日 CSV 合并到内存，并按 ts_code 建立索引，后续查询为 O(1)。"""
    global _mem_stock_by_code, _mem_adj_by_code, _mem_float_by_code, _mem_cache_range

    stock_req = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    adj_req = ["ts_code", "trade_date", "adj_factor"]
    float_req = ["ts_code", "trade_date", "float_share", "pe_ttm"]

    stock_frames: list[pd.DataFrame] = []
    adj_frames: list[pd.DataFrame] = []
    float_frames: list[pd.DataFrame] = []

    for trade_date in iter_date_strings(start_date, end_date):
        sp = get_stock_day_cache_path(trade_date)
        if sp.exists():
            df = read_csv_cache(sp, stock_req, ["ts_code", "trade_date"])
            if not df.empty:
                stock_frames.append(df)
        ap = get_adj_factor_day_cache_path(trade_date)
        if ap.exists():
            df = read_csv_cache(ap, adj_req, ["ts_code", "trade_date"])
            if not df.empty:
                adj_frames.append(df)
        fp = get_float_share_day_cache_path(trade_date)
        if fp.exists():
            df = read_csv_cache(fp, float_req, ["ts_code", "trade_date"])
            if not df.empty:
                float_frames.append(df)

    stock_all = pd.concat(stock_frames, ignore_index=True) if stock_frames else pd.DataFrame(columns=stock_req)
    adj_all = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame(columns=adj_req)
    float_all = pd.concat(float_frames, ignore_index=True) if float_frames else pd.DataFrame(columns=float_req)

    _mem_stock_by_code = {code: group.reset_index(drop=True) for code, group in stock_all.groupby("ts_code")} if not stock_all.empty else {}
    _mem_adj_by_code = {code: group.reset_index(drop=True) for code, group in adj_all.groupby("ts_code")} if not adj_all.empty else {}
    _mem_float_by_code = {code: group.reset_index(drop=True) for code, group in float_all.groupby("ts_code")} if not float_all.empty else {}
    _mem_cache_range = (start_date, end_date)


def extend_start_date(start_date: str, buffer_days: int = 180) -> str:
    start_dt = datetime.strptime(start_date, "%Y%m%d")
    extended_dt = start_dt - timedelta(days=buffer_days)
    return extended_dt.strftime("%Y%m%d")


def normalize_code_input(code: str) -> str:
    return code.strip().upper()


def infer_index_code_from_ts_code(ts_code: str) -> Optional[str]:
    return DEFAULT_INDEX_CODE_MAP.get(ts_code[:3])


def resolve_security(pro, code: str) -> dict:
    normalized = normalize_code_input(code)
    if "." in normalized:
        ts_code = normalized
        if ts_code.startswith(("510", "511", "512", "513", "515", "516", "517", "518", "159")):
            return {"ts_code": ts_code, "security_type": "etf", "name": ts_code}
        return {"ts_code": ts_code, "security_type": "stock", "name": ts_code}

    if not normalized.isdigit() or len(normalized) != 6:
        raise ValueError(f"代码格式错误，需为6位数字或完整 ts_code: {code}")

    stock_df = pro.stock_basic(exchange="SSE,SZSE", list_status="L", fields="ts_code,symbol,name")
    stock_matches = stock_df[stock_df["symbol"].astype(str) == normalized] if stock_df is not None and not stock_df.empty else pd.DataFrame()
    if not stock_matches.empty:
        row = stock_matches.iloc[0]
        return {"ts_code": str(row["ts_code"]), "security_type": "stock", "name": str(row["name"])}

    fund_df = pro.fund_basic(market="E", status="L", fields="ts_code,name")
    fund_matches = fund_df[fund_df["ts_code"].astype(str).str.startswith(normalized)] if fund_df is not None and not fund_df.empty else pd.DataFrame()
    if not fund_matches.empty:
        exact_matches = fund_matches[fund_matches["ts_code"].astype(str).str[:6] == normalized]
        if exact_matches.empty:
            exact_matches = fund_matches
        row = exact_matches.iloc[0]
        return {"ts_code": str(row["ts_code"]), "security_type": "etf", "name": str(row["name"])}

    raise ValueError(f"无法识别代码 {code}，未在股票或ETF清单中找到")


def get_cache_dir(*parts: str) -> Path:
    return Path(__file__).resolve().parent.joinpath("cache", *parts)


def get_stock_day_cache_path(trade_date: str) -> Path:
    return get_cache_dir("daily", "stocks", "by_day") / f"{trade_date}.csv"


def get_adj_factor_day_cache_path(trade_date: str) -> Path:
    return get_cache_dir("adj_factor", "by_day") / f"{trade_date}.csv"


def get_float_share_day_cache_path(trade_date: str) -> Path:
    return get_cache_dir("daily_basic", "by_day") / f"{trade_date}.csv"


def get_stock_code_cache_path(ts_code: str) -> Path:
    return get_cache_dir("daily", "stocks", "by_code") / f"{ts_code}.csv"


def get_index_cache_path(ts_code: str) -> Path:
    return get_cache_dir("daily", "indexes") / f"{ts_code}.csv"


def get_etf_cache_path(ts_code: str) -> Path:
    return get_cache_dir("daily", "etfs") / f"{ts_code}.csv"


def normalize_daily_df(df: pd.DataFrame, required_columns: list[str], dedupe_subset: list[str]) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=required_columns)
    if df.empty:
        return pd.DataFrame(columns=required_columns)

    df = df.copy()
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"日线数据缺少必要列: {missing}")

    for column in required_columns:
        if column not in {"trade_date", "ts_code"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].astype(str)
    if "ts_code" in df.columns:
        df["ts_code"] = df["ts_code"].astype(str)

    return (
        df[required_columns]
        .drop_duplicates(subset=dedupe_subset, keep="last")
        .sort_values(dedupe_subset)
        .reset_index(drop=True)
    )
import functools

@functools.lru_cache(maxsize=1000)
def _read_csv_cached_internal(file_path_str: str, required_columns: tuple, dedupe_subset: tuple) -> pd.DataFrame:
    file_path = Path(file_path_str)
    if not file_path.exists():
        return pd.DataFrame(columns=list(required_columns))
    df = pd.read_csv(file_path)
    return normalize_daily_df(df, required_columns=list(required_columns), dedupe_subset=list(dedupe_subset))


def read_csv_cache(file_path: Path, required_columns: list[str], dedupe_subset: list[str]) -> pd.DataFrame:
    return _read_csv_cached_internal(str(file_path), tuple(required_columns), tuple(dedupe_subset))


def write_csv_cache(file_path: Path, df: pd.DataFrame, required_columns: list[str], dedupe_subset: list[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_df = normalize_daily_df(df, required_columns=required_columns, dedupe_subset=dedupe_subset)
    normalized_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    _read_csv_cached_internal.cache_clear()



_ATTEMPTED_EMPTY_STOCK_DATES: set[str] = set()
_ENSURED_STOCK_DAY_RANGE: tuple[str, str] | None = None

def ensure_stock_day_cache(pro, start_date: str, end_date: str) -> None:
    global _ENSURED_STOCK_DAY_RANGE
    if _ENSURED_STOCK_DAY_RANGE == (start_date, end_date):
        return
    required_columns = ["ts_code", "trade_date", "open", "high", "low", "close", "vol"]
    for trade_date in iter_date_strings(start_date, end_date):
        file_path = get_stock_day_cache_path(trade_date)
        
        cache_valid = False
        if file_path.exists():
            try:
                cached = read_csv_cache(file_path, required_columns, ["ts_code", "trade_date"])
                if not cached.empty:
                    cache_valid = True
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError):
                pass
                
        if cache_valid:
            continue

        # 历史日期缓存无效直接跳过，只有 end_date 才触发刷新
        if trade_date != end_date:
            continue
        if trade_date in _ATTEMPTED_EMPTY_STOCK_DATES:
            continue
        _ATTEMPTED_EMPTY_STOCK_DATES.add(trade_date)
            
        df = api_call_with_retry(
            pro.daily,
            pro_api_instance=pro,
            start_date=trade_date,
            end_date=trade_date,
            timeout=120,
            fields=["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
        )
        if df is not None and not df.empty:
            write_csv_cache(file_path, df, required_columns=required_columns, dedupe_subset=["ts_code", "trade_date"])
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False, encoding="utf-8-sig")
            _read_csv_cached_internal.cache_clear()
    _ENSURED_STOCK_DAY_RANGE = (start_date, end_date)


_ATTEMPTED_EMPTY_ADJ_DATES: set[str] = set()
_ENSURED_ADJ_FACTOR_RANGE: tuple[str, str] | None = None

def ensure_adj_factor_day_cache(pro, start_date: str, end_date: str) -> None:
    global _ENSURED_ADJ_FACTOR_RANGE
    if _ENSURED_ADJ_FACTOR_RANGE == (start_date, end_date):
        return
    required_columns = ["ts_code", "trade_date", "adj_factor"]
    for trade_date in iter_date_strings(start_date, end_date):
        file_path = get_adj_factor_day_cache_path(trade_date)
        
        cache_valid = False
        if file_path.exists():
            try:
                cached = read_csv_cache(file_path, required_columns, ["ts_code", "trade_date"])
                if not cached.empty:
                    cache_valid = True
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError):
                pass
                
        if cache_valid:
            continue

        # 历史日期缓存无效直接跳过，只有 end_date 才触发刷新
        if trade_date != end_date:
            continue
        if trade_date in _ATTEMPTED_EMPTY_ADJ_DATES:
            continue
        _ATTEMPTED_EMPTY_ADJ_DATES.add(trade_date)
            
        df = api_call_with_retry(
            pro.adj_factor,
            pro_api_instance=pro,
            trade_date=trade_date,
            timeout=120,
            fields=["ts_code", "trade_date", "adj_factor"],
        )
        if df is not None and not df.empty:
            write_csv_cache(file_path, df, required_columns=required_columns, dedupe_subset=["ts_code", "trade_date"])
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False, encoding="utf-8-sig")
            _read_csv_cached_internal.cache_clear()
    _ENSURED_ADJ_FACTOR_RANGE = (start_date, end_date)


_ATTEMPTED_EMPTY_FLOAT_SHARE_DATES: set[str] = set()
_ENSURED_FLOAT_SHARE_RANGE: tuple[str, str] | None = None

def ensure_float_share_day_cache(pro, start_date: str, end_date: str) -> None:
    global _ENSURED_FLOAT_SHARE_RANGE
    if _ENSURED_FLOAT_SHARE_RANGE == (start_date, end_date):
        return
    required_columns = ["ts_code", "trade_date", "float_share", "pe_ttm"]
    for trade_date in iter_date_strings(start_date, end_date):
        file_path = get_float_share_day_cache_path(trade_date)
        
        cache_valid = False
        if file_path.exists():
            try:
                cached = read_csv_cache(file_path, required_columns, ["ts_code", "trade_date"])
                if not cached.empty and "pe_ttm" in cached.columns:
                    cache_valid = True
            except (ValueError, KeyError, FileNotFoundError, pd.errors.EmptyDataError):
                pass
                
        if cache_valid:
            continue

        # 历史日期缓存无效直接跳过，只有 end_date 才触发刷新
        if trade_date != end_date:
            continue
        if trade_date in _ATTEMPTED_EMPTY_FLOAT_SHARE_DATES:
            continue
        _ATTEMPTED_EMPTY_FLOAT_SHARE_DATES.add(trade_date)
        df = api_call_with_retry(
            pro.daily_basic,
            pro_api_instance=pro,
            trade_date=trade_date,
            timeout=120,
            fields=["ts_code", "trade_date", "float_share", "pe_ttm"],
        )
        if df is not None and not df.empty:
            write_csv_cache(file_path, df, required_columns=required_columns, dedupe_subset=["ts_code", "trade_date"])
        else:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(columns=required_columns).to_csv(file_path, index=False, encoding="utf-8-sig")
            _read_csv_cached_internal.cache_clear()
    _ENSURED_FLOAT_SHARE_RANGE = (start_date, end_date)


def reconstruct_qfq_prices(stock_df: pd.DataFrame, adj_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = stock_df.merge(adj_df, on=["ts_code", "trade_date"], how="left")
    merged_df["adj_factor"] = pd.to_numeric(merged_df["adj_factor"], errors="coerce")
    latest_adj_factor = merged_df["adj_factor"].dropna().iloc[-1] if merged_df["adj_factor"].notna().any() else np.nan
    if pd.isna(latest_adj_factor) or latest_adj_factor == 0:
        raise ValueError(f"无法构建前复权价格，缺少 adj_factor: {merged_df['ts_code'].iloc[0] if not merged_df.empty else ''}")

    for column in ["open", "high", "low", "close"]:
        merged_df[column] = pd.to_numeric(merged_df[column], errors="coerce") * merged_df["adj_factor"] / latest_adj_factor

    return normalize_daily_df(
        merged_df,
        required_columns=["trade_date", "open", "high", "low", "close", "vol"],
        dedupe_subset=["trade_date"],
    )


def build_stock_qfq_slice_from_day_cache(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    if start_date > end_date:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "vol"])

    # 优先从内存缓存取数据（O(1) 字典查找）
    if _mem_range_covers(start_date, end_date):
        stock_raw_df = _mem_stock_by_code.get(ts_code, pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol"])).copy()
        adj_factor_df = _mem_adj_by_code.get(ts_code, pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])).copy()
    else:
        ensure_stock_day_cache(pro, start_date, end_date)
        ensure_adj_factor_day_cache(pro, start_date, end_date)

        stock_frames: list[pd.DataFrame] = []
        adj_frames: list[pd.DataFrame] = []
        for trade_date in iter_date_strings(start_date, end_date):
            stock_day_df = read_csv_cache(
                get_stock_day_cache_path(trade_date),
                required_columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol"],
                dedupe_subset=["ts_code", "trade_date"],
            )
            if not stock_day_df.empty:
                stock_frames.append(stock_day_df[stock_day_df["ts_code"] == ts_code])

            adj_day_df = read_csv_cache(
                get_adj_factor_day_cache_path(trade_date),
                required_columns=["ts_code", "trade_date", "adj_factor"],
                dedupe_subset=["ts_code", "trade_date"],
            )
            if not adj_day_df.empty:
                adj_frames.append(adj_day_df[adj_day_df["ts_code"] == ts_code])

        stock_raw_df = pd.concat(stock_frames, ignore_index=True) if stock_frames else pd.DataFrame(columns=["ts_code", "trade_date", "open", "high", "low", "close", "vol"])
        adj_factor_df = pd.concat(adj_frames, ignore_index=True) if adj_frames else pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])

    if stock_raw_df.empty:
        return pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "vol"])

    return reconstruct_qfq_prices(stock_raw_df, adj_factor_df)



def load_stock_daily_from_cache(pro, ts_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    code_cache_path = get_stock_code_cache_path(ts_code)
    required_columns = ["trade_date", "open", "high", "low", "close", "vol"]
    existing_df = read_csv_cache(
        code_cache_path,
        required_columns=required_columns,
        dedupe_subset=["trade_date"],
    )

    if not existing_df.empty:
        cached_min = str(existing_df["trade_date"].min())
        cached_max = str(existing_df["trade_date"].max())
        if start_date >= cached_min and end_date <= cached_max:
            final_df = existing_df[(existing_df["trade_date"] >= start_date) & (existing_df["trade_date"] <= end_date)].reset_index(drop=True)
            if not final_df.empty:
                return final_df

    frames: list[pd.DataFrame] = [existing_df] if not existing_df.empty else []
    if existing_df.empty:
        missing_df = build_stock_qfq_slice_from_day_cache(pro, ts_code, start_date, end_date)
        if missing_df.empty:
            raise ValueError(f"无法获取 stock 日线数据: {ts_code}")
        frames.append(missing_df)
    else:
        cached_min = str(existing_df["trade_date"].min())
        cached_max = str(existing_df["trade_date"].max())
        if start_date < cached_min:
            left_df = build_stock_qfq_slice_from_day_cache(pro, ts_code, start_date, shift_date_str(cached_min, -1))
            if not left_df.empty:
                frames.append(left_df)
        if end_date > cached_max:
            right_df = build_stock_qfq_slice_from_day_cache(pro, ts_code, shift_date_str(cached_max, 1), end_date)
            if not right_df.empty:
                frames.append(right_df)

    merged_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required_columns)
    write_csv_cache(
        code_cache_path,
        merged_df,
        required_columns=required_columns,
        dedupe_subset=["trade_date"],
    )

    final_df = read_csv_cache(
        code_cache_path,
        required_columns=required_columns,
        dedupe_subset=["trade_date"],
    )
    final_df = final_df[(final_df["trade_date"] >= start_date) & (final_df["trade_date"] <= end_date)].reset_index(drop=True)
    if final_df.empty:
        raise ValueError(f"缓存中未找到指定区间的 stock 日线数据: {ts_code}")
    return final_df


def update_range_cache(
    file_path: Path,
    fetcher,
    ts_code: str,
    start_date: str,
    end_date: str,
    required_columns: list[str],
) -> pd.DataFrame:
    existing_df = read_csv_cache(file_path, required_columns=required_columns, dedupe_subset=["trade_date"])
    frames = [existing_df]

    if existing_df.empty:
        fetched_df = fetcher(ts_code=ts_code, start_date=start_date, end_date=end_date)
        frames.append(fetched_df)
    else:
        cached_min = str(existing_df["trade_date"].min())
        cached_max = str(existing_df["trade_date"].max())
        if start_date < cached_min:
            frames.append(fetcher(ts_code=ts_code, start_date=start_date, end_date=shift_date_str(cached_min, -1)))
        if end_date > cached_max:
            frames.append(fetcher(ts_code=ts_code, start_date=shift_date_str(cached_max, 1), end_date=end_date))

    merged_df = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True) if any(frame is not None and not frame.empty for frame in frames) else pd.DataFrame(columns=required_columns)
    write_csv_cache(file_path, merged_df, required_columns=required_columns, dedupe_subset=["trade_date"])
    final_df = read_csv_cache(file_path, required_columns=required_columns, dedupe_subset=["trade_date"])
    final_df = final_df[(final_df["trade_date"] >= start_date) & (final_df["trade_date"] <= end_date)].reset_index(drop=True)
    return final_df


def fetch_daily_data(pro, ts_code: str, security_type: str, start_date: str, end_date: str, is_index: bool = False) -> pd.DataFrame:
    required_columns = ["trade_date", "open", "high", "low", "close", "vol"]

    if is_index:
        df = update_range_cache(
            file_path=get_index_cache_path(ts_code),
            fetcher=lambda **kwargs: api_call_with_retry(
                pro.index_daily,
                pro_api_instance=pro,
                timeout=120,
                fields=["trade_date", "open", "high", "low", "close", "vol"],
                **kwargs,
            ),
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            required_columns=required_columns,
        )
    elif security_type == "stock":
        df = load_stock_daily_from_cache(pro, ts_code=ts_code, start_date=start_date, end_date=end_date)
    elif security_type == "etf":
        df = update_range_cache(
            file_path=get_etf_cache_path(ts_code),
            fetcher=lambda **kwargs: api_call_with_retry(
                pro.fund_daily,
                pro_api_instance=pro,
                timeout=120,
                fields=["trade_date", "open", "high", "low", "close", "vol"],
                **kwargs,
            ),
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            required_columns=required_columns,
        )
    else:
        raise ValueError(f"不支持的 security_type: {security_type}")

    if df is None or df.empty:
        raise ValueError(f"无法获取 {'指数' if is_index else security_type} 日线数据: {ts_code}")

    return normalize_daily_df(df, required_columns=required_columns, dedupe_subset=["trade_date"])


def fetch_float_share(pro, ts_code: str, start_date: str, end_date: str, security_type: str) -> pd.DataFrame:
    if security_type == "etf" or security_type == "fund":
        try:
            float_df = api_call_with_retry(
                pro.fund_share,
                pro_api_instance=pro,
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                timeout=120,
            )
            if float_df is not None and not float_df.empty and "fd_share" in float_df.columns:
                float_df = float_df.rename(columns={"fd_share": "float_share"})
                float_df["trade_date"] = float_df["trade_date"].astype(str)
                float_df["float_share"] = pd.to_numeric(float_df["float_share"], errors="coerce")
                return float_df[["trade_date", "float_share"]].sort_values("trade_date").reset_index(drop=True)
        except Exception as exc:
            print(f"获取 fund_share 失败: {exc}")
        return pd.DataFrame(columns=["trade_date", "float_share"])

    if security_type != "stock":
        return pd.DataFrame(columns=["trade_date", "float_share", "pe_ttm"])

    # 优先从内存缓存取数据（O(1) 字典查找）
    if _mem_range_covers(start_date, end_date):
        code_df = _mem_float_by_code.get(ts_code)
        if code_df is None or code_df.empty:
            return pd.DataFrame(columns=["trade_date", "float_share", "pe_ttm"])
        code_df = code_df.copy()
        code_df["trade_date"] = code_df["trade_date"].astype(str)
        code_df["float_share"] = pd.to_numeric(code_df["float_share"], errors="coerce")
        code_df["pe_ttm"] = pd.to_numeric(code_df["pe_ttm"], errors="coerce")
        return code_df[["trade_date", "float_share", "pe_ttm"]].sort_values("trade_date").reset_index(drop=True)

    ensure_float_share_day_cache(pro, start_date, end_date)

    required_columns = ["ts_code", "trade_date", "float_share", "pe_ttm"]
    frames: list[pd.DataFrame] = []
    for trade_date in iter_date_strings(start_date, end_date):
        day_df = read_csv_cache(
            get_float_share_day_cache_path(trade_date),
            required_columns=required_columns,
            dedupe_subset=["ts_code", "trade_date"],
        )
        if not day_df.empty:
            code_df = day_df[day_df["ts_code"] == ts_code]
            if not code_df.empty:
                frames.append(code_df)

    if not frames:
        return pd.DataFrame(columns=["trade_date", "float_share", "pe_ttm"])

    merged_df = pd.concat(frames, ignore_index=True)
    merged_df["trade_date"] = merged_df["trade_date"].astype(str)
    merged_df["float_share"] = pd.to_numeric(merged_df["float_share"], errors="coerce")
    merged_df["pe_ttm"] = pd.to_numeric(merged_df["pe_ttm"], errors="coerce")
    return merged_df[["trade_date", "float_share", "pe_ttm"]].sort_values("trade_date").reset_index(drop=True)


def build_output_path(ts_code: str, output_csv: Optional[str]) -> Path:
    if output_csv:
        return Path(output_csv).resolve()
    return (Path(__file__).resolve().parent / "result" / f"ths_indicator_{ts_code}.csv").resolve()


def calculate_mxs_indicators(stock_df: pd.DataFrame, index_df: pd.DataFrame, float_df: pd.DataFrame, security_type: str) -> pd.DataFrame:
    df = stock_df.merge(
        index_df.rename(
            columns={
                "open": "index_open",
                "high": "index_high",
                "low": "index_low",
                "close": "index_close",
                "vol": "index_vol",
            }
        )[["trade_date", "index_high", "index_low", "index_close"]],
        on="trade_date",
        how="left",
    )

    df = df.merge(float_df, on="trade_date", how="left")
    df[["index_high", "index_low", "index_close"]] = df[["index_high", "index_low", "index_close"]].ffill().bfill()
    df["float_share"] = df["float_share"].ffill().bfill()

    close = df["close"]
    open_price = df["open"]
    high = df["high"]
    low = df["low"]
    vol = df["vol"]
    index_close = df["index_close"]
    index_high = df["index_high"]
    index_low = df["index_low"]

    var5 = rolling_llv(low, 75)
    var6 = rolling_hhv(high, 75)
    var7 = (var6 - var5) / 100
    var8 = sma_tdx(safe_divide(close - var5, var7), 20, 1)
    var9 = sma_tdx(safe_divide(open_price - var5, var7), 20, 1)
    vara = 3 * var8 - 2 * sma_tdx(var8, 15, 1)
    retail_line = 100 - vara

    if security_type in ("stock", "etf", "fund") and df["float_share"].notna().any():
        capital_hands = df["float_share"] * 100.0
        vare = ref(low, 1) * 0.9
        varf = low * 0.9
        var10 = safe_divide(varf * vol + vare * (capital_hands - vol), capital_hands)
        var11 = ema(var10, 30)
        var12 = close - ref(close, 1)
        var13 = pd.Series(np.maximum(var12, 0), index=df.index)
        var14 = var12.abs()
        var15 = safe_divide(sma_tdx(var13, 7, 1), sma_tdx(var14, 7, 1)) * 100
        var16 = safe_divide(sma_tdx(var13, 13, 1), sma_tdx(var14, 13, 1)) * 100
        var17 = barscount(close)
        var18 = safe_divide(sma_tdx(var13, 6, 1), sma_tdx(var14, 6, 1)) * 100
        var19 = safe_divide((-200) * (rolling_hhv(high, 60) - close), rolling_hhv(high, 60) - rolling_llv(low, 60)) + 100
        var1a = safe_divide(close - rolling_llv(low, 15), rolling_hhv(high, 15) - rolling_llv(low, 15)) * 100
        var1b = sma_tdx((sma_tdx(var1a, 4, 1) - 50) * 2, 3, 1)
        var1c = safe_divide(index_close - rolling_llv(index_low, 14), rolling_hhv(index_high, 14) - rolling_llv(index_low, 14)) * 100
        var1d = sma_tdx(var1c, 4, 1)
        var1e = sma_tdx(var1d, 3, 1)
        var1f = safe_divide(rolling_hhv(high, 30) - close, close) * 100

        var20 = (
            (var18 <= 25)
            & (var19 < -95)
            & (var1f > 20)
            & (var1b < -30)
            & (var1e < 30)
            & ((var11 - close) >= -0.25)
            & (var15 < 22)
            & (var16 < 28)
            & (var17 > 50)
        )
        build_position_signal = np.where(
            cross(var20.astype(float), 0.5) & (count_condition(var20, 10) == 1),
            1,
            0,
        )
    else:
        build_position_signal = np.zeros(len(df), dtype=int)

    v6 = safe_divide(close, ref(close, 3)) >= 1.1
    v7 = backset(v6, 3)
    buy_breakout_signal = np.where(v7 & (count_condition(v7, 3) == 1), 1, 0)

    peak_bars = peakbars_close(close, percent=15, occurrence=1)
    head = pd.Series(np.where(peak_bars < 10, 100.0, 0.0), index=df.index)
    sell_signal = np.where(head > ref(head, 1).fillna(0), 1, 0)

    trough_bars = troughbars_close(close, percent=15, occurrence=1)
    bottom = pd.Series(np.where(trough_bars < 10, 50.0, 0.0), index=df.index)
    buy_bottom_signal = np.where(bottom > ref(bottom, 1).fillna(0), 1, 0)

    display_buy_signal = np.where((buy_breakout_signal > 0) | (buy_bottom_signal > 0), 1, 0)
    main_force_line = safe_divide(close - rolling_llv(low, 30), rolling_hhv(close, 30) - rolling_llv(low, 30)) * 100

    build_position_series = pd.Series(build_position_signal, index=df.index, dtype=float)
    prev_main_force = ref(main_force_line, 1)
    exec_buy_signal = np.where(
        (build_position_series == 1)
        | ((main_force_line > prev_main_force) & (prev_main_force < 20) & (main_force_line < 35)),
        1,
        0,
    )
    exec_sell_signal = np.where(
        cross_down(main_force_line, 80)
        | ((main_force_line < prev_main_force) & (prev_main_force > 80)),
        1,
        0,
    )

    # 深度整合 OBV 逻辑
    obv_df = calculate_obv(close, vol)
    obv_status, obv_slope = get_obv_status(obv_df["obv"], window=5)

    return pd.DataFrame(
        {
            "trade_date": df["trade_date"],
            "stock_open": open_price,
            "retail_line": retail_line,
            "build_position_signal": build_position_signal,
            "buy_breakout_signal": buy_breakout_signal,
            "buy_bottom_signal": buy_bottom_signal,
            "sell_signal": sell_signal,
            "display_buy_signal": display_buy_signal,
            "exec_buy_signal": exec_buy_signal,
            "exec_sell_signal": exec_sell_signal,
            "main_force_line": main_force_line,
            "obv_status": obv_status,
            "obv_trend_score": obv_slope,
        }
    )


def run_signal_backtest(
    result_df: pd.DataFrame,
    buy_col: str = "exec_buy_signal",
    sell_col: str = "exec_sell_signal",
    trade_mode: int = 1,
    use_obv_filter: bool = False,
) -> tuple[pd.DataFrame, dict]:
    trades: list[dict] = []
    in_position = False
    
    buy_signal_date: Optional[str] = None
    buy_trade_date: Optional[str] = None
    buy_price: Optional[float] = None
    
    for idx in range(1, len(result_df)):
        row = result_df.iloc[idx]
        prev_row = result_df.iloc[idx - 1]
        
        if not in_position:
            trigger_buy = False
            buy_sig_date = None
            
            if trade_mode == 1:
                if prev_row[buy_col] == 1:
                    trigger_buy = True
                    buy_sig_date = str(prev_row["trade_date"])
            elif trade_mode == 2 and idx >= 2:
                pp_row = result_df.iloc[idx - 2]
                if pp_row[buy_col] == 1 and prev_row[buy_col] == 0:
                    trigger_buy = True
                    buy_sig_date = str(pp_row["trade_date"])
            
            if trigger_buy:
                # OBV 过滤：启用时只有 OBV 向上（obv_status > 0）才允许买入
                if use_obv_filter:
                    if "obv_status" not in prev_row.index or float(prev_row["obv_status"]) <= 0:
                        trigger_buy = False

            if trigger_buy:
                in_position = True
                buy_signal_date = buy_sig_date
                buy_trade_date = str(row["trade_date"])
                buy_price = float(row["stock_open"])
                
        elif in_position and buy_price is not None:
            trigger_sell = False
            sell_sig_date = None
            
            if trade_mode == 1:
                if prev_row[sell_col] == 1:
                    trigger_sell = True
                    sell_sig_date = str(prev_row["trade_date"])
            elif trade_mode == 2 and idx >= 2:
                pp_row = result_df.iloc[idx - 2]
                if pp_row[sell_col] == 1 and prev_row[sell_col] == 0:
                    trigger_sell = True
                    sell_sig_date = str(pp_row["trade_date"])
                    
            if trigger_sell:
                sell_trade_date = str(row["trade_date"])
                sell_price = float(row["stock_open"])
                trade_return = sell_price / buy_price - 1
                trades.append({
                    "buy_signal_date": buy_signal_date,
                    "buy_trade_date": buy_trade_date,
                    "buy_price": buy_price,
                    "sell_signal_date": sell_sig_date,
                    "sell_trade_date": sell_trade_date,
                    "sell_price": sell_price,
                    "trade_return": trade_return,
                })
                in_position = False
                buy_signal_date = None
                buy_trade_date = None
                buy_price = None

    trade_df = pd.DataFrame(trades)
    if trade_df.empty:
        return trade_df, {
            "completed_trades": 0,
            "win_rate": 0.0,
            "final_equity": 1.0,
            "period_return": 0.0,
            "open_position": in_position,
            "open_position_buy_signal_date": buy_signal_date,
            "open_position_buy_trade_date": buy_trade_date,
            "open_position_buy_price": buy_price,
        }

    final_equity = float((1 + trade_df["trade_return"]).prod())
    win_rate = float((trade_df["trade_return"] > 0).mean())
    return trade_df, {
        "completed_trades": int(len(trade_df)),
        "win_rate": win_rate,
        "final_equity": final_equity,
        "period_return": final_equity - 1,
        "open_position": in_position,
        "open_position_buy_signal_date": buy_signal_date,
        "open_position_buy_trade_date": buy_trade_date,
        "open_position_buy_price": buy_price,
    }


def build_point_in_time_signal_table(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    float_df: pd.DataFrame,
    security_type: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    target_dates = stock_df[
        (stock_df["trade_date"] >= start_date) & (stock_df["trade_date"] <= end_date)
    ]["trade_date"].tolist()

    pt_rows: list[pd.Series] = []
    for trade_date in target_dates:
        stock_slice = stock_df[stock_df["trade_date"] <= trade_date].reset_index(drop=True)
        index_slice = index_df[index_df["trade_date"] <= trade_date].reset_index(drop=True)
        float_slice = float_df[float_df["trade_date"] <= trade_date].reset_index(drop=True)
        daily_result = calculate_mxs_indicators(stock_slice, index_slice, float_slice, security_type)
        pt_rows.append(daily_result.iloc[-1])

    if not pt_rows:
        return pd.DataFrame(columns=[
            "trade_date",
            "stock_open",
            "retail_line",
            "build_position_signal",
            "buy_breakout_signal",
            "buy_bottom_signal",
            "sell_signal",
            "display_buy_signal",
            "exec_buy_signal",
            "exec_sell_signal",
            "main_force_line",
        ])

    return pd.DataFrame(pt_rows).reset_index(drop=True)


def calculate_mxs_scan_signals(stock_df: pd.DataFrame) -> pd.DataFrame:
    close = stock_df["close"]
    low = stock_df["low"]
    
    v6 = safe_divide(close, ref(close, 3)) >= 1.1
    v7 = backset(v6, 3)
    buy_breakout_signal = np.where(v7 & (count_condition(v7, 3) == 1), 1, 0)

    peak_bars = peakbars_close(close, percent=15, occurrence=1)
    head = pd.Series(np.where(peak_bars < 10, 100.0, 0.0), index=stock_df.index)
    sell_signal = np.where(head > ref(head, 1).fillna(0), 1, 0)

    trough_bars = troughbars_close(close, percent=15, occurrence=1)
    bottom = pd.Series(np.where(trough_bars < 10, 50.0, 0.0), index=stock_df.index)
    buy_bottom_signal = np.where(bottom > ref(bottom, 1).fillna(0), 1, 0)

    display_buy_signal = np.where((buy_breakout_signal > 0) | (buy_bottom_signal > 0), 1, 0)

    main_force_line = safe_divide(close - rolling_llv(low, 30), rolling_hhv(close, 30) - rolling_llv(low, 30)) * 100
    prev_main_force = ref(main_force_line, 1)
    exec_sell_signal = np.where(
        cross_down(main_force_line, 80)
        | ((main_force_line < prev_main_force) & (prev_main_force > 80)),
        1,
        0,
    )

    # 深度整合 OBV 逻辑
    obv_df = calculate_obv(close, stock_df["vol"])
    obv_status, obv_slope = get_obv_status(obv_df["obv"], window=5)

    return pd.DataFrame(
        {
            "trade_date": stock_df["trade_date"],
            "buy_breakout_signal": buy_breakout_signal,
            "buy_bottom_signal": buy_bottom_signal,
            "sell_signal": sell_signal,
            "display_buy_signal": display_buy_signal,
            "exec_buy_signal": np.zeros(len(stock_df), dtype=int),
            "exec_sell_signal": exec_sell_signal,
            "obv_status": obv_status,
            "obv_trend_score": obv_slope,
        }
    )



def build_scan_indicator_result(
    pro,
    ts_code: str,
    security_name: str,
    start_date: str,
    end_date: str,
    scan_type: str,
    index_code: Optional[str] = None,
    security_type: str = "stock",
) -> tuple[pd.DataFrame, dict]:
    resolved_index_code = index_code or infer_index_code_from_ts_code(ts_code)
    if scan_type == "sell":
        data_start_date = extend_start_date(start_date, buffer_days=60)
        stock_df = fetch_daily_data(pro, ts_code, security_type, data_start_date, end_date, is_index=False)
        result_df = calculate_mxs_scan_signals(stock_df)
    else:
        if not resolved_index_code:
            raise ValueError(f"无法根据代码推断指数代码，请通过 --index-code 指定: {ts_code}")
        data_start_date = extend_start_date(start_date, buffer_days=180)
        stock_df = fetch_daily_data(pro, ts_code, security_type, data_start_date, end_date, is_index=False)
        index_df = fetch_daily_data(pro, resolved_index_code, security_type, data_start_date, end_date, is_index=True)
        float_df = fetch_float_share(pro, ts_code, data_start_date, end_date, security_type)
        result_df = calculate_mxs_indicators(stock_df, index_df, float_df, security_type)[["trade_date", "exec_buy_signal", "exec_sell_signal", "display_buy_signal", "sell_signal", "buy_breakout_signal", "buy_bottom_signal", "obv_status", "obv_trend_score"]]

    result_df = result_df[(result_df["trade_date"] >= start_date) & (result_df["trade_date"] <= end_date)].reset_index(drop=True)
    metadata = {
        "input_code": ts_code,
        "ts_code": ts_code,
        "security_type": security_type,
        "name": security_name,
        "start_date": start_date,
        "end_date": end_date,
        "index_code": resolved_index_code,
    }
    return result_df, metadata



def build_indicator_result(
    pro,
    code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    index_code: Optional[str] = None,
    include_pt_signals: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    security = resolve_security(pro, code)
    resolved_ts_code = security["ts_code"]
    security_type = security["security_type"]
    security_name = security["name"]

    if not start_date or not end_date:
        default_start, default_end = get_default_backtest_dates()
        start_date = start_date or default_start
        end_date = end_date or default_end

    resolved_index_code = index_code or infer_index_code_from_ts_code(resolved_ts_code)
    if not resolved_index_code:
        raise ValueError(f"无法根据代码推断指数代码，请通过 --index-code 指定: {code}")

    data_start_date = extend_start_date(start_date, buffer_days=180)
    stock_df = fetch_daily_data(pro, resolved_ts_code, security_type, data_start_date, end_date, is_index=False)
    index_df = fetch_daily_data(pro, resolved_index_code, security_type, data_start_date, end_date, is_index=True)
    float_df = fetch_float_share(pro, resolved_ts_code, data_start_date, end_date, security_type)

    result_df = calculate_mxs_indicators(stock_df, index_df, float_df, security_type)
    result_df = result_df[(result_df["trade_date"] >= start_date) & (result_df["trade_date"] <= end_date)].reset_index(drop=True)

    if include_pt_signals:
        pt_signal_df = build_point_in_time_signal_table(
            stock_df=stock_df,
            index_df=index_df,
            float_df=float_df,
            security_type=security_type,
            start_date=start_date,
            end_date=end_date,
        )
    else:
        pt_signal_df = pd.DataFrame(columns=result_df.columns)

    metadata = {
        "input_code": code,
        "ts_code": resolved_ts_code,
        "security_type": security_type,
        "name": security_name,
        "start_date": start_date,
        "end_date": end_date,
        "index_code": resolved_index_code,
    }
    return result_df, pt_signal_df, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 TuShare 获取股票或ETF数据并计算同花顺公式指标")
    parser.add_argument("--code", required=True, help="6位证券代码或完整 ts_code，例如 300760 / 510300 / 300760.SZ")
    parser.add_argument("--start-date", default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--index-code", default=None, help="指数代码，例如 399006.SZ")
    parser.add_argument("--trade-mode", type=int, choices=[1, 2], default=1, help="回测模式: 1=出信号即按次日开盘价买卖 2=信号消失后的次日开盘价买卖")
    parser.add_argument("--use-obv-filter", action="store_true", default=False, help="启用 OBV 过滤：只有 OBV 向上时才触发买入")
    parser.add_argument("--output-csv", default=None, help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pro = init_tushare()
    if pro is None:
        raise RuntimeError("未能初始化 Tushare，请检查 TUSHARE_TOKEN 环境变量")

    result_df, pt_signal_df, metadata = build_indicator_result(
        pro=pro,
        code=args.code,
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
    )

    output_path = build_output_path(metadata["ts_code"], args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"证券: {metadata['name']} ({metadata['ts_code']}, {metadata['security_type']})")
    print(f"指标结果已保存到: {output_path}")
    print(f"回测区间: {metadata['start_date']} - {metadata['end_date']}")

    signal_df = result_df[(result_df["buy_bottom_signal"] == 1) | (result_df["sell_signal"] == 1)]
    print("\n显示版买卖信号记录:")
    if signal_df.empty:
        print("未发现买卖信号")
    else:
        print(signal_df.to_string(index=False))

    pt_exec_signal_df = pt_signal_df[(pt_signal_df["exec_buy_signal"] == 1) | (pt_signal_df["exec_sell_signal"] == 1)]
    print("\n执行版买卖信号记录(逐日时点重算):")
    if pt_exec_signal_df.empty:
        print("未发现执行版买卖信号")
    else:
        print(pt_exec_signal_df.to_string(index=False))

    trade_df, summary = run_signal_backtest(pt_signal_df, trade_mode=args.trade_mode, use_obv_filter=args.use_obv_filter)
    print(f"\n回测交易记录 (模式 {args.trade_mode}):")
    if trade_df.empty:
        print("无已完成交易")
    else:
        display_trade_df = trade_df.copy()
        display_trade_df["buy_price"] = display_trade_df["buy_price"].map(lambda x: f"{x:.2f}")
        display_trade_df["sell_price"] = display_trade_df["sell_price"].map(lambda x: f"{x:.2f}")
        display_trade_df["trade_return"] = display_trade_df["trade_return"].map(lambda x: f"{x:.2%}")
        print(display_trade_df.to_string(index=False))

    print("\n回测汇总:")
    print(f"完成交易笔数: {summary['completed_trades']}")
    print(f"胜率: {summary['win_rate']:.2%}")
    print(f"周期收益率: {summary['period_return']:.2%}")
    if summary["open_position"]:
        print("周期结束时仍有未平仓仓位:")
        print(
            f"买入信号日={summary['open_position_buy_signal_date']}, "
            f"买入执行日={summary['open_position_buy_trade_date']}, "
            f"买入价={summary['open_position_buy_price']:.2f}"
        )
    else:
        print("周期结束时无持仓")


if __name__ == "__main__":
    main()
