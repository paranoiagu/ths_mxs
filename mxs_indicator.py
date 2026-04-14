import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest import (
    build_output_path,
    count_condition,
    cross,
    extend_start_date,
    fetch_daily_data,
    fetch_float_share,
    get_default_backtest_dates,
    infer_index_code_from_ts_code,
    normalize_code_input,
    peakbars_close,
    ref,
    resolve_security,
    rolling_avedev,
    rolling_hhv,
    rolling_llv,
    rolling_ma,
    safe_divide,
    sma_tdx,
    troughbars_close,
)
from util import init_tushare


load_dotenv()


def backset(condition: pd.Series, periods: int) -> pd.Series:
    values = condition.fillna(False).astype(bool).to_numpy()
    result = np.zeros(len(values), dtype=bool)
    for idx, is_true in enumerate(values):
        if not is_true:
            continue
        start = max(0, idx - periods + 1)
        result[start : idx + 1] = True
    return pd.Series(result, index=condition.index)

def build_review_output_path(ts_code: str, output_csv: str | None) -> Path:
    if output_csv:
        return Path(output_csv).resolve()
    return (Path(__file__).resolve().parent / "data" / f"ths_formula_review_{ts_code}.csv").resolve()


def calculate_formula_review_indicators(
    stock_df: pd.DataFrame,
    index_df: pd.DataFrame,
    float_df: pd.DataFrame,
    security_type: str,
) -> pd.DataFrame:
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
    df[["index_high", "index_low", "index_close"]] = df[["index_high", "index_low", "index_close"]].ffill()
    df["float_share"] = pd.to_numeric(df["float_share"], errors="coerce").ffill()

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

    if security_type == "stock" and df["float_share"].notna().any():
        capital_hands = df["float_share"] * 100.0
        vare = ref(low, 1) * 0.9
        varf = low * 0.9
        var10 = safe_divide(varf * vol + vare * (capital_hands - vol), capital_hands)
        var11 = var10.ewm(span=30, adjust=False, min_periods=1).mean()
        var12 = close - ref(close, 1)
        var13 = pd.Series(np.maximum(var12, 0), index=df.index)
        var14 = var12.abs()
        var15 = safe_divide(sma_tdx(var13, 7, 1), sma_tdx(var14, 7, 1)) * 100
        var16 = safe_divide(sma_tdx(var13, 13, 1), sma_tdx(var14, 13, 1)) * 100
        var17 = pd.Series(np.arange(1, len(df) + 1, dtype=float), index=df.index)
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
            95.0,
            0.0,
        )
    else:
        capital_hands = pd.Series(np.nan, index=df.index, dtype=float)
        build_position_signal = np.zeros(len(df), dtype=float)

    v6 = safe_divide(close, ref(close, 3)) >= 1.1
    v7 = backset(v6, 3)
    buy_breakout_signal = np.where(v7 & (count_condition(v7, 3) == 1), 50.0, 0.0)

    peak_bars = peakbars_close(close, percent=15, occurrence=1)
    head = pd.Series(np.where(peak_bars < 10, 100.0, 0.0), index=df.index)
    sell_signal = np.where(head > ref(head, 1).fillna(0), 50.0, 0.0)

    trough_bars = troughbars_close(close, percent=15, occurrence=1)
    bottom = pd.Series(np.where(trough_bars < 10, 50.0, 0.0), index=df.index)
    buy_bottom_signal = np.where(bottom > ref(bottom, 1).fillna(0), 40.0, 0.0)

    display_buy_signal = np.where((buy_breakout_signal > 0) | (buy_bottom_signal > 0), 1, 0)
    main_force_line = safe_divide(close - rolling_llv(low, 30), rolling_hhv(close, 30) - rolling_llv(low, 30)) * 100

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
            "main_force_line": main_force_line,
            "peak_bars_review": peak_bars,
            "trough_bars_review": trough_bars,
            "float_share": df["float_share"],
            "capital_hands": capital_hands,
        }
    )


def calculate_formula_review_scan_signals(stock_df: pd.DataFrame) -> pd.DataFrame:
    close = stock_df["close"]
    v6 = safe_divide(close, ref(close, 3)) >= 1.1
    v7 = backset(v6, 3)
    buy_breakout_signal = np.where(v7 & (count_condition(v7, 3) == 1), 50.0, 0.0)

    peak_bars = peakbars_close(close, percent=15, occurrence=1)
    head = pd.Series(np.where(peak_bars < 10, 100.0, 0.0), index=stock_df.index)
    sell_signal = np.where(head > ref(head, 1).fillna(0), 50.0, 0.0)

    trough_bars = troughbars_close(close, percent=15, occurrence=1)
    bottom = pd.Series(np.where(trough_bars < 10, 50.0, 0.0), index=stock_df.index)
    buy_bottom_signal = np.where(bottom > ref(bottom, 1).fillna(0), 40.0, 0.0)

    display_buy_signal = np.where((buy_breakout_signal > 0) | (buy_bottom_signal > 0), 1, 0)

    return pd.DataFrame(
        {
            "trade_date": stock_df["trade_date"],
            "buy_breakout_signal": buy_breakout_signal,
            "buy_bottom_signal": buy_bottom_signal,
            "sell_signal": sell_signal,
            "display_buy_signal": display_buy_signal,
        }
    )



def build_formula_review_scan_result(
    pro,
    ts_code: str,
    security_name: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, dict]:
    security_type = "stock"
    data_start_date = extend_start_date(start_date, buffer_days=270)
    stock_df = fetch_daily_data(pro, ts_code, security_type, data_start_date, end_date, is_index=False)
    result_df = calculate_formula_review_scan_signals(stock_df)
    result_df = result_df[(result_df["trade_date"] >= start_date) & (result_df["trade_date"] <= end_date)].reset_index(drop=True)

    metadata = {
        "input_code": ts_code,
        "ts_code": ts_code,
        "security_type": security_type,
        "name": security_name,
        "start_date": start_date,
        "end_date": end_date,
        "index_code": None,
    }
    return result_df, metadata



def build_formula_review_result(
    pro,
    code: str,
    start_date: str | None = None,
    end_date: str | None = None,
    index_code: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    normalized_code = normalize_code_input(code)
    security = resolve_security(pro, normalized_code)
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

    result_df = calculate_formula_review_indicators(stock_df, index_df, float_df, security_type)
    result_df = result_df[(result_df["trade_date"] >= start_date) & (result_df["trade_date"] <= end_date)].reset_index(drop=True)

    metadata = {
        "input_code": code,
        "ts_code": resolved_ts_code,
        "security_type": security_type,
        "name": security_name,
        "start_date": start_date,
        "end_date": end_date,
        "index_code": resolved_index_code,
    }
    return result_df, metadata



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="更贴近同花顺原显示公式的独立对照脚本")
    parser.add_argument("--code", required=True, help="6位证券代码或完整 ts_code，例如 300760 / 300760.SZ")
    parser.add_argument("--start-date", default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--index-code", default=None, help="指数代码，例如 399006.SZ")
    parser.add_argument("--output-csv", default=None, help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pro = init_tushare()
    if pro is None:
        raise RuntimeError("未能初始化 Tushare，请检查 TUSHARE_TOKEN 环境变量")

    result_df, metadata = build_formula_review_result(
        pro=pro,
        code=args.code,
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
    )

    output_path = build_review_output_path(metadata["ts_code"], args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"证券: {metadata['name']} ({metadata['ts_code']}, {metadata['security_type']})")
    print(f"对照版指标结果已保存到: {output_path}")
    print(f"区间: {metadata['start_date']} - {metadata['end_date']}")


    breakout_df = result_df[result_df["buy_breakout_signal"] > 0]
    print("\n突破型买点记录:")
    if breakout_df.empty:
        print("未发现突破型买点")
    else:
        print(breakout_df.to_string(index=False))

    bottom_df = result_df[result_df["buy_bottom_signal"] > 0]
    print("\n底部抄底买点记录:")
    if bottom_df.empty:
        print("未发现底部抄底买点")
    else:
        print(bottom_df.to_string(index=False))

    sell_df = result_df[result_df["sell_signal"] > 0]
    print("\n卖点记录:")
    if sell_df.empty:
        print("未发现卖点")
    else:
        print(sell_df.to_string(index=False))


if __name__ == "__main__":
    main()
