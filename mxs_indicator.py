import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest import (
    calculate_mxs_indicators,
    calculate_mxs_scan_signals,
    extend_start_date,
    fetch_daily_data,
    fetch_float_share,
    get_default_backtest_dates,
    infer_index_code_from_ts_code,
    normalize_code_input,
    resolve_security,
)
from util import init_tushare


load_dotenv()


def build_review_output_path(ts_code: str, output_csv: str | None) -> Path:
    if output_csv:
        return Path(output_csv).resolve()
    return (Path(__file__).resolve().parent / "data" / f"ths_formula_review_{ts_code}.csv").resolve()





def build_formula_review_scan_result(
    pro,
    ts_code: str,
    security_name: str,
    start_date: str,
    end_date: str,
    security_type: str = "stock",
) -> tuple[pd.DataFrame, dict]:
    data_start_date = extend_start_date(start_date, buffer_days=270)
    stock_df = fetch_daily_data(pro, ts_code, security_type, data_start_date, end_date, is_index=False)

    resolved_index_code = infer_index_code_from_ts_code(ts_code)
    if resolved_index_code:
        try:
            index_df = fetch_daily_data(pro, resolved_index_code, security_type, data_start_date, end_date, is_index=True)
        except Exception:
            index_df = pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "vol"])
    else:
        index_df = pd.DataFrame(columns=["trade_date", "open", "high", "low", "close", "vol"])

    float_df = fetch_float_share(pro, ts_code, data_start_date, end_date, security_type)

    if not index_df.empty:
        result_df = calculate_mxs_indicators(stock_df, index_df, float_df, security_type)
        keep_cols = ["trade_date", "buy_breakout_signal", "buy_bottom_signal", "sell_signal",
                     "display_buy_signal", "exec_buy_signal", "exec_sell_signal"]
        result_df = result_df[[c for c in keep_cols if c in result_df.columns]]
    else:
        result_df = calculate_mxs_scan_signals(stock_df)

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

    result_df = calculate_mxs_indicators(stock_df, index_df, float_df, security_type)
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
