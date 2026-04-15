import argparse
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from backtest import get_default_backtest_dates, normalize_code_input, extend_start_date, prewarm_all_caches
from obv import calculate_obv
import backtest
from mxs_indicator import build_formula_review_result, build_formula_review_scan_result
from util import init_tushare


load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用新公式扫描最近3个交易日内出现买点或卖点且之后未被反向信号覆盖的股票")
    parser.add_argument("--codes", default=None, help="指定扫描代码，逗号分隔，支持6位代码或完整 ts_code")
    parser.add_argument("--market", choices=["主板", "创业板", "科创板", "CDR", "北交所"], default=None, help="按市场过滤股票范围；仅在未指定 --codes 时生效")
    parser.add_argument("--start-date", default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--scan-type", choices=["buy", "sell", "both"], default="buy", help="扫描类型: buy/sell/both")
    parser.add_argument("--security-type", choices=["stock", "etf", "both"], default="stock", help="证券类型选择：stock(仅股票), etf(仅场内基金), both(全部)")
    parser.add_argument("--filter-obv", action="store_true", help="是否过滤掉 OBV 不是向上趋势的股票，默认不过滤全部输出")
    parser.add_argument("--output-csv", default=None, help="输出 CSV 路径，仅在单模式下生效")
    return parser.parse_args()


def get_recent_trade_dates(result_df: pd.DataFrame, end_date: str, days: int = 3) -> list[str]:
    trade_dates = result_df[result_df["trade_date"] <= end_date]["trade_date"].drop_duplicates().sort_values()
    return trade_dates.tail(days).astype(str).tolist()


def get_scan_window(end_date: str) -> tuple[str, str]:
    scan_start = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=270)).strftime("%Y%m%d")
    return scan_start, end_date


def find_recent_sell_without_buy(
    result_df: pd.DataFrame,
    end_date: str,
    sell_col: str = "sell_signal",
    buy_col: str = "display_buy_signal",
) -> str | None:
    recent_trade_dates = get_recent_trade_dates(result_df, end_date)
    recent_sell_df = result_df[
        result_df["trade_date"].astype(str).isin(recent_trade_dates)
        & (result_df[sell_col] > 0)
    ]
    if recent_sell_df.empty:
        return None

    last_sell_signal_date = str(recent_sell_df["trade_date"].max())
    later_buy_df = result_df[
        (result_df["trade_date"] > last_sell_signal_date)
        & (result_df["trade_date"] <= end_date)
        & (result_df[buy_col] > 0)
    ]
    if not later_buy_df.empty:
        return None

    return last_sell_signal_date


def build_output_path(end_date: str, output_csv: str | None, scan_type: str, market: str | None = None) -> Path:
    if output_csv:
        return Path(output_csv).resolve()
    market_suffix = f"_{market}" if market else ""
    if scan_type == "sell":
        filename = f"recent_sell_without_buy{market_suffix}_{end_date}.csv"
    else:
        filename = f"recent_buy_without_sell{market_suffix}_{end_date}.csv"
    return (Path(__file__).resolve().parent / "data" / filename).resolve()


def print_and_save_result(result_df: pd.DataFrame, output_path: Path, empty_message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n扫描结果已保存到: {output_path}")
    if result_df.empty:
        print(empty_message)
    else:
        print(result_df.to_string(index=False))


def sort_result_df(result_df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    return result_df.sort_values([date_column, "ts_code"], ascending=[False, True]).reset_index(drop=True)


def find_recent_buy_without_sell(
    result_df: pd.DataFrame,
    end_date: str,
    buy_col: str = "display_buy_signal",
    sell_col: str = "sell_signal",
) -> str | None:
    recent_trade_dates = get_recent_trade_dates(result_df, end_date)
    recent_buy_df = result_df[
        result_df["trade_date"].astype(str).isin(recent_trade_dates)
        & (result_df[buy_col] > 0)
    ]
    if recent_buy_df.empty:
        return None

    last_buy_signal_date = str(recent_buy_df["trade_date"].max())
    later_sell_df = result_df[
        (result_df["trade_date"] > last_buy_signal_date)
        & (result_df["trade_date"] <= end_date)
        & (result_df[sell_col] > 0)
    ]
    if not later_sell_df.empty:
        return None

    return last_buy_signal_date


def load_stock_universe(pro, codes_arg: str | None, market: str | None = None, security_type: str = "stock") -> list[dict]:
    if codes_arg:
        codes = [normalize_code_input(code) for code in codes_arg.split(",") if code.strip()]
        return [{"input_code": code} for code in codes]

    items = []
    
    if security_type in ("stock", "both"):
        stock_df = pro.stock_basic(exchange="SSE,SZSE", list_status="L", market=market or "", fields="ts_code,symbol,name")
        if stock_df is not None and not stock_df.empty:
            stock_df["input_code"] = stock_df["symbol"].astype(str)
            stock_df["security_type"] = "stock"
            items.extend(stock_df[["input_code", "ts_code", "name", "security_type"]].to_dict(orient="records"))

    if security_type in ("etf", "both"):
        fund_df = pro.fund_basic(market="E", status="L", fields="ts_code,name")
        if fund_df is not None and not fund_df.empty:
            fund_df["input_code"] = fund_df["ts_code"].str.split(".").str[0]
            fund_df["security_type"] = "etf"
            items.extend(fund_df[["input_code", "ts_code", "name", "security_type"]].to_dict(orient="records"))
            
    return items


def main() -> None:
    args = parse_args()
    pro = init_tushare()
    if pro is None:
        raise RuntimeError("未能初始化 Tushare，请检查 TUSHARE_TOKEN 环境变量")

    _start_date, end_date = (args.start_date, args.end_date)
    if not end_date:
        _default_start, default_end = get_default_backtest_dates()
        end_date = default_end
    start_date, end_date = get_scan_window(end_date)

    universe = load_stock_universe(pro, args.codes, args.market, args.security_type)
    if not universe:
        print("未获取到可扫描股票列表")
        return

    data_start_date = extend_start_date(start_date, buffer_days=270)
    prewarm_all_caches(pro, data_start_date, end_date)

    # 非 --codes 模式下，利用 daily_basic 中的 pe_ttm 过滤亏损股（pe_ttm 为空表示亏损）
    if not args.codes and backtest._mem_range_covers(data_start_date, end_date):
        loss_codes: set[str] = set()
        for code, df in backtest._mem_float_by_code.items():
            if df.empty:
                continue
            latest = df.iloc[-1]
            pe = latest.get("pe_ttm")
            if pd.isna(pe):
                loss_codes.add(code)
        before = len(universe)
        universe = [item for item in universe if item.get("ts_code") not in loss_codes]
        print(f"已过滤亏损股 {before - len(universe)} 支，剩余 {len(universe)} 支\n")

    # OBV 的过滤与状态已经内嵌在 backtest.py 的核心函数中，以保障复权数据的精确度。
    buy_matches: list[dict] = []
    sell_matches: list[dict] = []
    total = len(universe)

    for idx, item in enumerate(universe, start=1):
        input_code = normalize_code_input(item["input_code"])
        ts_code = item.get("ts_code")
        security_name = item.get("name")
        print(f"[{idx}/{total}] 扫描 {input_code}")
        try:
            if ts_code and security_name:
                result_df, metadata = build_formula_review_scan_result(
                    pro=pro,
                    ts_code=ts_code,
                    security_name=security_name,
                    start_date=start_date,
                    end_date=end_date,
                    security_type=item.get("security_type", "stock"),
                )
            else:
                result_df, metadata = build_formula_review_result(
                    pro=pro,
                    code=input_code,
                    start_date=start_date,
                    end_date=end_date,
                    index_code=None,
                )
        except Exception as exc:
            print(f"跳过 {input_code}: {exc}")
            continue

        if metadata["security_type"] not in ("stock", "etf", "fund"):
            continue

        if "ST" in str(metadata["name"]).upper():
            continue

        recent_trade_dates = get_recent_trade_dates(result_df, end_date)
        result_df = result_df[result_df["trade_date"].astype(str).isin(recent_trade_dates)].reset_index(drop=True)
        if result_df.empty:
            continue

        if ts_code and security_name:
            metadata["name"] = security_name
            metadata["ts_code"] = ts_code

        if args.scan_type in {"buy", "both"}:
            buy_signal_date = find_recent_buy_without_sell(
                result_df,
                end_date=end_date,
                buy_col="display_buy_signal",
                sell_col="sell_signal",
            )
            if buy_signal_date is not None:
                target_rows = result_df[result_df["trade_date"] == buy_signal_date]
                if not target_rows.empty:
                    obv_status = target_rows.iloc[0].get("obv_status", 0)
                    obv_trend_score = target_rows.iloc[0].get("obv_trend_score", 0.0)
                    if args.filter_obv and obv_status == 0:
                        pass
                    else:
                        buy_matches.append(
                            {
                                "ts_code": metadata["ts_code"],
                                "name": metadata["name"],
                                "buy_signal_date": buy_signal_date,
                                "obv_status": "强势多头📈" if obv_status == 1 else "弱势空头📉",
                                "trend_score": round(obv_trend_score, 4),
                            }
                        )

        if args.scan_type in {"sell", "both"}:
            sell_signal_date = find_recent_sell_without_buy(
                result_df,
                end_date=end_date,
                sell_col="sell_signal",
                buy_col="display_buy_signal",
            )
            if sell_signal_date is not None:
                target_rows = result_df[result_df["trade_date"] == sell_signal_date]
                if not target_rows.empty:
                    obv_status = target_rows.iloc[0].get("obv_status", 0)
                    obv_trend_score = target_rows.iloc[0].get("obv_trend_score", 0.0)
                    if args.filter_obv and obv_status == 0:
                        pass
                    else:
                        sell_matches.append(
                            {
                                "ts_code": metadata["ts_code"],
                                "name": metadata["name"],
                                "sell_signal_date": sell_signal_date,
                                "obv_status": "强势多头📈" if obv_status == 1 else "弱势空头📉",
                                "trend_score": round(obv_trend_score, 4),
                            }
                        )

    if args.scan_type in {"buy", "both"}:
        buy_result_df = sort_result_df(pd.DataFrame(buy_matches), "buy_signal_date")
        buy_output_path = build_output_path(
            end_date=end_date,
            output_csv=args.output_csv if args.scan_type == "buy" else None,
            scan_type="buy",
            market=args.market,
        )
        print_and_save_result(buy_result_df, buy_output_path, "最近一周内没有符合条件的新公式买点股票")

    if args.scan_type in {"sell", "both"}:
        sell_result_df = sort_result_df(pd.DataFrame(sell_matches), "sell_signal_date")
        sell_output_path = build_output_path(
            end_date=end_date,
            output_csv=args.output_csv if args.scan_type == "sell" else None,
            scan_type="sell",
            market=args.market,
        )
        print_and_save_result(sell_result_df, sell_output_path, "最近一周内没有符合条件的新公式卖点股票")


if __name__ == "__main__":
    main()
