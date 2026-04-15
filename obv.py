import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from backtest import get_default_backtest_dates, extend_start_date, resolve_security, fetch_daily_data
from util import init_tushare

load_dotenv()


def calculate_obv(close: pd.Series, vol: pd.Series) -> pd.DataFrame:
    """计算 OBV 系列指标：返回包含 obv 和 ma_obv 均线的 DataFrame"""
    direction = np.sign(close.diff())
    # 第一天的 diff 为 NaN，设为 0
    direction.iloc[0] = 0
    obv = (vol * direction).cumsum()
    return pd.DataFrame({"obv": obv})

def get_obv_status(obv: pd.Series, ma_window: int = 30) -> tuple[bool, pd.Series]:
    """
    通过行情软件常用的法则判定：
    计算 OBV 的 MA(ma_window) 均线
    返回: (当前OBV是否大于均线也就是金叉向上状态, ma_obv序列)
    """
    ma_obv = obv.rolling(window=ma_window, min_periods=1).mean()
    
    is_above_ma = False
    if len(obv) > 0 and len(ma_obv) > 0:
        is_above_ma = obv.iloc[-1] > ma_obv.iloc[-1]
        
    return is_above_ma, ma_obv


def parse_args():
    parser = argparse.ArgumentParser(description="获取单只股票或ETF的 OBV 状态")
    parser.add_argument("--code", required=True, help="股票或ETF代码，如 000001、510300 等")
    parser.add_argument("--start-date", default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--window", type=int, default=30, help="MAOBV 的均线天数，行情软件标准默认使用30天")
    return parser.parse_args()


def main():
    args = parse_args()
    
    pro = init_tushare()
    if pro is None:
        print("Tushare 初始化失败，请检查环境变量 TUSHARE_TOKEN！")
        return
        
    start_date = args.start_date
    end_date = args.end_date
    if not start_date or not end_date:
        ds, de = get_default_backtest_dates()
        start_date = start_date or ds
        end_date = end_date or de
        
    try:
        security = resolve_security(pro, args.code)
    except Exception as e:
        print(f"解析代码失败: {e}")
        return
        
    ts_code = security["ts_code"]
    sec_name = security["name"]
    sec_type = security["security_type"]
    
    # 给前面的数据预留足够的时间窗口，以确保累加平滑
    data_start_date = extend_start_date(start_date, buffer_days=180)
    
    print(f"正在获取 {sec_name} ({ts_code}) 的日线数据 ...")
    try:
        df = fetch_daily_data(pro, ts_code, sec_type, data_start_date, end_date, is_index=False)
    except Exception as e:
        print(f"获取数据失败: {e}")
        return
        
    if df.empty:
        print("在要求的日期区间内没有获取到日线数据")
        return
        
    # 确保时间正序排列
    df = df.sort_values("trade_date").reset_index(drop=True)
    
    # 计算 OBV 以及 30日 均线
    obv_df = calculate_obv(df["close"], df["vol"])
    df["obv"] = obv_df["obv"]
    
    # 按照标准行情软件指标判定法则
    is_above_ma, ma_obv_series = get_obv_status(df["obv"], ma_window=args.window)
    df["ma_obv"] = ma_obv_series
    
    mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
    period_df = df[mask].copy()
    
    if period_df.empty:
        print(f"在指定的区间 {start_date} ~ {end_date} 内没有有效数据可以分析！")
        return
        
    print("-" * 50)
    print(f"[{sec_name} - {ts_code}] OBV (能量潮) 定向分析")
    print(f"分析区间: {start_date} 至 {end_date}")
    print(f"对比法则: 当前 OBV 穿越 MAOBV({args.window})")
    
    diff_percent = 0.0
    if df["ma_obv"].iloc[-1] != 0:
        diff_percent = (df["obv"].iloc[-1] / df["ma_obv"].iloc[-1] - 1) * 100
        
    status = f"强势多头 📈 (位于均线之上 {diff_percent:+.2f}%)" if is_above_ma else f"弱势空头 📉 (位于均线之下 {diff_percent:+.2f}%)"
    print(f"当前状态: {status}")
    print("-" * 50)
    
    # 打印最终几天的参考数值
    print("最近 10 个交易日的 OBV vs 均线 数据：")
    
    # 为了直观打印，将 DataFrame float 格式化一下
    recent_5 = df.tail(10)[["trade_date", "close", "vol", "obv", "ma_obv"]].copy()
    for col in ["obv", "ma_obv"]:
        recent_5[col] = recent_5[col].apply(lambda x: f"{x:,.0f}")
        
    print(recent_5.to_string(index=False))


if __name__ == "__main__":
    main()
