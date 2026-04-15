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

def get_obv_status_local(obv: pd.Series, window: int = 5) -> tuple[pd.Series, pd.Series]:
    # Placeholder for older logic or unused function. 
    # The actual get_obv_status is imported from backtest.py.
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="获取单只股票或ETF的 OBV 状态")
    parser.add_argument("--code", required=True, help="股票或ETF代码，如 000001、510300 等")
    parser.add_argument("--start-date", default=None, help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    parser.add_argument("--window", type=int, default=5, help="计算OBV趋势的时间窗口，用户指定默认5天")
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
    
    from backtest import get_obv_status
    
    # 按照 N日趋势 判定法则，获得介于 [-1, 1] 的相关系数得分
    obv_status_series, trend_score_series = get_obv_status(df["obv"], window=args.window)
    df["trend_score"] = trend_score_series
    
    mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
    period_df = df[mask].copy()
    
    if period_df.empty:
        print(f"在指定的区间 {start_date} ~ {end_date} 内没有有效数据可以分析！")
        return
        
    print("-" * 50)
    print(f"[{sec_name} - {ts_code}] OBV 定向分析")
    print(f"分析区间: {start_date} 至 {end_date}")
    print(f"对比法则: 最近 {args.window} 个交易日的趋势系数 [-1, 1]")
    
    rising = False
    score_val = 0.0
    if len(df) > 0:
        score_val = df["trend_score"].iloc[-1]
        if pd.notna(score_val) and score_val > 0:
            rising = True
            
    status = "向上 (Rising) 📈" if rising else "向下/震荡 (Falling/Flat) 📉"
    print(f"当前状态: {status}")
    print(f"趋势得分: {score_val:+.4f} (越接近 +1 意味着该阶段上攻越纯粹)")
    print("-" * 50)
    
    # 打印最终几天的参考数值
    print(f"最近 10 个交易日的 OBV vs 趋势系数 数据：")
    
    # 为了直观打印，将 DataFrame float 格式化一下
    recent_5 = df.tail(10)[["trade_date", "close", "vol", "obv", "trend_score"]].copy()
    for col in ["obv"]:
        recent_5[col] = recent_5[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "NaN")
    recent_5["trend_score"] = recent_5["trend_score"].apply(lambda x: f"{x:+.4f}" if pd.notna(x) else "NaN")
        
    print(recent_5.to_string(index=False))


if __name__ == "__main__":
    main()
