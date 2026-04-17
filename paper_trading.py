"""模拟交易程序：基于 MXS 指标体系的多股票资金统筹买卖

使用真实历史行情数据，从指定日期开始，按照与回测相同的信号逻辑（默认 mode=2, OBV 过滤）
进行多股票的资金统筹买卖，并将所有交易流水记录到 CSV。
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

import backtest
from backtest import (
    build_scan_indicator_result,
    extend_start_date,
    get_default_end_date,
    prewarm_all_caches,
)
from util import init_tushare, shift_date_str


load_dotenv()


class PaperAccount:
    """模拟交易账户：管理资金、持仓和交易记录"""

    def __init__(self, initial_capital: float, commission_rate: float, stamp_duty: float):
        self.initial_capital = initial_capital
        self.available_cash = initial_capital
        self.commission_rate = commission_rate
        self.stamp_duty = stamp_duty
        # ts_code -> {shares, buy_price, buy_date, cost, name, signal_date}
        self.positions: dict[str, dict] = {}
        self.trades: list[dict] = []
        self.completed_trades: list[dict] = []

    def buy(self, ts_code: str, name: str, price: float, shares: int,
            signal_date: str, trade_date: str) -> bool:
        """执行买入。返回是否成功（资金不足时返回 False）"""
        amount = price * shares
        commission = amount * self.commission_rate
        total_cost = amount + commission

        if total_cost > self.available_cash + 0.01:
            return False

        self.available_cash -= total_cost
        self.positions[ts_code] = {
            "shares": shares,
            "buy_price": price,
            "buy_date": trade_date,
            "cost": total_cost,
            "name": name,
            "signal_date": signal_date,
        }
        return True

    def sell(self, ts_code: str, price: float, signal_date: str,
             trade_date: str) -> Optional[dict]:
        """执行卖出。返回交易详情 dict，未持仓返回 None"""
        if ts_code not in self.positions:
            return None

        pos = self.positions[ts_code]
        shares = pos["shares"]
        amount = price * shares
        commission = amount * self.commission_rate
        stamp_duty = amount * self.stamp_duty
        total_fee = commission + stamp_duty
        net_proceeds = amount - total_fee

        self.available_cash += net_proceeds
        profit = net_proceeds - pos["cost"]

        result = {
            "ts_code": ts_code,
            "name": pos["name"],
            "shares": shares,
            "buy_date": pos["buy_date"],
            "buy_price": pos["buy_price"],
            "cost": pos["cost"],
            "sell_price": price,
            "amount": amount,
            "total_fee": total_fee,
            "net_proceeds": net_proceeds,
            "profit": profit,
            "signal_date": signal_date,
            "trade_date": trade_date,
        }

        self.completed_trades.append(result)
        del self.positions[ts_code]
        return result

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """计算总资产 = 可用现金 + 持仓市值"""
        pos_value = sum(
            pos["shares"] * current_prices.get(ts_code, pos["buy_price"])
            for ts_code, pos in self.positions.items()
        )
        return self.available_cash + pos_value


class PaperTrader:
    """模拟交易引擎：按交易日驱动，先卖后买"""

    def __init__(self, pro, account: PaperAccount, trade_mode: int = 2,
                 use_obv_filter: bool = True):
        self.pro = pro
        self.account = account
        self.trade_mode = trade_mode
        self.use_obv_filter = use_obv_filter
        # ts_code -> indicator DataFrame（含 exec_buy_signal, exec_sell_signal, obv_status 等）
        self.indicator_data: dict[str, pd.DataFrame] = {}
        # ts_code -> 股票名称
        self.stock_names: dict[str, str] = {}
        # ts_code -> {trade_date -> open_price}  （未复权真实价格）
        self._open_prices: dict[str, dict[str, float]] = {}
        # ts_code -> {trade_date -> close_price}（未复权真实价格）
        self._close_prices: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # 初始化阶段
    # ------------------------------------------------------------------

    def load_universe(self) -> list[dict]:
        """加载主板股票列表（排除 ST）"""
        stock_df = self.pro.stock_basic(
            exchange="SSE,SZSE", list_status="L", market="主板",
            fields="ts_code,symbol,name",
        )
        if stock_df is None or stock_df.empty:
            return []
        stock_df = stock_df[~stock_df["name"].str.upper().str.contains("ST")]
        return stock_df.to_dict(orient="records")

    def precalculate_indicators(self, universe: list[dict],
                                start_date: str, end_date: str) -> None:
        """预热缓存 → 过滤亏损股 → 逐股票计算指标 → 构建价格索引"""
        # 往前多取几天，保证 mode=2 在 start_date 起就能检测到信号
        indicator_start = shift_date_str(start_date, -10)
        data_start_date = extend_start_date(indicator_start, buffer_days=270)

        # 1) 预热缓存
        prewarm_all_caches(self.pro, data_start_date, end_date)

        # 2) 过滤亏损股（pe_ttm 为空代表亏损）
        filtered = universe
        if backtest._mem_range_covers(data_start_date, end_date):
            loss_codes: set[str] = set()
            for code, df in backtest._mem_float_by_code.items():
                if df.empty:
                    continue
                latest = df.iloc[-1]
                pe = latest.get("pe_ttm")
                if pd.isna(pe):
                    loss_codes.add(code)
            before = len(universe)
            filtered = [item for item in universe if item.get("ts_code") not in loss_codes]
            print(f"已过滤亏损股 {before - len(filtered)} 支，剩余 {len(filtered)} 支")

        # 3) 逐股票计算指标
        total = len(filtered)
        print(f"预计算指标: {total} 支股票 ({indicator_start} ~ {end_date})")

        for idx, item in enumerate(filtered, start=1):
            ts_code = item["ts_code"]
            name = item["name"]

            if idx % 200 == 0 or idx == total:
                print(f"  [{idx}/{total}] 处理中...")

            try:
                result_df, metadata = build_scan_indicator_result(
                    pro=self.pro,
                    ts_code=ts_code,
                    security_name=name,
                    start_date=indicator_start,
                    end_date=end_date,
                    scan_type="buy",
                    security_type="stock",
                )
                if metadata.get("security_type") != "stock":
                    continue
                self.indicator_data[ts_code] = result_df
                self.stock_names[ts_code] = name
            except Exception:
                continue

        print(f"指标预计算完成: {len(self.indicator_data)} 支股票有效")

        # 4) 构建价格索引（未复权真实价格，从内存缓存取）
        self._build_price_lookup()

    def _build_price_lookup(self) -> None:
        """从 _mem_stock_by_code 构建 open / close 快速索引"""
        for ts_code in self.indicator_data:
            raw_df = backtest._mem_stock_by_code.get(ts_code)
            if raw_df is None or raw_df.empty:
                continue
            td = raw_df["trade_date"].astype(str)
            self._open_prices[ts_code] = dict(zip(td, raw_df["open"].astype(float)))
            self._close_prices[ts_code] = dict(zip(td, raw_df["close"].astype(float)))

    # ------------------------------------------------------------------
    # 价格查询
    # ------------------------------------------------------------------

    def _get_open_price(self, ts_code: str, trade_date: str) -> Optional[float]:
        return self._open_prices.get(ts_code, {}).get(trade_date)

    def _get_close_price(self, ts_code: str, trade_date: str) -> Optional[float]:
        return self._close_prices.get(ts_code, {}).get(trade_date)

    def _get_latest_close(self, ts_code: str, end_date: str) -> Optional[float]:
        """获取截至 end_date 的最新收盘价"""
        close_dict = self._close_prices.get(ts_code)
        if not close_dict:
            return None
        for td in sorted(close_dict, reverse=True):
            if td <= end_date:
                return close_dict[td]
        return None

    def _get_all_current_prices(self, trade_date: str) -> dict[str, float]:
        """获取所有持仓在指定日期的开盘价"""
        prices: dict[str, float] = {}
        for ts_code in self.account.positions:
            price = self._get_open_price(ts_code, trade_date)
            if price is not None:
                prices[ts_code] = price
        return prices

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def _get_trading_dates(self, start_date: str, end_date: str) -> list[str]:
        all_dates: set[str] = set()
        for result_df in self.indicator_data.values():
            mask = (result_df["trade_date"].astype(str) >= start_date) & \
                   (result_df["trade_date"].astype(str) <= end_date)
            all_dates.update(result_df.loc[mask, "trade_date"].astype(str).tolist())
        return sorted(all_dates)

    # ------------------------------------------------------------------
    # 信号检测（mode=2）
    # ------------------------------------------------------------------

    def _check_buy_signal(self, ts_code: str, trade_date: str) -> Optional[str]:
        """检查 mode=2 买入条件：T-2 exec_buy=1, T-1 exec_buy=0，含 OBV 过滤。
        返回信号日期或 None。"""
        result_df = self.indicator_data.get(ts_code)
        if result_df is None:
            return None

        mask = result_df["trade_date"].astype(str) == trade_date
        if not mask.any():
            return None
        idx = mask.idxmax()
        if idx < 2:
            return None

        pp_row = result_df.iloc[idx - 2]
        prev_row = result_df.iloc[idx - 1]

        # mode=2: T-2 信号=1, T-1 信号=0
        if pp_row["exec_buy_signal"] != 1 or prev_row["exec_buy_signal"] != 0:
            return None

        # OBV 过滤：检查 T-1 的 obv_status
        if self.use_obv_filter:
            if float(prev_row.get("obv_status", 0)) <= 0:
                return None

        return str(pp_row["trade_date"])

    def _check_sell_signal(self, ts_code: str, trade_date: str) -> Optional[str]:
        """检查 mode=2 卖出条件：T-2 exec_sell=1, T-1 exec_sell=0。"""
        result_df = self.indicator_data.get(ts_code)
        if result_df is None:
            return None

        mask = result_df["trade_date"].astype(str) == trade_date
        if not mask.any():
            return None
        idx = mask.idxmax()
        if idx < 2:
            return None

        pp_row = result_df.iloc[idx - 2]
        prev_row = result_df.iloc[idx - 1]

        if pp_row["exec_sell_signal"] != 1 or prev_row["exec_sell_signal"] != 0:
            return None

        return str(pp_row["trade_date"])

    # ------------------------------------------------------------------
    # 候选股票筛选
    # ------------------------------------------------------------------

    def _get_sell_candidates(self, trade_date: str) -> list[tuple[str, str, float]]:
        """找出满足卖出条件的持仓。返回 [(ts_code, signal_date, open_price)]"""
        candidates: list[tuple[str, str, float]] = []
        for ts_code in list(self.account.positions):
            signal_date = self._check_sell_signal(ts_code, trade_date)
            if signal_date:
                price = self._get_open_price(ts_code, trade_date)
                if price is not None and price > 0:
                    candidates.append((ts_code, signal_date, price))
        return candidates

    def _get_buy_candidates(self, trade_date: str) -> list[tuple[str, str, float]]:
        """找出满足买入条件且未持仓的股票。返回 [(ts_code, signal_date, open_price)]"""
        held = set(self.account.positions)
        candidates: list[tuple[str, str, float]] = []

        for ts_code in self.indicator_data:
            if ts_code in held:
                continue
            signal_date = self._check_buy_signal(ts_code, trade_date)
            if signal_date:
                price = self._get_open_price(ts_code, trade_date)
                if price is not None and price > 0:
                    candidates.append((ts_code, signal_date, price))

        return candidates

    # ------------------------------------------------------------------
    # 交易执行
    # ------------------------------------------------------------------

    def _execute_sells(self, trade_date: str) -> int:
        """执行当日所有卖出，返回成功笔数"""
        candidates = self._get_sell_candidates(trade_date)
        for ts_code, signal_date, price in candidates:
            result = self.account.sell(ts_code, price, signal_date, trade_date)
            if result:
                current_prices = self._get_all_current_prices(trade_date)
                total_value = self.account.get_total_value(current_prices)
                self.account.trades.append({
                    "日期": trade_date,
                    "股票代码": ts_code,
                    "名称": result["name"],
                    "操作": "卖出",
                    "信号日期": signal_date,
                    "价格": round(price, 2),
                    "股数": result["shares"],
                    "成交金额": round(result["amount"], 2),
                    "手续费": round(result["total_fee"], 2),
                    "账户余额": round(self.account.available_cash, 2),
                    "总资产": round(total_value, 2),
                })
        return len(candidates)

    def _execute_buys(self, trade_date: str) -> int:
        """资金平分买入，不够时取前 N 只。返回成功笔数"""
        candidates = self._get_buy_candidates(trade_date)
        if not candidates:
            return 0

        available = self.account.available_cash
        n = len(candidates)

        # 过滤：budget 不够 1 手的候选
        affordable: list[tuple[str, str, float]] = []
        for ts_code, signal_date, price in candidates:
            one_lot_cost = price * 100 * (1 + self.account.commission_rate)
            if available / n >= one_lot_cost:
                affordable.append((ts_code, signal_date, price))

        if not affordable:
            return 0

        # 重新平分资金
        budget_per_stock = available / len(affordable)

        bought = 0
        for ts_code, signal_date, price in affordable:
            shares = int(math.floor(
                budget_per_stock / (price * (1 + self.account.commission_rate)) / 100
            ) * 100)
            if shares < 100:
                continue

            name = self.stock_names.get(ts_code, ts_code)
            success = self.account.buy(ts_code, name, price, shares, signal_date, trade_date)
            if success:
                bought += 1
                amount = price * shares
                commission = amount * self.account.commission_rate
                current_prices = self._get_all_current_prices(trade_date)
                total_value = self.account.get_total_value(current_prices)
                self.account.trades.append({
                    "日期": trade_date,
                    "股票代码": ts_code,
                    "名称": name,
                    "操作": "买入",
                    "信号日期": signal_date,
                    "价格": round(price, 2),
                    "股数": shares,
                    "成交金额": round(amount, 2),
                    "手续费": round(commission, 2),
                    "账户余额": round(self.account.available_cash, 2),
                    "总资产": round(total_value, 2),
                })

        return bought

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, start_date: str, end_date: str) -> None:
        """按交易日驱动模拟交易"""
        trading_dates = self._get_trading_dates(start_date, end_date)
        if not trading_dates:
            print("没有可用的交易日")
            return

        print(f"\n开始模拟交易: {trading_dates[0]} ~ {trading_dates[-1]}, "
              f"共 {len(trading_dates)} 个交易日")
        print(f"初始资金: {self.account.initial_capital:,.2f}\n")

        for trade_date in trading_dates:
            # 先卖后买
            sell_count = self._execute_sells(trade_date)
            buy_count = self._execute_buys(trade_date)

            if sell_count > 0 or buy_count > 0:
                current_prices = self._get_all_current_prices(trade_date)
                total_value = self.account.get_total_value(current_prices)
                print(f"  {trade_date}: 卖出 {sell_count} 只, 买入 {buy_count} 只, "
                      f"持仓 {len(self.account.positions)} 只, "
                      f"余额 {self.account.available_cash:,.2f}, "
                      f"总资产 {total_value:,.2f}")

        print("\n模拟交易完成")

    # ------------------------------------------------------------------
    # CSV 输出
    # ------------------------------------------------------------------

    def save_to_csv(self, end_date: str) -> None:
        """保存交易流水、持仓快照、绩效汇总到 CSV"""
        output_dir = Path(__file__).resolve().parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1) 交易流水
        if self.account.trades:
            trades_df = pd.DataFrame(self.account.trades)
            trades_path = output_dir / f"paper_trading_trades_{end_date}.csv"
            trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
            print(f"交易流水已保存到: {trades_path}")
        else:
            print("无交易记录")

        # 2) 持仓快照
        if self.account.positions:
            rows = []
            for ts_code, pos in self.account.positions.items():
                latest_price = self._get_latest_close(ts_code, end_date)
                market_value = (pos["shares"] * latest_price) if latest_price else pos["cost"]
                floating_pnl = market_value - pos["cost"] if latest_price else 0
                rows.append({
                    "股票代码": ts_code,
                    "名称": pos["name"],
                    "买入日期": pos["buy_date"],
                    "买入价": round(pos["buy_price"], 2),
                    "现价": round(latest_price, 2) if latest_price else "N/A",
                    "持仓股数": pos["shares"],
                    "成本": round(pos["cost"], 2),
                    "市值": round(market_value, 2),
                    "浮动盈亏": round(floating_pnl, 2),
                })
            positions_df = pd.DataFrame(rows)
            positions_path = output_dir / f"paper_trading_positions_{end_date}.csv"
            positions_df.to_csv(positions_path, index=False, encoding="utf-8-sig")
            print(f"持仓快照已保存到: {positions_path}")
        else:
            print("当前无持仓")

        # 3) 绩效汇总
        current_prices: dict[str, float] = {}
        for ts_code in self.account.positions:
            p = self._get_latest_close(ts_code, end_date)
            if p:
                current_prices[ts_code] = p
        total_value = self.account.get_total_value(current_prices)
        total_return = total_value / self.account.initial_capital - 1

        completed = self.account.completed_trades
        win_count = sum(1 for t in completed if t["profit"] > 0)
        win_rate = win_count / len(completed) if completed else 0
        max_loss = min((t["profit"] for t in completed), default=0)

        summary_df = pd.DataFrame([{
            "初始资金": self.account.initial_capital,
            "当前总资产": round(total_value, 2),
            "总收益率": f"{total_return:.2%}",
            "完成交易笔数": len(completed),
            "胜率": f"{win_rate:.2%}",
            "最大单笔亏损": round(max_loss, 2),
        }])
        summary_path = output_dir / f"paper_trading_summary_{end_date}.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"绩效汇总已保存到: {summary_path}")


# ======================================================================
# CLI 入口
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模拟交易程序：基于 MXS 指标体系的多股票资金统筹买卖")
    parser.add_argument("--start-date", default=None,
                        help="模拟开始日期 YYYYMMDD，覆盖 .env 中的 PAPER_START_DATE")
    parser.add_argument("--end-date", default=None,
                        help="模拟结束日期 YYYYMMDD，默认到最近交易日")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 从 .env 读取配置，命令行参数可覆盖 start_date
    initial_capital = float(os.getenv("PAPER_INITIAL_CAPITAL", "1000000"))
    start_date = args.start_date or os.getenv("PAPER_START_DATE", "20251001")
    end_date = args.end_date or get_default_end_date()
    trade_mode = int(os.getenv("PAPER_TRADE_MODE", "2"))
    use_obv_filter = os.getenv("PAPER_USE_OBV_FILTER", "true").lower() == "true"
    commission_rate = float(os.getenv("PAPER_COMMISSION_RATE", "0.0003"))
    stamp_duty = float(os.getenv("PAPER_STAMP_DUTY", "0.001"))

    print("=" * 60)
    print("模拟交易程序")
    print("=" * 60)
    print(f"初始资金: {initial_capital:,.2f}")
    print(f"交易期间: {start_date} ~ {end_date}")
    print(f"交易模式: mode={trade_mode}")
    print(f"OBV 过滤: {'启用' if use_obv_filter else '禁用'}")
    print(f"佣金率: {commission_rate:.4f}")
    print(f"印花税率: {stamp_duty:.4f}\n")

    pro = init_tushare()
    if pro is None:
        raise RuntimeError("未能初始化 Tushare，请检查 TUSHARE_TOKEN 环境变量")

    account = PaperAccount(initial_capital, commission_rate, stamp_duty)
    trader = PaperTrader(pro, account, trade_mode, use_obv_filter)

    # 加载主板股票
    universe = trader.load_universe()
    if not universe:
        print("未获取到主板股票列表")
        return
    print(f"加载主板股票: {len(universe)} 支\n")

    # 预计算指标 + 构建价格索引
    trader.precalculate_indicators(universe, start_date, end_date)

    # 运行模拟
    trader.run(start_date, end_date)

    # 保存结果
    trader.save_to_csv(end_date)


if __name__ == "__main__":
    main()
