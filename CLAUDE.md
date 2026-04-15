# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python quantitative trading system that translates the TongHuaShun (同花顺) "抄底_梅小苏" (bottom hunting) technical indicator into standalone Python, enabling historical backtesting and automated market-wide signal screening for Chinese A-shares.

All code comments, CLI output, and documentation are in Chinese.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Calculate indicators for a single stock
python mxs_indicator.py --code 300760
python mxs_indicator.py --code 300760 --start-date 20230101 --end-date 20231231 --output-csv output.csv

# Run backtest on a stock
python backtest.py --code 300760 --start-date 20220101 --end-date 20240101

# Screen for buy/sell signals across the market
python stock_screener.py --market 创业板 --scan-type buy
python stock_screener.py --codes 300760,000001,600519 --scan-type both
```

## Environment Setup

Requires a Tushare Pro API token. Copy `.env.example` to `.env` and set `TUSHARE_TOKEN`. Python 3.10+ recommended.

## Architecture

### Data Flow

1. **Data Acquisition** (`util.py` + `backtest.py`): Fetches daily OHLCV, index, and float-share data via Tushare API with local file caching under `data/cache/`
2. **Indicator Calculation** (`backtest.py`): Translates TongHuaShun formula functions (MA, EMA, LLV, HHV, SMA_TDX, AVEDEV, zig_zag) into pandas/numpy vectorized operations
3. **Signal Generation** (`backtest.py`): Produces buy/sell signals from crossover detection and pattern matching
4. **Output**: Backtest trade records & stats, screening CSVs, individual stock indicator exports

### Key Modules

- **`util.py`** — Tushare API initialization, date helpers, retry logic with exponential backoff
- **`backtest.py`** — Core engine: all indicator calculations, data fetching/caching, backtesting loop, signal generation. This is the largest and most important file
- **`mxs_indicator.py`** — CLI wrapper for indicator calculation and CSV export on individual stocks
- **`stock_screener.py`** — CLI for market-wide signal scanning across boards (主板/创业板/科创板/北交所)

### Indicator Calculations

The TongHuaShun formula functions are reimplemented in `backtest.py`. Key signal types:
- **主力建仓** (build position): Large position accumulation detection
- **买入/卖出** (buy/sell): Breakout, bottom, and peak detection signals
- **散户线** (retail line) and **主力线** (main force line): Relative strength indicators

### Data Caching

Market data is cached as CSV files under `data/cache/` keyed by date and security code to minimize Tushare API calls. Cached data is reused across backtests and screening runs.

### Signal Output Files

Scan results are saved under `data/recent_buy_without_sell_*.csv` and `data/ths_indicator_*.csv`.

## Reference Files

- `mxs_formula.txt` — Original TongHuaShun indicator formula source
- `ths_api_manual.txt` — TongHuaShun formula function reference manual

## Stock Code Conventions

- 6-digit codes accepted (e.g., `300760`) — the system auto-appends the market suffix (`.SZ`/`.SH`/`.BJ`)
- Supports: Main Board (主板), ChiNext (创业板, 300xxx), STAR Market (科创板, 688xxx), CDR, Beijing Stock Exchange (北交所, 8xxxxx/4xxxxx)
- ETFs and index codes also supported
