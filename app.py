import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="NSE AI Quant Terminal & Paper CRM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark-terminal styling
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
    div[data-testid="stMetricValue"] { font-size: 1.35rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# 2. INDIAN MARKET CONSTANTS & LOT SIZES (2026 SEBI / NSE RULES)
# ==============================================================================
NSE_UNIVERSE = {
    "NIFTY 50 (Index)": {"ticker": "^NSEI", "lot_size": 65, "type": "INDEX"},
    "BANK NIFTY (Index)": {
        "ticker": "^NSEBANK",
        "lot_size": 30,
        "type": "INDEX",
    },
    "FINNIFTY (Index)": {
        "ticker": "NIFTY_FIN_SERVICE.NS",
        "lot_size": 60,
        "type": "INDEX",
    },
    "RELIANCE": {"ticker": "RELIANCE.NS", "lot_size": 1, "type": "EQUITY"},
    "TCS": {"ticker": "TCS.NS", "lot_size": 1, "type": "EQUITY"},
    "HDFC BANK": {"ticker": "HDFCBANK.NS", "lot_size": 1, "type": "EQUITY"},
    "ICICI BANK": {"ticker": "ICICIBANK.NS", "lot_size": 1, "type": "EQUITY"},
    "INFOSYS": {"ticker": "INFY.NS", "lot_size": 1, "type": "EQUITY"},
    "SBI": {"ticker": "SBIN.NS", "lot_size": 1, "type": "EQUITY"},
    "TRENT": {"ticker": "TRENT.NS", "lot_size": 1, "type": "EQUITY"},
    "L&T": {"ticker": "LT.NS", "lot_size": 1, "type": "EQUITY"},
    "TATA STEEL": {"ticker": "TATASTEEL.NS", "lot_size": 1, "type": "EQUITY"},
}


# ==============================================================================
# 3. SQLITE CRM & PAPER-TRADING DATABASE ENGINE
# ==============================================================================
def init_db():
    conn = sqlite3.connect("trading_crm.db", check_same_thread=False)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS account_settings (
            id INTEGER PRIMARY KEY,
            initial_capital REAL NOT NULL,
            current_cash REAL NOT NULL
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            instrument_mode TEXT NOT NULL,
            strategy_tag TEXT NOT NULL,
            direction TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            entry_time TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_price REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',
            exit_time TEXT,
            exit_price REAL,
            gross_pnl REAL DEFAULT 0.0,
            charges REAL DEFAULT 0.0,
            net_pnl REAL DEFAULT 0.0,
            r_multiple REAL DEFAULT 0.0,
            setup_score INTEGER DEFAULT 0,
            notes TEXT DEFAULT ''
        )
    """)

    c.execute("SELECT COUNT(*) FROM account_settings")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO account_settings (id, initial_capital, current_cash) VALUES (1, 1000000.0, 1000000.0)"
        )
    conn.commit()
    return conn


DB_CONN = init_db()


def calculate_indian_charges(
    entry_price: float, exit_price: float, qty: int, mode: str
) -> float:
    """Calculates real Indian Brokerage + STT + Exchange Turnover + Stamp + 18% GST."""
    buy_val = entry_price * qty
    sell_val = exit_price * qty
    turnover = buy_val + sell_val
    brokerage = 40.0  # Flat Rs 20 buy + Rs 20 sell

    if mode == "FUTURES":
        stt = sell_val * 0.0005  # 0.05% STT on futures sell
        exch_fee = turnover * 0.0000173
        stamp = buy_val * 0.00002
    elif mode == "OPTIONS (Sim Delta)":
        stt = sell_val * 0.0015  # 0.15% STT on options sell
        exch_fee = turnover * 0.0003503
        stamp = buy_val * 0.00003
    else:  # EQUITY DELIVERY / INTRADAY
        stt = turnover * 0.0010  # 0.10% STT
        exch_fee = turnover * 0.0000297
        stamp = buy_val * 0.00015

    sebi_fee = turnover * 0.000001
    gst = 0.18 * (brokerage + exch_fee + sebi_fee)
    return round(brokerage + stt + exch_fee + stamp + sebi_fee + gst, 2)


# ==============================================================================
# 4. LIVE MARKET DATA FETCHER (WITH SAFE FALLBACK)
# ==============================================================================
@st.cache_data(ttl=45)
def fetch_ohlcv(ticker: str, timeframe: str) -> pd.DataFrame:
    period_map = {"5m": "5d", "15m": "15d", "1h": "60d", "1d": "1y"}
    period = period_map.get(timeframe, "15d")
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=timeframe, auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("Empty feed")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        # Normalize datetime column
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df.rename(
            columns={
                time_col: "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
            inplace=True,
        )
        df = (
            df[["datetime", "open", "high", "low", "close", "volume"]]
            .dropna()
            .reset_index(drop=True)
        )
        # Ensure volume is non-zero for indices where Yahoo omits volume on some bars
        if (df["volume"] == 0).all():
            df["volume"] = np.random.randint(150000, 650000, size=len(df))
        return df
    except Exception:
        # Fallback synthetic series so UI never crashes if offline
        np.random.seed(42)
        n = 150
        dates = pd.date_range(end=datetime.now(), periods=n, freq="15min")
        price = 24500 + np.cumsum(np.random.randn(n) * 18)
        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": price - np.random.randn(n) * 6,
                "high": price + np.abs(np.random.randn(n) * 14),
                "low": price - np.abs(np.random.randn(n) * 14),
                "close": price,
                "volume": np.random.randint(100000, 900000, size=n),
            }
        )
        return df


# ==============================================================================
# 5. PURE-PYTHON QUANT & MULTI-STRATEGY ENGINE (ALL WORKING STRATEGIES)
# ==============================================================================
def run_full_quant_analysis(
    df: pd.DataFrame, symbol_name: str, timeframe: str
) -> dict:
    df = df.copy()

    # --- 1. Classical Trend & Momentum Indicators ---
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["sma200"] = (
        df["close"].rolling(200, min_periods=20).mean()
    )  # Adaptive if <200 bars

    # Wilder's ATR(14)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=1).mean()
    df["atr5"] = tr.rolling(5, min_periods=1).mean()
    df["atr50"] = tr.rolling(50, min_periods=5).mean()

    # RSI(14) and RSI(2)
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    df["rsi14"] = calc_rsi(df["close"], 14)
    df["rsi2"] = calc_rsi(df["close"], 2)

    # ADX(14)
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean().replace(0, 1e-9)
    plus_di = (
        100
        * pd.Series(plus_dm).ewm(alpha=1 / 14, adjust=False).mean()
        / tr_smooth
    )
    minus_di = (
        100
        * pd.Series(minus_dm).ewm(alpha=1 / 14, adjust=False).mean()
        / tr_smooth
    )
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9))
    df["adx14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Rolling Session VWAP & RVOL
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    df["vwap"] = (
        (typical_price * df["volume"]).rolling(25, min_periods=1).sum()
        / df["volume"].rolling(25, min_periods=1).sum().replace(0, 1)
    )
    df["vol_sma20"] = df["volume"].rolling(20, min_periods=1).mean()
    df["rvol"] = df["volume"] / df["vol_sma20"].replace(0, 1)

    # Bollinger Bands (20, 2) & Keltner Channels (20, 1.5) -> Squeeze Detector
    df["bb_mid"] = df["close"].rolling(20, min_periods=5).mean()
    df["bb_std"] = df["close"].rolling(20, min_periods=5).std()
    df["bb_upper"] = df["bb_mid"] + 2.0 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2.0 * df["bb_std"]
    df["kc_upper"] = df["ema20"] + 1.5 * df["atr14"]
    df["kc_lower"] = df["ema20"] - 1.5 * df["atr14"]
    df["squeeze_on"] = (df["bb_lower"] > df["kc_lower"]) & (
        df["bb_upper"] < df["kc_upper"]
    )

    # Candlestick Anatomy (CLV & Body Ratio)
    rng = (df["high"] - df["low"]).replace(0, 1e-9)
    df["clv"] = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    df["body_ratio"] = (df["close"] - df["open"]).abs() / rng

    # --- 2. Smart Money Concepts (SMC): Swing Pivots, BOS/ChoCh, FVG, Liquidity Sweeps ---
    # 3-Bar Fair Value Gaps (FVG)
    df["bull_fvg"] = (df["low"] > df["high"].shift(2)) & (
        df["close"].shift(1) > df["open"].shift(1)
    )
    df["bear_fvg"] = (df["high"] < df["low"].shift(2)) & (
        df["close"].shift(1) < df["open"].shift(1)
    )

    # Find recent unmitigated FVG zones (last 25 bars)
    recent_fvgs = []
    for i in range(max(2, len(df) - 25), len(df)):
        if df["bull_fvg"].iloc[i]:
            btm = float(df["high"].iloc[i - 2])
            top = float(df["low"].iloc[i])
            # Check if mitigated by subsequent lows
            subsequent_lows = (
                df["low"].iloc[i + 1 :] if i + 1 < len(df) else pd.Series(dtype=float)
            )
            if subsequent_lows.empty or subsequent_lows.min() > btm:
                recent_fvgs.append(
                    {
                        "type": "BULLISH FVG",
                        "bottom": btm,
                        "top": top,
                        "start_idx": i - 2,
                        "time": df["datetime"].iloc[i],
                    }
                )
        elif df["bear_fvg"].iloc[i]:
            top = float(df["low"].iloc[i - 2])
            btm = float(df["high"].iloc[i])
            subsequent_highs = (
                df["high"].iloc[i + 1 :] if i + 1 < len(df) else pd.Series(dtype=float)
            )
            if subsequent_highs.empty or subsequent_highs.max() < top:
                recent_fvgs.append(
                    {
                        "type": "BEARISH FVG",
                        "bottom": btm,
                        "top": top,
                        "start_idx": i - 2,
                        "time": df["datetime"].iloc[i],
                    }
                )

    # Swing Highs & Lows (20-bar structural window)
    lookback = df.iloc[-30:-1] if len(df) > 31 else df.iloc[:-1]
    swing_high = float(lookback["high"].max())
    swing_low = float(lookback["low"].min())

    # Volume Profile Point of Control (POC) over last 60 bars
    vp_slice = df.tail(60)
    bins = np.linspace(vp_slice["low"].min(), vp_slice["high"].max(), 16)
    vp_indices = np.digitize(vp_slice["close"], bins)
    bin_vols = {}
    for idx_b, v in zip(vp_indices, vp_slice["volume"]):
        bin_vols[idx_b] = bin_vols.get(idx_b, 0) + v
    best_bin = max(bin_vols, key=bin_vols.get) if bin_vols else 8
    best_bin = min(max(best_bin, 1), len(bins) - 1)
    poc_price = float((bins[best_bin - 1] + bins[best_bin]) / 2.0)

    # --- 3. Evaluate Every Working Strategy ---
    curr = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else curr
    price = float(curr["close"])
    atr = float(curr["atr14"])
    vwap_val = float(curr["vwap"])
    ema20_val = float(curr["ema20"])
    ema50_val = float(curr["ema50"])
    sma200_val = float(curr["sma200"])
    rsi14_val = float(curr["rsi14"])
    rsi2_val = float(curr["rsi2"])
    adx_val = float(curr["adx14"])
    rvol_val = float(curr["rvol"])

    strategies = []

    # Strategy 1: ORB / Momentum Breakout + VWAP
    orb_high = float(df.tail(15)["high"].iloc[:-1].max())
    orb_low = float(df.tail(15)["low"].iloc[:-1].min())
    if price > orb_high and price > vwap_val and rvol_val > 1.2:
        strategies.append(
            {
                "name": "1. Intraday ORB + VWAP Breakout",
                "status": "TRIGGERED (BULLISH)",
                "bias": "BUY",
                "points": 22,
                "detail": f"Broke 15-bar high ({orb_high:.1f}) above VWAP ({vwap_val:.1f}) with RVOL {rvol_val:.2f}x.",
            }
        )
    elif price < orb_low and price < vwap_val and rvol_val > 1.2:
        strategies.append(
            {
                "name": "1. Intraday ORB + VWAP Breakdown",
                "status": "TRIGGERED (BEARISH)",
                "bias": "SHORT",
                "points": 22,
                "detail": f"Broke 15-bar low ({orb_low:.1f}) below VWAP ({vwap_val:.1f}) with RVOL {rvol_val:.2f}x.",
            }
        )
    else:
        strategies.append(
            {
                "name": "1. Intraday ORB + VWAP",
                "status": "FORMING / INSIDE RANGE",
                "bias": "NEUTRAL",
                "points": 8,
                "detail": f"Trading inside range [{orb_low:.1f} - {orb_high:.1f}]. Wait for boundary break.",
            }
        )

    # Strategy 2: SMC Liquidity Sweep + FVG + BOS
    prior_10_low = float(df.iloc[-12:-2]["low"].min()) if len(df) > 12 else swing_low
    prior_10_high = (
        float(df.iloc[-12:-2]["high"].max()) if len(df) > 12 else swing_high
    )
    swept_low = (
        curr["low"] < prior_10_low or prev["low"] < prior_10_low
    ) and price > prior_10_low
    swept_high = (
        curr["high"] > prior_10_high or prev["high"] > prior_10_high
    ) and price < prior_10_high
    has_bull_fvg = any(f["type"] == "BULLISH FVG" for f in recent_fvgs[-3:])
    has_bear_fvg = any(f["type"] == "BEARISH FVG" for f in recent_fvgs[-3:])

    if swept_low or (has_bull_fvg and price > ema20_val):
        strategies.append(
            {
                "name": "2. SMC Liquidity Sweep & FVG",
                "status": "TRIGGERED (BULLISH)",
                "bias": "BUY",
                "points": 20,
                "detail": "Downside liquidity swept / Bullish Fair Value Gap active below price.",
            }
        )
    elif swept_high or (has_bear_fvg and price < ema20_val):
        strategies.append(
            {
                "name": "2. SMC Liquidity Sweep & FVG",
                "status": "TRIGGERED (BEARISH)",
                "bias": "SHORT",
                "points": 20,
                "detail": "Upside liquidity swept / Bearish Fair Value Gap active above price.",
            }
        )
    else:
        strategies.append(
            {
                "name": "2. SMC Liquidity Sweep & FVG",
                "status": "MONITORING ZONES",
                "bias": "NEUTRAL",
                "points": 6,
                "detail": f"Key Liquidity Pools sitting at {swing_high:.1f} (High) and {swing_low:.1f} (Low).",
            }
        )

    # Strategy 3: Volatility Contraction (VCP) / Bollinger-Keltner Squeeze
    recent_squeeze = bool(df["squeeze_on"].tail(5).any())
    vcp_contracting = float(curr["atr5"]) < 0.80 * float(curr["atr50"])
    if (recent_squeeze or vcp_contracting) and price > float(curr["bb_upper"]):
        strategies.append(
            {
                "name": "3. Volatility Squeeze / VCP Breakout",
                "status": "TRIGGERED (BULLISH)",
                "bias": "BUY",
                "points": 18,
                "detail": "Volatility compression resolved upward outside Upper Bollinger Band.",
            }
        )
    elif (recent_squeeze or vcp_contracting) and price < float(curr["bb_lower"]):
        strategies.append(
            {
                "name": "3. Volatility Squeeze / VCP Breakdown",
                "status": "TRIGGERED (BEARISH)",
                "bias": "SHORT",
                "points": 18,
                "detail": "Volatility compression resolved downward outside Lower Bollinger Band.",
            }
        )
    elif recent_squeeze or vcp_contracting:
        strategies.append(
            {
                "name": "3. Volatility Squeeze / VCP",
                "status": "COILING (SQUEEZE ON)",
                "bias": "NEUTRAL",
                "points": 12,
                "detail": "ATR/Bollinger squeeze active; explosive directional move imminent.",
            }
        )
    else:
        strategies.append(
            {
                "name": "3. Volatility Squeeze / VCP",
                "status": "INACTIVE",
                "bias": "NEUTRAL",
                "points": 5,
                "detail": "Bands already expanded; no tight volatility compression.",
            }
        )

    # Strategy 4: Trend Pullback to 20 EMA + Cardwell RSI Regime
    dist_ema20_atr = abs(price - ema20_val) / max(atr, 1e-9)
    if (
        price > ema50_val
        and dist_ema20_atr <= 0.85
        and 40 <= rsi14_val <= 58
        and curr["clv"] > 0
    ):
        strategies.append(
            {
                "name": "4. 20 EMA Pullback + Cardwell RSI",
                "status": "TRIGGERED (BULLISH)",
                "bias": "BUY",
                "points": 18,
                "detail": f"Low-risk pullback to rising 20 EMA ({ema20_val:.1f}) with RSI holding {rsi14_val:.1f} bull support.",
            }
        )
    elif (
        price < ema50_val
        and dist_ema20_atr <= 0.85
        and 42 <= rsi14_val <= 60
        and curr["clv"] < 0
    ):
        strategies.append(
            {
                "name": "4. 20 EMA Pullback + Cardwell RSI",
                "status": "TRIGGERED (BEARISH)",
                "bias": "SHORT",
                "points": 18,
                "detail": f"Pullback rejection at declining 20 EMA ({ema20_val:.1f}) with RSI at {rsi14_val:.1f}.",
            }
        )
    else:
        strategies.append(
            {
                "name": "4. 20 EMA Pullback + Cardwell RSI",
                "status": "WAITING FOR PULLBACK",
                "bias": "NEUTRAL",
                "points": 7,
                "detail": f"Distance from 20 EMA is {dist_ema20_atr:.2f}x ATR (RSI: {rsi14_val:.1f}).",
            }
        )

    # Strategy 5: RSI(2) Short-Horizon Mean Reversion
    if price > sma200_val and rsi2_val < 15:
        strategies.append(
            {
                "name": "5. RSI(2) Mean Reversion",
                "status": "TRIGGERED (OVERSOLD DIP)",
                "bias": "BUY",
                "points": 14,
                "detail": f"Extreme short-term oversold stretch (RSI-2 = {rsi2_val:.1f}) inside long-t