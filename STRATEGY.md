# 📘 Technical Analysis & Strategy Documentation

This document details the core logic, algorithms, and technical indicators used in the **BTC Advanced Charting** system, specifically focusing on the **Elliott Wave + Fibonacci Golden Pocket** strategy.

---

## 🌊 Elliott Wave + Fibonacci Strategy (The Core Engine)

This strategy is built upon a strict **3-Module Framework** designed to identify high-probability trade setups by combining Trend, Structure, and Momentum.

### Module 1: Data Infrastructure (Hạ tầng Dữ liệu)

The foundation of the analysis relying on clean, multi-timeframe data.

*   **Primary Timeframe:** `1H` (Hourly). This is the execution timeframe.
*   **Trend Timeframe:** `4H` (Four-Hour). Used for broader context (optional).
*   **Data Depth:** `500 candles`. This ensures sufficient data points for:
    *   EMA 200 calculation (requires pre-loading).
    *   Rolling window analysis (100 candles lookback).
*   **Preprocessing:** Handling missing data (NaN) and normalizing timestamps to local time.

---

### Module 2: Analysis Engine (Bộ não Phân tích)

This module processes raw data to extract structured market insights using a 3-step pipeline.

#### 1. Trend Filter (Bộ lọc Xu hướng)
We strictly trade **with the trend**. The trend is determined by the alignment of price relative to key Moving Averages.

*   **Bullish Trend (UPTREND):**
    *   Condition: `Price > EMA 50 > EMA 200`
    *   Action: Only look for **LONG** setups.
*   **Bearish Trend (DOWNTREND):**
    *   Condition: `Price < EMA 50 < EMA 200`
    *   Action: Only look for **SHORT** setups.
*   **Neutral/Sideway:**
    *   Condition: Any other arrangement (e.g., Price between EMAs, or EMAs crossed).
    *   Action: **No Trade** (Filter out).

#### 2. Wave Structure (Cấu trúc Sóng)
Once a trend is confirmed, we identify the **Impulse Wave** (Sóng Đẩy) to trade the pullback.

*   **Algorithm:** Sliding Window (Rolling 100).
*   **Key Points:** Identify `Highest High` and `Lowest Low` within the last 100 candles.
*   **Impulse Identification:**
    *   *Uptrend:* Impulse is the move from **Lowest Low** $\rightarrow$ **Highest High**.
    *   *Downtrend:* Impulse is the move from **Highest High** $\rightarrow$ **Lowest Low**.

#### 3. Fibonacci OTE (Vùng Tối ưu)
We calculate the Fibonacci Retracement levels for the identified Impulse Wave.

*   **Target Zone:** **Optimal Trade Entry (OTE)**.
*   **Range:** `0.618` to `0.786`.
*   **Trigger Condition:** The *current candle* (Low for Long, High for Short) must touch or penetrate this zone.

---

### Module 3: Signal Confirmation (Bộ lọc Tín hiệu)

A setup is valid only if confirmed by price action at the OTE zone.

#### 1. Pattern Recognition (Mô hình Nến)
We analyze the **Last Closed Candle** (`Close[1]`) when price is in the OTE zone.

*   **Pinbar (Nến Rút Chân):**
    *   *Bullish:* Lower wick $\ge$ 2/3 of total range. Indicates rejection of lower prices.
    *   *Bearish:* Upper wick $\ge$ 2/3 of total range. Indicates rejection of higher prices.
*   **Engulfing (Nến Nhấn Chìm):**
    *   *Bullish:* Green body completely covers previous Red body.
    *   *Bearish:* Red body completely covers previous Green body.

#### 2. Momentum Filter (RSI - Check)
An optional filter to ensure we are entering at extreme momentum levels (oversold/overbought).

*   **Long Setup:** `RSI(14) < 40` (or near Oversold) at the moment of zone test.
*   **Short Setup:** `RSI(14) > 60` (or near Overbought) at the moment of zone test.

---

## 🛠 Other Indicators

### Smart Money Concepts (SMC)
*   **Order Blocks (OB):** The last opposite candle before a strong move. Acts as support/resistance.
*   **Fair Value Gaps (FVG):** Imbalances in price delivery. Price often returns to fill these gaps.
*   **Liquidity (BSL/SSL):** Buy-Side and Sell-Side Liquidity pools (Swing Highs/Lows).

### Gann Fan
Uses geometric angles from major pivots to identify hidden support/resistance lines.
*   **Signal:** Trends are strong when price stays above the 1x1 angle.

### Lunar Trading
Correlates moon phases with potential market turning points.
*   **New Moon:** Often associated with accumulation/bottoms.
*   **Full Moon:** Often associated with distribution/tops.
