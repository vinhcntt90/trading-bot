# BTC Advanced Charting & Trading Analysis

**A professional-grade framework for Bitcoin technical analysis.**

This project consolidates advanced trading strategies into a modular, easy-to-run Python application. It integrates concepts like Smart Money Concepts (SMC), Gann Fan, Lunar Cycles, and Derivatives Data to provide a comprehensive market bias.

---

## 🚀 Key Features

### 📊 Advanced Technical Analysis
*   **Smart Money Concepts (SMC)**: Automatically detects Order Blocks, Fair Value Gaps (FVG), Liquidity Pools (BSL/SSL), and Wyckoff Phases.
*   **Gann Fan**: Trends and support/resistance based on geometric angles.
*   **Astro-Trading**: Lunar phase sentiment analysis and Mercury Retrograde alerts.
*   **Classic Indicators**: RSI, MACD, Bollinger Bands, Moving Averages (EMA 9/21/50/200), and Volume Profile (POC/VA).

### 🧠 Intelligent Strategy Engine
*   **Multi-Timeframe Consensus**: Analyzes 15m, 1H, 4H, and 1D charts to determine an overall market bias.
*   **Confluence Detection**: Identifies high-probability entry zones where multiple indicators overlap (e.g., FVG + Pivot + Fib).
*   **Golden Pocket Strategy**: Specialized setup for Fibonacci retracements (0.618-0.65).
*   **Automated Trading Plan**: Generates concrete Long/Short setups with Entry, Stop Loss, and Take Profit levels.

### 📈 Visualization & Alerts
*   **Interactive Charts**: Generates high-resolution Plotly charts saved as images for easy sharing.
*   **Telegram Integration**: Sends detailed analysis reports and charts directly to your Telegram channel/group.
*   **Console Dashboard**: Clean, summarized output in your terminal.

---

## 🛠️ Installation

1.  **Clone the Project**:
    ```bash
    git clone https://github.com/yourusername/btc_advanced_charting.git
    cd btc_advanced_charting
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration**:
    Create a `.env` file in the root directory with your settings:
    ```ini
    # Telegram Configuration (Required for notifications)
    TELEGRAM_BOT_TOKEN=your_bot_token_here
    TELEGRAM_CHAT_IDS=-100xxxxxxxx,-100xxxxxxxx

    # System Configuration (Optional)
    # ARTIFACTS_DIR=./output
    ```

---

## 🖥️ Usage

### 1. Run Full Analysis
To perform the analysis, generate charts, and send Telegram notifications:

```bash
python run.py
```

### 2. Auto Runner (Windows)
To run the analysis automatically every 15 minutes (at :00, :15, :30, :45):
Double-click `run_auto.bat` in the project folder.

### 3. Test Mode (No Notifications)
To run everything locally without sending messages to Telegram (useful for testing/debugging):

```bash
python run.py --test
```

---

## 📂 Project Structure

```
btc_advanced_charting/
├── .env                 # Configuration & Secrets
├── requirements.txt     # Dependencies
├── run.py               # Main Entry Point
├── output/              # Generated Artifacts (Models, Charts)
└── src/                 # Source Code Module
    ├── __init__.py
    ├── analysis.py      # Core Strategy & Logic (SMC, Gann, Plan)
    ├── config.py        # Configuration Loader
    ├── data.py          # Data Fetching (Binance API)
    ├── indicators.py    # Technical Indicators Library
    ├── main.py          # Orchestrator
    ├── notifications.py # Telegram Bot Integration
    ├── plotting.py      # Plotly Visualization
    └── reporting.py     # Terminal Output Formatting
```

## ⚠️ Disclaimer
*This software is for educational purposes only. Do not trade using money you cannot afford to lose. The authors are not responsible for any financial losses.*
