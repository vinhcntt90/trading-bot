import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Security / Credentials
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8439259405:AAGDBu8xbVVRv9_k8TGTJ3vpi-sPucYrdY4")
    TELEGRAM_CHAT_IDS = [
        chat_id.strip() 
        for chat_id in os.getenv("TELEGRAM_CHAT_IDS", "-1003863476083,-1003875263588").split(",") 
        if chat_id.strip()
    ]
    
    # Bật/Tắt gửi Telegram (True = gửi, False = không gửi)
    SEND_TO_TELEGRAM = True
    
    # AI / LLM Configuration
    LLM_ENABLED = True
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", os.path.join(os.path.expanduser('~'), 'Desktop'))
    
    # Charting
    DEFAULT_TIMEFRAME = "15m"
    LIMIT = 1000

    @classmethod
    def validate(cls):
        if not cls.TELEGRAM_BOT_TOKEN:
            print("[!] Warning: TELEGRAM_BOT_TOKEN not found in .env")
        if not cls.TELEGRAM_CHAT_IDS:
            print("[!] Warning: TELEGRAM_CHAT_IDS not found in .env")
        
        # Ensure output directory exists
        if not os.path.exists(cls.ARTIFACTS_DIR):
            os.makedirs(cls.ARTIFACTS_DIR)

# Auto-validate on import
Config.validate()


# ============================================
# Dark theme colors matching TradingView
# ============================================
COLORS = {
    'bg': '#0d1117',
    'chart_bg': '#0d1117',
    'text': '#c9d1d9',
    'grid': '#21262d',
    'candle_up': '#26a69a',
    'candle_down': '#ef5350',
    'ema9': '#2196f3',       # Blue
    'ema21': '#ffc107',      # Yellow/Gold
    'ema50': '#ff5722',      # Orange
    'ema200': '#e91e63',     # Pink
    'ema_cloud': '#00bcd433', # Cyan cloud
    'volume_up': '#26a69a88',
    'volume_down': '#ef535088',
    'support': '#26a69a',
    'resistance': '#ef5350',
    'trendline': '#00bcd4',   # Cyan
    'projection': '#ffc107',  # Yellow
    'bullish': '#00c853',
    'bearish': '#ff1744',
    'neutral': '#9e9e9e',
    'pivot': '#9c27b0',       # Purple for pivot
    'poc': '#ff9800',         # Orange for POC
    'gann': '#e040fb',        # Purple/Magenta for Gann
    'lunar': '#00e5ff',       # Cyan for Lunar
    'trend_high': '#ef5350',  # Red for Trend High
    'trend_mid': '#ffeb3b',   # Yellow for Trend Mid
    'trend_low': '#26a69a',   # Green for Trend Low
    'lro': '#2196f3',         # Blue for LRO
}


# ============================================
# CHARTPRIME INDICATORS CONFIGURATION
# ============================================
CHARTPRIME_CONFIG = {
    'trend_levels_length': 10,       # Trend Levels lookback
    'lro_length': 20,                # Linear Regression Oscillator lookback
    'lro_upper_threshold': 1.5,      # Overbought threshold
    'lro_lower_threshold': -1.5,     # Oversold threshold
    'timeframe': '15m',              # Only apply to 15m candles
}


# ============================================
# SUPPORT RESISTANCE CHANNELS [SRChannel]
# Based on PineScript by LonesomeTheBlue
# ============================================
SRCHANNEL_CONFIG = {
    'pivot_period': 10,          # Pivot Period (checks left & right bars)
    'channel_width_pct': 5,      # Maximum Channel Width % (of 300-bar range)
    'min_strength': 1,           # Minimum strength (at least 1 pivot per channel)
    'max_channels': 6,           # Maximum number of S/R channels to show
    'loopback': 100,             # Loopback period for pivot detection (reduced for data availability)
}


# ============================================
# LUXALGO SUPPORT & RESISTANCE DYNAMIC
# ============================================
LUXALGO_CONFIG = {
    'length': 8,           # Pivot detection length
    'mult': 50,            # ATR multiplier for zone width (50 = 0.5 ATR)
    'atr_length': 4,       # ATR period
}


# ============================================
# GANN FAN ANALYSIS
# ============================================
GANN_ANGLES = {
    '1x8': 82.5,    # 1 unit price per 8 units time
    '1x4': 75.0,
    '1x3': 71.25,
    '1x2': 63.75,
    '1x1': 45.0,    # Main Gann line - 45 degrees
    '2x1': 26.25,
    '3x1': 18.75,
    '4x1': 15.0,
    '8x1': 7.5,
}
