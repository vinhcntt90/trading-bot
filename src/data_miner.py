"""
Data Mining Module for Phase 2: Historical Data Collection & Labeling

This script:
1. Downloads historical BTC data from Binance (1-2 years)
2. Calculates all technical indicators
3. Runs Elliott Wave logic on historical data
4. Labels each signal as WIN/LOSS based on future price action
5. Exports dataset for ML training
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from binance.client import Client

# ============================================
# CONFIGURATION
# ============================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
SYMBOL = 'BTCUSDT'
TIMEFRAMES = ['1h']  # Focus on 1H for Elliott Wave
HISTORY_DAYS = 1460  # 4 years of data (Increased for robust AI)
LOOKBACK_CANDLES = 100  # For Elliott Wave detection
FUTURE_CANDLES = 20  # How many candles to check for Win/Loss

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


class HistoricalDataMiner:
    """Downloads and processes historical data for ML training."""
    
    def __init__(self):
        self.client = Client()
        print("[*] Historical Data Miner initialized")
    
    def fetch_historical_klines(self, symbol='BTCUSDT', interval='1h', days=365):
        """
        Fetch historical klines data from Binance.
        Binance has a 1000 candle limit per request, so we need to loop.
        """
        print(f"\n[*] Fetching {days} days of {symbol} {interval} data...")
        
        all_klines = []
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        current_start = start_time
        batch_size = 1000
        
        # Calculate approximate candle duration
        interval_minutes = {
            '15m': 15, '1h': 60, '4h': 240, '1d': 1440
        }
        candle_minutes = interval_minutes.get(interval, 60)
        candles_per_day = 1440 / candle_minutes
        total_candles_needed = int(days * candles_per_day)
        
        print(f"    Total candles needed: ~{total_candles_needed}")
        
        fetched = 0
        while fetched < total_candles_needed:
            try:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=batch_size,
                    startTime=int(current_start.timestamp() * 1000)
                )
                
                if not klines:
                    break
                    
                all_klines.extend(klines)
                fetched += len(klines)
                
                # Update start time for next batch
                last_timestamp = klines[-1][0]
                current_start = datetime.fromtimestamp(last_timestamp / 1000) + timedelta(minutes=candle_minutes)
                
                print(f"    Fetched: {fetched}/{total_candles_needed} candles", end='\r')
                
                # Rate limit
                time.sleep(0.1)
                
                if len(klines) < batch_size:
                    break
                    
            except Exception as e:
                print(f"\n    [!] Error fetching batch: {e}")
                time.sleep(1)
                continue
        
        print(f"\n    ✅ Total fetched: {len(all_klines)} candles")
        
        # Convert to DataFrame
        df = pd.DataFrame(all_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'trades',
            'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators for the dataset."""
        print("[*] Calculating indicators...")
        
        # EMAs
        df['EMA9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['EMA21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['EMA50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['EMA200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        df['BB_Std'] = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # Volume
        df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
        df['Vol_Ratio'] = df['volume'] / df['Vol_MA20']
        
        # EMA Slopes (Rate of Change)
        df['EMA50_Slope'] = df['EMA50'].pct_change(periods=5) * 100
        df['EMA200_Slope'] = df['EMA200'].pct_change(periods=10) * 100
        
        # Price relative to EMAs
        df['Price_vs_EMA50'] = (df['close'] - df['EMA50']) / df['EMA50'] * 100
        df['Price_vs_EMA200'] = (df['close'] - df['EMA200']) / df['EMA200'] * 100
        
        print("    ✅ Indicators calculated")
        return df
    
    def find_fractals(self, df, left=5, right=5):
        """Find Fractal Highs and Lows for impulse wave detection."""
        fractal_highs = pd.Series(index=df.index, dtype=float)
        fractal_lows = pd.Series(index=df.index, dtype=float)
        
        for i in range(left, len(df) - right):
            is_high = True
            is_low = True
            
            for j in range(1, left + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j]:
                    is_high = False
                if df['low'].iloc[i] >= df['low'].iloc[i - j]:
                    is_low = False
            
            for j in range(1, right + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_high = False
                if df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_low = False
            
            if is_high:
                fractal_highs.iloc[i] = df['high'].iloc[i]
            if is_low:
                fractal_lows.iloc[i] = df['low'].iloc[i]
        
        return fractal_highs, fractal_lows
    
    def detect_candlestick_pattern(self, df, idx):
        """Detect Pinbar/Engulfing at a specific index."""
        if idx < 2 or idx >= len(df):
            return None, None, 0
        
        curr = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        curr_body = abs(curr['close'] - curr['open'])
        curr_range = curr['high'] - curr['low']
        curr_upper_wick = curr['high'] - max(curr['open'], curr['close'])
        curr_lower_wick = min(curr['open'], curr['close']) - curr['low']
        
        prev_body = abs(prev['close'] - prev['open'])
        
        if curr_range == 0:
            return None, None, 0
        
        # Bullish Pinbar
        if curr_lower_wick >= 2 * curr_body and curr_upper_wick < curr_body:
            return 'PINBAR', 'BULLISH', 70
        
        # Bearish Pinbar
        if curr_upper_wick >= 2 * curr_body and curr_lower_wick < curr_body:
            return 'PINBAR', 'BEARISH', 70
        
        # Bullish Engulfing
        if prev['close'] < prev['open'] and curr['close'] > curr['open']:
            if curr['close'] > prev['open'] and curr['open'] < prev['close']:
                return 'ENGULFING', 'BULLISH', 70
        
        # Bearish Engulfing
        if prev['close'] > prev['open'] and curr['close'] < curr['open']:
            if curr['open'] > prev['close'] and curr['close'] < prev['open']:
                return 'ENGULFING', 'BEARISH', 70
        
        return None, None, 0
    
    def run_elliott_wave_scan(self, df, lookback=100):
        """
        Scan historical data for Elliott Wave setups.
        Returns list of potential trade signals with their outcomes.
        """
        print(f"\n[*] Scanning for Elliott Wave setups (lookback={lookback})...")
        
        signals = []
        
        # Start from position where we have enough history
        start_idx = max(200, lookback)  # Need EMA200 + lookback
        end_idx = len(df) - FUTURE_CANDLES  # Leave room to check outcome
        
        for i in range(start_idx, end_idx):
            # Current state at position i
            current_price = df['close'].iloc[i]
            ema50 = df['EMA50'].iloc[i]
            ema200 = df['EMA200'].iloc[i]
            rsi = df['RSI'].iloc[i]
            
            # Skip if NaN
            if pd.isna(ema50) or pd.isna(ema200) or pd.isna(rsi):
                continue
            
            # === MODULE 1: TREND FILTER ===
            if current_price > ema50 > ema200:
                ema_trend = 'UPTREND'
            elif current_price < ema50 < ema200:
                ema_trend = 'DOWNTREND'
            else:
                continue  # Skip sideway
            
            # === MODULE 2: WAVE STRUCTURE ===
            window_df = df.iloc[i - lookback:i]
            highest_idx = window_df['high'].idxmax()
            lowest_idx = window_df['low'].idxmin()
            highest_price = window_df['high'].max()
            lowest_price = window_df['low'].min()
            
            fib_range = highest_price - lowest_price
            if fib_range <= 0:
                continue
            
            # Determine impulse based on EMA trend
            if ema_trend == 'UPTREND':
                impulse_type = 'UP'
                ote_high = highest_price - fib_range * 0.618
                ote_low = highest_price - fib_range * 0.786
            else:
                impulse_type = 'DOWN'
                ote_low = lowest_price + fib_range * 0.618
                ote_high = lowest_price + fib_range * 0.786
            
            # === MODULE 3: CHECK IF IN OTE ===
            price_in_ote = ote_low <= current_price <= ote_high
            
            if not price_in_ote:
                continue
            
            # Check candlestick pattern (optional - for feature)
            pattern, pattern_type, strength = self.detect_candlestick_pattern(df, i - 1)
            
            # For ML training: we want ALL signals in OTE, not just ones with patterns
            # Pattern match is recorded as a feature, not a filter
            has_pattern = pattern is not None
            pattern_matches_trend = False
            
            if ema_trend == 'UPTREND' and pattern_type == 'BULLISH':
                pattern_matches_trend = True
            elif ema_trend == 'DOWNTREND' and pattern_type == 'BEARISH':
                pattern_matches_trend = True
            
            # Determine action based on trend
            action = 'LONG' if ema_trend == 'UPTREND' else 'SHORT'
            
            # === SIGNAL FOUND! Now check outcome ===
            entry_price = current_price
            atr = df['ATR'].iloc[i] if not pd.isna(df['ATR'].iloc[i]) else fib_range * 0.02
            
            if action == 'LONG':
                sl_price = ote_low - atr * 1.5
                tp_price = highest_price  # TP at swing high
            else:  # SHORT
                sl_price = ote_high + atr * 1.5
                tp_price = lowest_price  # TP at swing low
            
            # Check future candles for outcome
            outcome, exit_price, exit_idx = self.check_trade_outcome(
                df, i, action, entry_price, sl_price, tp_price, FUTURE_CANDLES
            )
            
            # Calculate R:R achieved
            if action == 'LONG':
                risk = entry_price - sl_price
                reward = exit_price - entry_price
            else:
                risk = sl_price - entry_price
                reward = entry_price - exit_price
            
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Collect signal data
            signal = {
                'timestamp': df.index[i],
                'action': action,
                'trend': ema_trend,
                'pattern': pattern if pattern else 'NONE',
                'pattern_type': pattern_type if pattern_type else 'NONE',
                'has_pattern': 1 if has_pattern else 0,
                'pattern_matches_trend': 1 if pattern_matches_trend else 0,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'exit_price': exit_price,
                'outcome': outcome,
                'outcome_binary': 1 if outcome == 'WIN' else 0,
                'rr_achieved': rr_ratio,
                # Features for ML
                'rsi': rsi,
                'ema50_slope': df['EMA50_Slope'].iloc[i],
                'ema200_slope': df['EMA200_Slope'].iloc[i],
                'price_vs_ema50': df['Price_vs_EMA50'].iloc[i],
                'price_vs_ema200': df['Price_vs_EMA200'].iloc[i],
                'vol_ratio': df['Vol_Ratio'].iloc[i],
                'bb_width': df['BB_Width'].iloc[i],
                'atr': atr,
                'fib_range': fib_range,
            }
            signals.append(signal)
        
        print(f"    ✅ Found {len(signals)} signals")
        return signals
    
    def check_trade_outcome(self, df, entry_idx, action, entry, sl, tp, max_candles):
        """
        Check if trade hit TP or SL within max_candles.
        Returns: (outcome, exit_price, exit_idx)
        """
        for i in range(1, max_candles + 1):
            idx = entry_idx + i
            if idx >= len(df):
                break
            
            candle_high = df['high'].iloc[idx]
            candle_low = df['low'].iloc[idx]
            
            if action == 'LONG':
                # Check SL first (assumes SL hit before TP if both in same candle)
                if candle_low <= sl:
                    return 'LOSS', sl, idx
                if candle_high >= tp:
                    return 'WIN', tp, idx
            else:  # SHORT
                if candle_high >= sl:
                    return 'LOSS', sl, idx
                if candle_low <= tp:
                    return 'WIN', tp, idx
        
        # Neither hit within max_candles - use close of last candle
        exit_idx = min(entry_idx + max_candles, len(df) - 1)
        exit_price = df['close'].iloc[exit_idx]
        
        if action == 'LONG':
            outcome = 'WIN' if exit_price > entry else 'LOSS'
        else:
            outcome = 'WIN' if exit_price < entry else 'LOSS'
        
        return outcome, exit_price, exit_idx
    
    def save_dataset(self, signals, filename='elliott_signals.csv'):
        """Save signal dataset to CSV."""
        if not signals:
            print("[!] No signals to save")
            return None
        
        df = pd.DataFrame(signals)
        filepath = os.path.join(DATA_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"\n[+] Saved {len(signals)} signals to {filepath}")
        
        # Print summary
        wins = len([s for s in signals if s['outcome'] == 'WIN'])
        losses = len([s for s in signals if s['outcome'] == 'LOSS'])
        win_rate = wins / len(signals) * 100 if signals else 0
        
        print(f"\n=== Dataset Summary ===")
        print(f"    Total Signals: {len(signals)}")
        print(f"    Wins: {wins} ({win_rate:.1f}%)")
        print(f"    Losses: {losses} ({100-win_rate:.1f}%)")
        
        return df


def run_data_mining():
    """Main function to run the data mining process."""
    print("\n" + "="*60)
    print("  PHASE 2: DATA MINING")
    print("="*60)
    
    miner = HistoricalDataMiner()
    
    # Step 1: Fetch historical data
    df = miner.fetch_historical_klines(
        symbol=SYMBOL,
        interval='1h',
        days=HISTORY_DAYS
    )
    
    if df is None or len(df) == 0:
        print("[!] Failed to fetch data")
        return
    
    # Save raw data
    raw_filepath = os.path.join(DATA_DIR, 'btc_1h_raw.csv')
    df.to_csv(raw_filepath)
    print(f"[+] Saved raw data to {raw_filepath}")
    
    # Step 2: Calculate indicators
    df = miner.calculate_indicators(df)
    
    # Step 3: Scan for Elliott Wave signals
    signals = miner.run_elliott_wave_scan(df, lookback=LOOKBACK_CANDLES)
    
    # Step 4: Save labeled dataset
    miner.save_dataset(signals)
    
    print("\n[*] Data mining complete!")
    return signals


if __name__ == '__main__':
    run_data_mining()
