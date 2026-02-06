import pandas as pd
import numpy as np
import math
import os
import pickle
from datetime import datetime
from .indicators import calculate_indicators, calculate_sr_channels, detect_pivot_points
from .config import Config

# ============================================
# ML MODEL LOADER (Phase 4)
# ============================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_ml_model = None
_ml_scaler = None
_ml_features = None

def load_ml_model():
    """Load trained ML model for win probability prediction."""
    global _ml_model, _ml_scaler, _ml_features
    
    if _ml_model is not None:
        return True  # Already loaded
    
    model_path = os.path.join(MODEL_DIR, 'elliott_model.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
    features_path = os.path.join(MODEL_DIR, 'features.txt')
    
    if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
        print("  [!] ML Model not found. Running without AI filter.")
        return False
    
    try:
        with open(model_path, 'rb') as f:
            _ml_model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            _ml_scaler = pickle.load(f)
        with open(features_path, 'r') as f:
            _ml_features = f.read().strip().split('\n')
        print("  [+] ML Model loaded successfully (AI Brain Active)")
        return True
    except Exception as e:
        print(f"  [!] Error loading ML model: {e}")
        return False


def predict_win_probability(df, signal_data):
    """
    Predict win probability for ANY strategy signal using trained ML model.
    Agnostic to strategy type (Elliott, Golden Pocket, etc.)
    
    Args:
        df: DataFrame with indicators calculated
        signal_data: Dict containing strategy results (needs 'action', 'trend')
    
    Returns:
        float: Win probability (0.0 to 1.0), or None if model not available
    """
    global _ml_model, _ml_scaler, _ml_features
    
    if _ml_model is None:
        load_ml_model()
    
    if _ml_model is None or _ml_scaler is None:
        return None
    
    try:
        # Extract features
        current_idx = -1
        
        # Determine trend match based on whatever signal data we have
        signal_trend = signal_data.get('trend', 'NEUTRAL')
        signal_action = signal_data.get('action', 'WAIT')
        
        # Check candlestick pattern (if available in signal_data, else we detect below)
        cp = signal_data.get('candlestick_pattern')

        has_pattern = 0
        pattern_matches_trend = 0
        
        # Try to infer pattern from df directly if not provided
        # This mirrors what we did in data_miner
        # (Re-implementing simplified detection here for robustness)
        curr_row = df.iloc[-2] # Last closed candle
        prev_row = df.iloc[-3]
        
        # Simple Pinbar Check
        body = abs(curr_row['close'] - curr_row['open'])
        wick_up = curr_row['high'] - max(curr_row['open'], curr_row['close'])
        wick_down = min(curr_row['open'], curr_row['close']) - curr_row['low']
        
        pattern_type = None
        if wick_down > 2 * body: pattern_type = 'BULLISH'
        elif wick_up > 2 * body: pattern_type = 'BEARISH'
        
        if pattern_type:
            has_pattern = 1
            if (signal_action == 'LONG' and pattern_type == 'BULLISH') or \
               (signal_action == 'SHORT' and pattern_type == 'BEARISH'):
                pattern_matches_trend = 1

        features = {
            'rsi': df['RSI'].iloc[current_idx] if 'RSI' in df.columns else 50,
            'ema50_slope': df['EMA50'].pct_change(periods=5).iloc[current_idx] * 100 if 'EMA50' in df.columns else 0,
            'ema200_slope': df['EMA200'].pct_change(periods=10).iloc[current_idx] * 100 if 'EMA200' in df.columns else 0,
            'price_vs_ema50': (df['close'].iloc[current_idx] - df['EMA50'].iloc[current_idx]) / df['EMA50'].iloc[current_idx] * 100 if 'EMA50' in df.columns else 0,
            'price_vs_ema200': (df['close'].iloc[current_idx] - df['EMA200'].iloc[current_idx]) / df['EMA200'].iloc[current_idx] * 100 if 'EMA200' in df.columns else 0,
            'vol_ratio': df['Vol_Ratio'].iloc[current_idx] if 'Vol_Ratio' in df.columns else 1,
            'bb_width': df['BB_Width'].iloc[current_idx] if 'BB_Width' in df.columns else 0.02,
            'has_pattern': has_pattern,
            'pattern_matches_trend': pattern_matches_trend,
        }
        
        # Build feature vector in correct order
        X = np.array([[features.get(f, 0) for f in _ml_features]])
        
        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale and predict
        X_scaled = _ml_scaler.transform(X)
        prob = _ml_model.predict_proba(X_scaled)[0, 1]
        
        return prob
    
    except Exception as e:
        print(f"  [!] ML Prediction error: {e}")
        return None

# ============================================
# CONFIGURATION
# ============================================
CHARTPRIME_CONFIG = {
    'trend_levels_length': 10,
    'lro_length': 20,
    'lro_upper_threshold': 1.5,
    'lro_lower_threshold': -1.5,
    'timeframe': '15m',
}

GANN_ANGLES = {
    '1x8': 82.5, '1x4': 75.0, '1x3': 71.25, '1x2': 63.75,
    '1x1': 45.0, '2x1': 26.25, '3x1': 18.75, '4x1': 15.0, '8x1': 7.5,
}

# ============================================
# HELPERS
# ============================================
def get_fibonacci_levels(df):
    """Calculate Fibonacci levels"""
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    diff = swing_high - swing_low
    
    return {
        '0.0': swing_high,
        '0.236': swing_high - 0.236 * diff,
        '0.382': swing_high - 0.382 * diff,
        '0.5': swing_high - 0.5 * diff,
        '0.618': swing_high - 0.618 * diff,
        '0.786': swing_high - 0.786 * diff,
        '1.0': swing_low
    }, swing_high, swing_low

def calculate_pivot_points(df):
    """Calculate Classic Pivot Points from Daily Data"""
    # Use last closed candle
    high = df['high'].iloc[-2]
    low = df['low'].iloc[-2]
    close = df['close'].iloc[-2]
    
    pp = (high + low + close) / 3
    r1 = (2 * pp) - low
    r2 = pp + (high - low)
    r3 = high + 2 * (pp - low)
    s1 = (2 * pp) - high
    s2 = pp - (high - low)
    s3 = low - 2 * (high - pp)
    
    return {'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3}

def calculate_poc(df, num_bins=50):
    """Calculate Point of Control (POC)"""
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, num_bins + 1)
    volume_profile = np.zeros(num_bins)
    
    for i in range(len(df)):
        row = df.iloc[i]
        candle_low = row['low']
        candle_high = row['high']
        candle_vol = row['volume']
        
        for j in range(num_bins):
            bin_low = price_bins[j]
            bin_high = price_bins[j + 1]
            if bin_high >= candle_low and bin_low <= candle_high:
                overlap_low = max(bin_low, candle_low)
                overlap_high = min(bin_high, candle_high)
                candle_range = candle_high - candle_low if candle_high > candle_low else 1
                overlap_pct = (overlap_high - overlap_low) / candle_range
                volume_profile[j] += candle_vol * overlap_pct
                
    poc_index = np.argmax(volume_profile)
    poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
    
    # Value Area
    total_volume = np.sum(volume_profile)
    target_volume = total_volume * 0.7
    sorted_indices = np.argsort(volume_profile)[::-1]
    va_volume = 0
    va_indices = []
    
    for idx in sorted_indices:
        va_indices.append(idx)
        va_volume += volume_profile[idx]
        if va_volume >= target_volume: break
            
    va_high = price_bins[max(va_indices) + 1]
    va_low = price_bins[min(va_indices)]
    
    return {'poc': poc_price, 'va_high': va_high, 'va_low': va_low}

# ============================================
# SMC & ADVANCED ANALYSIS
# ============================================
def detect_wyckoff_phase(df, lookback=50):
    """Detect Wyckoff market phase based on price action and volume"""
    if len(df) < lookback:
        return {'phase': 'UNKNOWN', 'confidence': 0, 'description': 'Insufficient data'}
    
    recent = df.tail(lookback)
    price_range = recent['high'].max() - recent['low'].min()
    current_price = df['close'].iloc[-1]
    avg_volume = recent['volume'].mean()
    recent_volume = df['volume'].iloc[-5:].mean()
    
    swing_high = recent['high'].max()
    swing_low = recent['low'].min()
    price_position = (current_price - swing_low) / price_range if price_range > 0 else 0.5
    
    volume_increasing = recent_volume > avg_volume * 1.2
    volume_decreasing = recent_volume < avg_volume * 0.8
    
    ema20 = recent['close'].ewm(span=20).mean().iloc[-1]
    ema50 = recent['close'].ewm(span=50).mean().iloc[-1] if len(recent) >= 50 else ema20
    
    uptrend = ema20 > ema50 and current_price > ema20
    downtrend = ema20 < ema50 and current_price < ema20
    
    # Spring detection
    recent_lows = recent['low'].tail(10)
    potential_spring = recent_lows.iloc[-1] < recent_lows.iloc[:-1].min() and current_price > recent_lows.iloc[-1]
    
    # UTAD detection
    recent_highs = recent['high'].tail(10)
    potential_utad = recent_highs.iloc[-1] > recent_highs.iloc[:-1].max() and current_price < recent_highs.iloc[-1]
    
    if price_position < 0.3 and volume_decreasing and not downtrend:
        phase = 'ACCUMULATION'
        confidence = 70 + (10 if potential_spring else 0)
        description = 'Smart money accumulating at lows'
        signal = 'BULL'
    elif price_position > 0.7 and volume_decreasing and not uptrend:
        phase = 'DISTRIBUTION'
        confidence = 70 + (10 if potential_utad else 0)
        description = 'Smart money distributing at highs'
        signal = 'BEAR'
    elif uptrend and volume_increasing:
        phase = 'MARKUP'
        confidence = 65
        description = 'Uptrend after accumulation'
        signal = 'BULL'
    elif downtrend and volume_increasing:
        phase = 'MARKDOWN'
        confidence = 65
        description = 'Downtrend after distribution'
        signal = 'BEAR'
    elif potential_spring:
        phase = 'SPRING'
        confidence = 75
        description = 'False breakdown - bullish reversal signal'
        signal = 'BULL'
    elif potential_utad:
        phase = 'UTAD'
        confidence = 75
        description = 'Upthrust After Distribution - bearish reversal'
        signal = 'BEAR'
    else:
        phase = 'RANGING'
        confidence = 50
        description = 'Sideways consolidation'
        signal = 'NEUT'
    
    return {
        'phase': phase,
        'confidence': confidence,
        'description': description,
        'signal': signal,
        'price_position': price_position,
        'swing_high': swing_high,
        'swing_low': swing_low,
    }


def find_fvg(df, min_gap_pct=0.1, lookback=50):
    """Find Fair Value Gaps (imbalance zones)"""
    fvgs = []
    current_price = df['close'].iloc[-1]
    start_idx = max(2, len(df) - lookback)
    
    for i in range(start_idx, len(df)):
        # Bullish FVG
        if df['high'].iloc[i-2] < df['low'].iloc[i]:
            gap_size = df['low'].iloc[i] - df['high'].iloc[i-2]
            gap_pct = (gap_size / current_price) * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    'type': 'bullish',
                    'top': df['low'].iloc[i],
                    'bottom': df['high'].iloc[i-2],
                    'mid': (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2,
                    'size_pct': gap_pct,
                    'index': i,
                    'filled': current_price <= df['low'].iloc[i],
                })
        
        # Bearish FVG
        if df['low'].iloc[i-2] > df['high'].iloc[i]:
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i]
            gap_pct = (gap_size / current_price) * 100
            if gap_pct >= min_gap_pct:
                fvgs.append({
                    'type': 'bearish',
                    'top': df['low'].iloc[i-2],
                    'bottom': df['high'].iloc[i],
                    'mid': (df['low'].iloc[i-2] + df['high'].iloc[i]) / 2,
                    'size_pct': gap_pct,
                    'index': i,
                    'filled': current_price >= df['high'].iloc[i],
                })
    
    unfilled = [f for f in fvgs if not f['filled']]
    unfilled.sort(key=lambda x: abs(x['mid'] - current_price))
    
    return {
        'all_fvgs': fvgs,
        'unfilled': unfilled[:5],
        'nearest_bullish': next((f for f in unfilled if f['type'] == 'bullish'), None),
        'nearest_bearish': next((f for f in unfilled if f['type'] == 'bearish'), None),
    }


def find_order_blocks(df, lookback=50, min_move_pct=1.0):
    """Find Order Blocks (last opposite candle before a strong move)"""
    order_blocks = []
    current_price = df['close'].iloc[-1]
    start_idx = max(3, len(df) - lookback)
    
    for i in range(start_idx, len(df) - 1):
        future_high = df['high'].iloc[i+1:min(i+6, len(df))].max()
        future_low = df['low'].iloc[i+1:min(i+6, len(df))].min()
        
        candle_close = df['close'].iloc[i]
        candle_open = df['open'].iloc[i]
        is_bearish = candle_close < candle_open
        is_bullish = candle_close > candle_open
        
        # Bullish OB
        up_move_pct = ((future_high - candle_close) / candle_close) * 100
        if is_bearish and up_move_pct >= min_move_pct:
            order_blocks.append({
                'type': 'bullish',
                'top': df['high'].iloc[i],
                'bottom': df['low'].iloc[i],
                'mid': (df['high'].iloc[i] + df['low'].iloc[i]) / 2,
                'index': i,
                'strength': up_move_pct,
                'mitigated': current_price < df['low'].iloc[i],
            })
        
        # Bearish OB
        down_move_pct = ((candle_close - future_low) / candle_close) * 100
        if is_bullish and down_move_pct >= min_move_pct:
            order_blocks.append({
                'type': 'bearish',
                'top': df['high'].iloc[i],
                'bottom': df['low'].iloc[i],
                'mid': (df['high'].iloc[i] + df['low'].iloc[i]) / 2,
                'index': i,
                'strength': down_move_pct,
                'mitigated': current_price > df['high'].iloc[i],
            })
    
    unmitigated = [ob for ob in order_blocks if not ob['mitigated']]
    unmitigated.sort(key=lambda x: abs(x['mid'] - current_price))
    
    return {
        'all_obs': order_blocks,
        'unmitigated': unmitigated[:5],
        'nearest_bullish': next((ob for ob in unmitigated if ob['type'] == 'bullish'), None),
        'nearest_bearish': next((ob for ob in unmitigated if ob['type'] == 'bearish'), None),
    }


def find_liquidity_zones(df, lookback=50, tolerance=0.1):
    """Find Liquidity Zones (equal highs/lows where stop losses cluster)"""
    current_price = df['close'].iloc[-1]
    recent = df.tail(lookback)
    
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent) - 2):
        if (recent['high'].iloc[i] > recent['high'].iloc[i-1] and
            recent['high'].iloc[i] > recent['high'].iloc[i-2] and
            recent['high'].iloc[i] > recent['high'].iloc[i+1] and
            recent['high'].iloc[i] > recent['high'].iloc[i+2]):
            swing_highs.append(recent['high'].iloc[i])
        
        if (recent['low'].iloc[i] < recent['low'].iloc[i-1] and
            recent['low'].iloc[i] < recent['low'].iloc[i-2] and
            recent['low'].iloc[i] < recent['low'].iloc[i+1] and
            recent['low'].iloc[i] < recent['low'].iloc[i+2]):
            swing_lows.append(recent['low'].iloc[i])
    
    bsl_zones = []
    for i, h1 in enumerate(swing_highs):
        for h2 in swing_highs[i+1:]:
            diff_pct = abs(h1 - h2) / h1 * 100
            if diff_pct <= tolerance:
                bsl_zones.append({
                    'level': max(h1, h2),
                    'type': 'BSL',
                    'description': 'Equal Highs - Buy-Side Liquidity',
                    'distance_pct': ((max(h1, h2) - current_price) / current_price) * 100,
                })
    
    ssl_zones = []
    for i, l1 in enumerate(swing_lows):
        for l2 in swing_lows[i+1:]:
            diff_pct = abs(l1 - l2) / l1 * 100
            if diff_pct <= tolerance:
                ssl_zones.append({
                    'level': min(l1, l2),
                    'type': 'SSL',
                    'description': 'Equal Lows - Sell-Side Liquidity',
                    'distance_pct': ((current_price - min(l1, l2)) / current_price) * 100,
                })
    
    if swing_highs:
        bsl_zones.append({
            'level': max(swing_highs),
            'type': 'BSL',
            'description': 'Swing High - Buy-Side Liquidity',
            'distance_pct': ((max(swing_highs) - current_price) / current_price) * 100,
        })
    
    if swing_lows:
        ssl_zones.append({
            'level': min(swing_lows),
            'type': 'SSL',
            'description': 'Swing Low - Sell-Side Liquidity',
            'distance_pct': ((current_price - min(swing_lows)) / current_price) * 100,
        })
    
    bsl_zones.sort(key=lambda x: abs(x['distance_pct']))
    ssl_zones.sort(key=lambda x: abs(x['distance_pct']))
    
    return {
        'bsl': bsl_zones[:3],
        'ssl': ssl_zones[:3],
        'nearest_bsl': bsl_zones[0] if bsl_zones else None,
        'nearest_ssl': ssl_zones[0] if ssl_zones else None,
    }


def analyze_smc(df):
    """Combined Smart Money Concepts analysis"""
    wyckoff = detect_wyckoff_phase(df)
    fvgs = find_fvg(df)
    obs = find_order_blocks(df)
    liquidity = find_liquidity_zones(df)
    
    current_price = df['close'].iloc[-1]
    signals = []
    score = 0
    
    # 1. Wyckoff Signal (±2)
    if wyckoff['phase'] in ['ACCUMULATION', 'SPRING', 'MARKUP']:
        score += 2
        signals.append(('Wyckoff', f"{wyckoff['phase']} ({wyckoff['confidence']}%)", 'BULL'))
    elif wyckoff['phase'] in ['DISTRIBUTION', 'UTAD', 'MARKDOWN']:
        score -= 2
        signals.append(('Wyckoff', f"{wyckoff['phase']} ({wyckoff['confidence']}%)", 'BEAR'))
    else:
        signals.append(('Wyckoff', f"{wyckoff['phase']}", 'NEUT'))
    
    # 2. FVG Signal (±1)
    nearest_fvg = None
    if fvgs['nearest_bullish'] and fvgs['nearest_bearish']:
        bull_dist = abs(fvgs['nearest_bullish']['mid'] - current_price)
        bear_dist = abs(fvgs['nearest_bearish']['mid'] - current_price)
        nearest_fvg = fvgs['nearest_bullish'] if bull_dist < bear_dist else fvgs['nearest_bearish']
    elif fvgs['nearest_bullish']:
        nearest_fvg = fvgs['nearest_bullish']
    elif fvgs['nearest_bearish']:
        nearest_fvg = fvgs['nearest_bearish']
    
    if nearest_fvg:
        dist_pct = abs(nearest_fvg['mid'] - current_price) / current_price * 100
        if dist_pct < 2:
            if nearest_fvg['type'] == 'bullish' and current_price > nearest_fvg['mid']:
                score += 1
                signals.append(('FVG', f"Bullish FVG ${nearest_fvg['mid']:,.0f} (Support)", 'BULL'))
            elif nearest_fvg['type'] == 'bearish' and current_price < nearest_fvg['mid']:
                score -= 1
                signals.append(('FVG', f"Bearish FVG ${nearest_fvg['mid']:,.0f} (Resistance)", 'BEAR'))
    
    # 3. Order Block Signal (±1)
    nearest_ob = None
    if obs['nearest_bullish'] and obs['nearest_bearish']:
        bull_dist = abs(obs['nearest_bullish']['mid'] - current_price)
        bear_dist = abs(obs['nearest_bearish']['mid'] - current_price)
        nearest_ob = obs['nearest_bullish'] if bull_dist < bear_dist else obs['nearest_bearish']
    elif obs['nearest_bullish']:
        nearest_ob = obs['nearest_bullish']
    elif obs['nearest_bearish']:
        nearest_ob = obs['nearest_bearish']
    
    if nearest_ob:
        dist_pct = abs(nearest_ob['mid'] - current_price) / current_price * 100
        if dist_pct < 1.5:
            if nearest_ob['type'] == 'bullish':
                score += 1
                signals.append(('OB', f"Bullish OB ${nearest_ob['mid']:,.0f} (Demand)", 'BULL'))
            else:
                score -= 1
                signals.append(('OB', f"Bearish OB ${nearest_ob['mid']:,.0f} (Supply)", 'BEAR'))
    
    # 4. Liquidity Warning
    if liquidity['nearest_bsl'] and liquidity['nearest_bsl']['distance_pct'] < 2:
        signals.append(('Liquidity', f"BSL ${liquidity['nearest_bsl']['level']:,.0f} ⚠️", 'BEAR'))
        score -= 1
    elif liquidity['nearest_ssl'] and abs(liquidity['nearest_ssl']['distance_pct']) < 2:
        signals.append(('Liquidity', f"SSL ${liquidity['nearest_ssl']['level']:,.0f} ⚠️", 'BULL'))
        score += 1
    
    return {
        'wyckoff': wyckoff,
        'fvgs': fvgs,
        'order_blocks': obs,
        'liquidity': liquidity,
        'signals': signals,
        'score': score,
        'nearest_fvg': nearest_fvg,
        'nearest_ob': nearest_ob,
    }


# ============================================
# GANN & LUNAR
# ============================================
def calculate_gann_fan(df, pivot_type='low'):
    """Calculate Gann Fan lines from pivot point"""
    from .config import GANN_ANGLES
    
    if pivot_type == 'low':
        pivot_idx = df['low'].idxmin()
        pivot_price = df.loc[pivot_idx, 'low']
        direction = 1
    else:
        pivot_idx = df['high'].idxmax()
        pivot_price = df.loc[pivot_idx, 'high']
        direction = -1
    
    pivot_pos = df.index.get_loc(pivot_idx)
    price_range = df['high'].max() - df['low'].min()
    time_range = len(df)
    scale = price_range / time_range
    
    gann_lines = {}
    for name, angle in GANN_ANGLES.items():
        angle_rad = math.radians(angle)
        slope = math.tan(angle_rad) * scale * direction
        x_end = len(df) + 20
        y_end = pivot_price + slope * (x_end - pivot_pos)
        
        gann_lines[name] = {
            'start': (pivot_pos, pivot_price),
            'end': (x_end, y_end),
            'slope': slope,
            'angle': angle
        }
    
    return gann_lines, pivot_pos, pivot_price


def get_gann_signal(current_price, gann_lines, pivot_price):
    """Determine Gann signal based on current price position"""
    above = []
    below = []
    
    for name, line in gann_lines.items():
        current_gann_price = line['start'][1] + line['slope'] * (100 - line['start'][0])
        if current_gann_price > current_price:
            above.append((name, current_gann_price))
        else:
            below.append((name, current_gann_price))
    
    if not above:
        return 'STRONG_BULLISH', 'Above all Gann lines'
    elif not below:
        return 'STRONG_BEARISH', 'Below all Gann lines'
    else:
        nearest_above = min(above, key=lambda x: x[1])
        nearest_below = max(below, key=lambda x: x[1])
        
        for name, line in gann_lines.items():
            if name == '1x1':
                gann_1x1 = line['start'][1] + line['slope'] * (100 - line['start'][0])
                diff_pct = abs(current_price - gann_1x1) / current_price * 100
                if diff_pct < 1:
                    return 'AT_1x1', f'At main Gann 1x1 line (${gann_1x1:,.0f})'
        
        return 'BETWEEN', f'Between {nearest_below[0]} and {nearest_above[0]}'


def get_moon_phase_simple(date=None):
    """Calculate moon phase using simple algorithm"""
    if date is None:
        date = datetime.now()
    
    new_moon_ref = datetime(2000, 1, 6, 18, 14)
    lunar_cycle = 29.530588853
    days_since = (date - new_moon_ref).total_seconds() / 86400
    phase_position = (days_since % lunar_cycle) / lunar_cycle
    phase_percentage = phase_position * 100
    
    if phase_position < 0.0625:
        phase_name, emoji = 'New Moon', '🌑'
    elif phase_position < 0.1875:
        phase_name, emoji = 'Waxing Crescent', '🌒'
    elif phase_position < 0.3125:
        phase_name, emoji = 'First Quarter', '🌓'
    elif phase_position < 0.4375:
        phase_name, emoji = 'Waxing Gibbous', '🌔'
    elif phase_position < 0.5625:
        phase_name, emoji = 'Full Moon', '🌕'
    elif phase_position < 0.6875:
        phase_name, emoji = 'Waning Gibbous', '🌖'
    elif phase_position < 0.8125:
        phase_name, emoji = 'Last Quarter', '🌗'
    elif phase_position < 0.9375:
        phase_name, emoji = 'Waning Crescent', '🌘'
    else:
        phase_name, emoji = 'New Moon', '🌑'
    
    return {
        'phase_name': phase_name,
        'emoji': emoji,
        'percentage': phase_percentage,
        'position': phase_position,
        'days_to_full': (0.5 - phase_position) * lunar_cycle if phase_position < 0.5 else ((1 - phase_position) + 0.5) * lunar_cycle,
        'days_to_new': (1 - phase_position) * lunar_cycle if phase_position > 0 else 0
    }


def get_lunar_trading_signal(moon_phase):
    """Generate trading signal based on lunar phase"""
    position = moon_phase['position']
    
    if 0.45 <= position <= 0.55:
        signal = 'CAUTION'
        desc = 'Full Moon - High volatility, potential reversal'
        sentiment = 'NEUTRAL'
    elif 0.95 <= position or position <= 0.05:
        signal = 'OPPORTUNITY'
        desc = 'New Moon - New trend potential, accumulation'
        sentiment = 'BULLISH'
    elif 0.2 <= position <= 0.3:
        signal = 'MOMENTUM'
        desc = 'First Quarter - Waxing momentum, bullish bias'
        sentiment = 'BULLISH'
    elif 0.7 <= position <= 0.8:
        signal = 'CORRECTION'
        desc = 'Last Quarter - Waning energy, bearish bias'
        sentiment = 'BEARISH'
    else:
        signal = 'NORMAL'
        desc = 'No significant lunar influence'
        sentiment = 'NEUTRAL'
    
    return {
        'signal': signal,
        'description': desc,
        'sentiment': sentiment,
        'phase': moon_phase
    }


def get_mercury_retrograde():
    """Check if Mercury is in retrograde"""
    mercury_retrogrades_2026 = [
        (datetime(2026, 1, 9), datetime(2026, 1, 29)),
        (datetime(2026, 5, 2), datetime(2026, 5, 25)),
        (datetime(2026, 8, 28), datetime(2026, 9, 21)),
        (datetime(2026, 12, 19), datetime(2027, 1, 8)),
    ]
    
    now = datetime.now()
    
    for start, end in mercury_retrogrades_2026:
        if start <= now <= end:
            return {
                'is_retrograde': True,
                'start': start,
                'end': end,
                'days_remaining': (end - now).days,
                'signal': 'CAUTION',
                'description': 'Mercury Retrograde - Avoid major decisions, expect miscommunication'
            }
    
    return {
        'is_retrograde': False,
        'signal': 'NORMAL',
        'description': 'Mercury Direct - Normal trading conditions'
    }


# ============================================
# MAIN TRADING PLAN
# ============================================
def analyze_timeframe_detailed(df, tf_name):
    """Analyze single timeframe"""
    df = calculate_indicators(df)
    current = df['close'].iloc[-1]
    
    bias = 'NEUTRAL'
    bullish = 0
    bearish = 0
    
    if df['RSI'].iloc[-1] < 40: bullish += 1
    elif df['RSI'].iloc[-1] > 60: bearish += 1
    
    if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1]: bullish += 1
    else: bearish += 1
    
    if df['close'].iloc[-1] > df['EMA50'].iloc[-1]: bullish += 1
    else: bearish += 1
    
    if bullish > bearish: bias = 'BULLISH'
    elif bearish > bullish: bias = 'BEARISH'
    
    return {'timeframe': tf_name, 'price': current, 'bias': bias, 'rsi': df['RSI'].iloc[-1]}

def calculate_trading_plan(df, pivots, poc_data, timeframes_bias, gann_data=None, lunar_data=None, derivatives_data=None, smc_data=None, chartprime_data=None):
    """Calculate Trading Plan"""
    score = 0
    signals = []
    
    # Bias Score
    bull_count = sum(1 for b in timeframes_bias.values() if b == 'BULLISH')
    bear_count = sum(1 for b in timeframes_bias.values() if b == 'BEARISH')
    
    if bull_count >= 3: score += 2
    elif bear_count >= 3: score -= 2
    
    # RSI Score
    rsi = df['RSI'].iloc[-1]
    if rsi < 30: score += 2
    elif rsi > 70: score -= 2
    
    # Direction
    if score >= 1: direction = 'LONG'
    elif score <= -1: direction = 'SHORT'
    else: direction = 'WAIT'
    
    # Phase 5: AI Win Probability for Recommendation (ALWAYS Calculate)
    # If WAIT, we still predict for the likely next action based on score tendency
    if direction == 'WAIT':
        # Use bias to determine potential action
        potential_action = 'LONG' if bull_count > bear_count else 'SHORT'
    else:
        potential_action = direction
    
    rec_signal_data = {
        'action': potential_action,
        'trend': 'BULLISH' if potential_action == 'LONG' else 'BEARISH'
    }
    rec_win_prob = predict_win_probability(df, rec_signal_data)
    
    # Adjust Score based on AI Confidence (only if not WAIT)
    if direction != 'WAIT' and rec_win_prob is not None:
        if rec_win_prob >= 0.65: score += 1  # Bonus for high AI confidence
        elif rec_win_prob < 0.40: score -= 1 # Penalty for low AI confidence
    
    # Plans
    long_plan = {
        'entry': pivots['S1'], 'sl': pivots['S2'], 
        'tp1': pivots['PP'], 'tp2': pivots['R1'], 'tp3': pivots['R2'],
        'win_probability': rec_win_prob if direction == 'LONG' else None
    }
    short_plan = {
        'entry': pivots['R1'], 'sl': pivots['R2'], 
        'tp1': pivots['PP'], 'tp2': pivots['S1'], 'tp3': pivots['S2'],
        'win_probability': rec_win_prob if direction == 'SHORT' else None
    }
    
    return {
        'direction': direction, 
        'score': score, 
        'signal_analysis': signals, 
        'long': long_plan, 
        'short': short_plan,
        'ai_win_prob': rec_win_prob,  # Return for reporting
        'current_price': df['close'].iloc[-1]
    }

def calculate_golden_pocket_strategy(df, smc_data=None, pivots=None):
    """
    Advanced Golden Pocket Strategy with:
    1. Fractal-based Swing detection (not fixed lookback)
    2. EMA 200 + Structure Trend Filter
    3. ATR-buffered Stop Loss
    """
    result = {
        'valid': False,
        'trend': 'NEUTRAL',
        'swing_high': 0,
        'swing_low': 0,
        'swing_high_idx': 0,
        'swing_low_idx': 0,
        'golden_pocket': {'high': 0, 'low': 0},
        'entry': 0,
        'entry1': 0,  # Aggressive entry (outer edge - easier to fill)
        'entry2': 0,  # Conservative entry (inner edge - better R/R)
        'sl': 0,
        'tp1': 0,
        'tp2': 0,
        'action': 'WAIT',
        'reason': '',
        'atr': 0,
    }
    
    if len(df) < 100:
        result['reason'] = 'Insufficient data (need 100+ candles)'
        return result
    
    current_price = df['close'].iloc[-1]
    
    # ===== HELPER: Find Fractals (Swing High/Low) =====
    def find_fractals(df, left=5, right=5):
        """Find Fractal Highs and Lows (Williams Fractals)"""
        highs = []
        lows = []
        
        for i in range(left, len(df) - right):
            # Fractal High: center candle high is highest among left+right candles
            is_fractal_high = True
            for j in range(1, left + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j]:
                    is_fractal_high = False
                    break
            if is_fractal_high:
                for j in range(1, right + 1):
                    if df['high'].iloc[i] <= df['high'].iloc[i + j]:
                        is_fractal_high = False
                        break
            if is_fractal_high:
                highs.append({'idx': i, 'price': df['high'].iloc[i]})
            
            # Fractal Low: center candle low is lowest among left+right candles
            is_fractal_low = True
            for j in range(1, left + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j]:
                    is_fractal_low = False
                    break
            if is_fractal_low:
                for j in range(1, right + 1):
                    if df['low'].iloc[i] >= df['low'].iloc[i + j]:
                        is_fractal_low = False
                        break
            if is_fractal_low:
                lows.append({'idx': i, 'price': df['low'].iloc[i]})
        
        return highs, lows
    
    # ===== 1. FIND FRACTAL SWINGS (Use larger period for significant swings) =====
    # Use left=10, right=10 to find more significant swing points
    fractal_highs, fractal_lows = find_fractals(df, left=10, right=10)
    
    # Fallback to smaller period if no fractals found
    if not fractal_highs or not fractal_lows:
        fractal_highs, fractal_lows = find_fractals(df, left=5, right=5)
    
    if not fractal_highs or not fractal_lows:
        result['reason'] = 'No fractal swings found'
        return result
    
    # Find the HIGHEST High and LOWEST Low among recent fractals (last 50 candles worth)
    recent_lookback = len(df) - 50  # Only consider fractals in recent 50 candles scope
    
    # Filter fractals that are within reasonable range
    valid_highs = [f for f in fractal_highs if f['idx'] > len(df) - 200]
    valid_lows = [f for f in fractal_lows if f['idx'] > len(df) - 200]
    
    if not valid_highs or not valid_lows:
        valid_highs = fractal_highs[-5:]  # Use last 5
        valid_lows = fractal_lows[-5:]
    
    # Get the HIGHEST fractal high and LOWEST fractal low
    best_high = max(valid_highs, key=lambda x: x['price'])
    best_low = min(valid_lows, key=lambda x: x['price'])
    
    # Structure comparison - use recent fractals
    last_fractal_high = fractal_highs[-1]
    last_fractal_low = fractal_lows[-1]
    prev_fractal_high = fractal_highs[-2] if len(fractal_highs) >= 2 else None
    prev_fractal_low = fractal_lows[-2] if len(fractal_lows) >= 2 else None
    
    result['swing_high'] = best_high['price']
    result['swing_low'] = best_low['price']
    result['swing_high_idx'] = best_high['idx']
    result['swing_low_idx'] = best_low['idx']
    
    # ===== 2. EMA 200 TREND FILTER =====
    # Calculate EMA 200 if not in dataframe
    if 'EMA200' in df.columns:
        ema200 = df['EMA200'].iloc[-1]
    else:
        ema200 = df['close'].ewm(span=200, adjust=False).mean().iloc[-1]
    
    price_above_ema200 = current_price > ema200
    price_below_ema200 = current_price < ema200
    
    # ===== 3. STRUCTURE CHECK (HH/HL vs LH/LL) =====
    is_hh = prev_fractal_high and last_fractal_high['price'] > prev_fractal_high['price']
    is_hl = prev_fractal_low and last_fractal_low['price'] > prev_fractal_low['price']
    is_lh = prev_fractal_high and last_fractal_high['price'] < prev_fractal_high['price']
    is_ll = prev_fractal_low and last_fractal_low['price'] < prev_fractal_low['price']
    
    bullish_structure = is_hh or is_hl
    bearish_structure = is_lh or is_ll
    
    # ===== 4. DETERMINE TREND =====
    if price_above_ema200 and bullish_structure:
        trend = 'BULLISH'
    elif price_below_ema200 and bearish_structure:
        trend = 'BEARISH'
    elif price_above_ema200:
        trend = 'BULLISH'  # EMA filter takes priority
    elif price_below_ema200:
        trend = 'BEARISH'
    else:
        trend = 'NEUTRAL'
    
    result['trend'] = trend
    
    # Debug output
    print(f"\n  🔍 Advanced GP Strategy Debug:")
    print(f"    EMA200: ${ema200:,.0f} | Price {'>' if price_above_ema200 else '<'} EMA200")
    print(f"    Best Swing High: ${best_high['price']:,.0f} (idx {best_high['idx']})")
    print(f"    Best Swing Low: ${best_low['price']:,.0f} (idx {best_low['idx']})")
    print(f"    Fibo Range: ${best_high['price'] - best_low['price']:,.0f}")
    print(f"    Structure: HH={is_hh} HL={is_hl} LH={is_lh} LL={is_ll}")
    print(f"    -> Trend: {trend}")
    
    if trend == 'NEUTRAL':
        result['reason'] = 'Trend is NEUTRAL - No valid setup'
        return result
    
    # ===== 5. ATR CALCULATION (for SL buffer) =====
    # Use ATR 14
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
    else:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        atr = tr.rolling(window=14).mean().iloc[-1]
    
    result['atr'] = atr
    
    # ===== 6. GOLDEN POCKET CALCULATION =====
    # Use the BEST (highest/lowest) swings for Fibo, not just the last ones
    swing_high = best_high['price']
    swing_low = best_low['price']
    fib_range = swing_high - swing_low
    
    if fib_range <= 0:
        result['reason'] = 'No valid swing range'
        return result
    
    if trend == 'BULLISH':
        # Bullish: High is more recent, looking for pullback down to buy
        gp_618 = swing_high - fib_range * 0.618
        gp_5 = swing_high - fib_range * 0.5
        gp_high = gp_5  # Upper edge
        gp_low = gp_618  # Lower edge
        fib_786 = swing_high - fib_range * 0.786
        sl_price = fib_786 - (atr * 1.5)  # ATR buffer
        tp1_price = swing_high
        tp2_price = swing_high + fib_range * 0.272
        
    elif trend == 'BEARISH':
        # Bearish: Low is more recent, looking for pullback up to sell
        gp_618 = swing_low + fib_range * 0.618
        gp_5 = swing_low + fib_range * 0.5
        gp_high = gp_618  # Upper edge
        gp_low = gp_5  # Lower edge
        fib_786 = swing_low + fib_range * 0.786
        sl_price = fib_786 + (atr * 1.5)  # ATR buffer
        tp1_price = swing_low
        tp2_price = swing_low - fib_range * 0.272
    else:
        result['reason'] = 'Trend is NEUTRAL'
        return result
    
    result['golden_pocket'] = {'high': gp_high, 'low': gp_low}
    result['sl'] = sl_price
    result['tp1'] = tp1_price
    result['tp2'] = tp2_price
    
    print(f"    Golden Pocket: ${gp_low:,.0f} - ${gp_high:,.0f}")
    print(f"    ATR(14): ${atr:,.0f} | SL with buffer: ${sl_price:,.0f}")
    
    # ===== 7. ENTRY LOGIC =====
    price_in_gp = gp_low <= current_price <= gp_high
    price_near_gp = gp_low * 0.995 <= current_price <= gp_high * 1.005
    
    if trend == 'BULLISH':
        # Entry1: Lower edge (0.618) - dễ khớp hơn
        # Entry2: Upper edge (0.5) - R/R tốt hơn
        result['entry'] = gp_618  # Legacy (average)
        result['entry1'] = gp_618  # Aggressive - outer edge
        result['entry2'] = gp_5    # Conservative - inner edge
        
        if price_in_gp or price_near_gp:
            result['valid'] = True
            result['action'] = 'LONG'
            result['reason'] = f'Price in Golden Pocket zone (Fractal Swing)'
        elif current_price > gp_high:
            result['action'] = 'MISSED'
            result['reason'] = 'Price already above GP zone'
        elif current_price < gp_low:
            result['action'] = 'WAIT'
            result['reason'] = f'Wait for bounce from GP (${gp_low:,.0f} - ${gp_high:,.0f})'
        else:
            result['action'] = 'WAIT'
            result['reason'] = f'Wait for pullback to GP zone'
    
    elif trend == 'BEARISH':
        # Entry1: Upper edge (0.618) - dễ khớp hơn
        # Entry2: Lower edge (0.5) - R/R tốt hơn
        result['entry'] = gp_618  # Legacy (average)
        result['entry1'] = gp_618  # Aggressive - outer edge
        result['entry2'] = gp_5    # Conservative - inner edge
        
        if price_in_gp or price_near_gp:
            result['valid'] = True
            result['action'] = 'SHORT'
            result['reason'] = f'Price in Golden Pocket zone (Fractal Swing)'
        elif current_price < gp_low:
            result['action'] = 'MISSED'
            result['reason'] = 'Price already below GP zone'
        elif current_price > gp_high:
            result['action'] = 'WAIT'
            result['reason'] = f'Wait for rejection from GP (${gp_low:,.0f} - ${gp_high:,.0f})'
        else:
            result['action'] = 'WAIT'
            result['reason'] = f'Wait for pullback to GP zone'
    
    print(f"    -> Action: {result['action']} | Reason: {result['reason']}")
    
    return result


# ============================================
# ELLIOTT WAVE + FIBONACCI INDICATOR (NEW)
# ============================================

def detect_candlestick_pattern(df):
    """
    Detect Pinbar and Engulfing patterns on the last closed candle.
    Returns: dict with pattern info
    """
    if len(df) < 3:
        return {'pattern': None, 'type': None}
    
    # Use last CLOSED candle (index -2), not current forming candle
    curr = df.iloc[-2]
    prev = df.iloc[-3]
    
    curr_open = curr['open']
    curr_high = curr['high']
    curr_low = curr['low']
    curr_close = curr['close']
    
    prev_open = prev['open']
    prev_close = prev['close']
    
    # Body calculations
    curr_body = abs(curr_close - curr_open)
    curr_upper_wick = curr_high - max(curr_open, curr_close)
    curr_lower_wick = min(curr_open, curr_close) - curr_low
    curr_range = curr_high - curr_low
    
    prev_body = abs(prev_close - prev_open)
    
    result = {'pattern': None, 'type': None, 'strength': 0}
    
    # ===== PINBAR DETECTION =====
    # Bullish Pinbar: Long lower wick (>= 2x body), small upper wick
    if curr_body > 0 and curr_range > 0:
        body_ratio = curr_body / curr_range
        
        # Bullish Pinbar
        if curr_lower_wick >= 2 * curr_body and curr_upper_wick < curr_body:
            if curr_close > curr_open:  # Green candle preferred
                result = {'pattern': 'PINBAR', 'type': 'BULLISH', 'strength': 80}
            else:
                result = {'pattern': 'PINBAR', 'type': 'BULLISH', 'strength': 60}
        
        # Bearish Pinbar
        elif curr_upper_wick >= 2 * curr_body and curr_lower_wick < curr_body:
            if curr_close < curr_open:  # Red candle preferred
                result = {'pattern': 'PINBAR', 'type': 'BEARISH', 'strength': 80}
            else:
                result = {'pattern': 'PINBAR', 'type': 'BEARISH', 'strength': 60}
    
    # ===== ENGULFING DETECTION =====
    # Bullish Engulfing: Current green body engulfs previous red body
    if prev_close < prev_open:  # Previous is red
        if curr_close > curr_open:  # Current is green
            if curr_close > prev_open and curr_open < prev_close:
                engulf_strength = 70 + min(30, (curr_body / prev_body - 1) * 20) if prev_body > 0 else 70
                if result['strength'] < engulf_strength:
                    result = {'pattern': 'ENGULFING', 'type': 'BULLISH', 'strength': engulf_strength}
    
    # Bearish Engulfing: Current red body engulfs previous green body
    if prev_close > prev_open:  # Previous is green
        if curr_close < curr_open:  # Current is red
            if curr_open > prev_close and curr_close < prev_open:
                engulf_strength = 70 + min(30, (curr_body / prev_body - 1) * 20) if prev_body > 0 else 70
                if result['strength'] < engulf_strength:
                    result = {'pattern': 'ENGULFING', 'type': 'BEARISH', 'strength': engulf_strength}
    
    return result


def calculate_elliott_wave_fibo(df, lookback=100, use_ema_filter=True, use_rsi_filter=False):
    """
    Elliott Wave + Fibonacci OTE Strategy with 3-Module Logic.
    
    Module 1 (Data): Uses the provided df with EMA50/EMA200/RSI columns.
    Module 2 (Analysis):
        - Step 1: Trend Filter (EMA 50/200).
        - Step 2: Wave Structure (Impulse detection based on Trend).
        - Step 3: Fibonacci OTE (0.618 - 0.786).
    Module 3 (Signal): Candlestick pattern + RSI confirmation.
    
    Returns: dict with action, entry, sl, tp, reason, etc.
    """
    result = {
        'valid': False,
        'trend': 'NEUTRAL',
        'ema_trend': 'NEUTRAL',  # EMA-based trend
        'impulse_type': None,  # 'UP' or 'DOWN'
        'swing_high': 0,
        'swing_low': 0,
        'swing_high_idx': 0,
        'swing_low_idx': 0,
        'fib_levels': {},
        'ote_zone': {'high': 0, 'low': 0},
        'entry': 0,
        'sl': 0,
        'tp1': 0,
        'tp2': 0,
        'action': 'WAIT',
        'reason': '',
        'candlestick_pattern': None,
        'wave_context': '',
        'rsi': 0,
    }
    
    if len(df) < lookback:
        result['reason'] = f'Không đủ dữ liệu (cần {lookback}+ nến)'
        return result
    
    current_price = df['close'].iloc[-1]
    recent_df = df.tail(lookback).copy()
    
    # ============================================
    # MODULE 2 - STEP 1: EMA TREND FILTER
    # ============================================
    # Get EMAs
    ema50 = df['EMA50'].iloc[-1] if 'EMA50' in df.columns else None
    ema200 = df['EMA200'].iloc[-1] if 'EMA200' in df.columns else None
    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
    
    result['rsi'] = rsi
    
    ema_trend = 'NEUTRAL'
    if ema50 is not None and ema200 is not None:
        if current_price > ema50 > ema200:
            ema_trend = 'UPTREND'
        elif current_price < ema50 < ema200:
            ema_trend = 'DOWNTREND'
        else:
            ema_trend = 'SIDEWAY'
    
    result['ema_trend'] = ema_trend
    
    # Apply filter if enabled
    if use_ema_filter and ema_trend == 'SIDEWAY':
        result['action'] = 'WAIT'
        result['reason'] = '⚪ EMA lộn xộn (Sideway) - Không giao dịch'
        print(f"\n  🌊 Elliott Wave Fibo Analysis:")
        print(f"    EMA Trend: {ema_trend} | EMA50: ${ema50:,.0f} | EMA200: ${ema200:,.0f}")
        print(f"    -> Action: WAIT (Sideway Market)")
        return result
    
    # ============================================
    # MODULE 2 - STEP 2: FIND WAVE STRUCTURE
    # ============================================
    def find_fractals_simple(data, left=5, right=5):
        """Find Fractal Highs and Lows"""
        highs = []
        lows = []
        
        for i in range(left, len(data) - right):
            is_high = True
            is_low = True
            
            for j in range(1, left + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i - j]:
                    is_high = False
                if data['low'].iloc[i] >= data['low'].iloc[i - j]:
                    is_low = False
            
            for j in range(1, right + 1):
                if data['high'].iloc[i] <= data['high'].iloc[i + j]:
                    is_high = False
                if data['low'].iloc[i] >= data['low'].iloc[i + j]:
                    is_low = False
            
            if is_high:
                highs.append({'idx': i, 'price': data['high'].iloc[i], 'global_idx': len(df) - lookback + i})
            if is_low:
                lows.append({'idx': i, 'price': data['low'].iloc[i], 'global_idx': len(df) - lookback + i})
        
        return highs, lows
    
    fractal_highs, fractal_lows = find_fractals_simple(recent_df, left=5, right=5)
    
    if not fractal_highs or not fractal_lows:
        result['reason'] = 'Không tìm thấy Fractal Swing'
        return result
    
    # Find the HIGHEST High and LOWEST Low
    highest = max(fractal_highs, key=lambda x: x['price'])
    lowest = min(fractal_lows, key=lambda x: x['price'])
    
    result['swing_high'] = highest['price']
    result['swing_low'] = lowest['price']
    result['swing_high_idx'] = highest['global_idx']
    result['swing_low_idx'] = lowest['global_idx']
    
    fib_range = highest['price'] - lowest['price']
    
    if fib_range <= 0:
        result['reason'] = 'Không có biên độ sóng hợp lệ'
        return result
    
    # ============================================
    # IMPULSE DIRECTION - BASED ON EMA TREND!
    # ============================================
    # Key change: We no longer determine trend from swing order.
    # EMA determines the ALLOWED direction.
    if use_ema_filter:
        if ema_trend == 'UPTREND':
            impulse_type = 'UP'
            trend = 'BULLISH'
            wave_context = 'Trend TĂNG (EMA) - Chỉ tìm kèo LONG'
        elif ema_trend == 'DOWNTREND':
            impulse_type = 'DOWN'
            trend = 'BEARISH'
            wave_context = 'Trend GIẢM (EMA) - Chỉ tìm kèo SHORT'
        else:
            impulse_type = None
            trend = 'NEUTRAL'
            wave_context = 'Sideway - Không giao dịch'
    else:
        # Fallback to old logic: determine by swing order
        if highest['idx'] > lowest['idx']:
            impulse_type = 'UP'
            trend = 'BULLISH'
            wave_context = 'Sóng đẩy Tăng - Đợi hồi về Fibo để MUA'
        else:
            impulse_type = 'DOWN'
            trend = 'BEARISH'
            wave_context = 'Sóng đẩy Giảm - Đợi hồi lên Fibo để BÁN'
    
    result['impulse_type'] = impulse_type
    result['trend'] = trend
    result['wave_context'] = wave_context
    
    if impulse_type is None:
        result['action'] = 'WAIT'
        result['reason'] = '⚪ Không xác định được xu hướng'
        return result
    
    # ============================================
    # MODULE 2 - STEP 3: FIBONACCI OTE
    # ============================================
    if impulse_type == 'UP':
        # Retracement from HIGH down
        fib_0 = highest['price']
        fib_236 = highest['price'] - fib_range * 0.236
        fib_382 = highest['price'] - fib_range * 0.382
        fib_5 = highest['price'] - fib_range * 0.5
        fib_618 = highest['price'] - fib_range * 0.618
        fib_705 = highest['price'] - fib_range * 0.705
        fib_786 = highest['price'] - fib_range * 0.786
        fib_1 = lowest['price']
        
        ote_high = fib_618
        ote_low = fib_786
        
    else:  # Impulse DOWN
        # Retracement from LOW up
        fib_0 = lowest['price']
        fib_236 = lowest['price'] + fib_range * 0.236
        fib_382 = lowest['price'] + fib_range * 0.382
        fib_5 = lowest['price'] + fib_range * 0.5
        fib_618 = lowest['price'] + fib_range * 0.618
        fib_705 = lowest['price'] + fib_range * 0.705
        fib_786 = lowest['price'] + fib_range * 0.786
        fib_1 = highest['price']
        
        ote_high = fib_786
        ote_low = fib_618
    
    result['fib_levels'] = {
        '0.0': fib_0, '0.236': fib_236, '0.382': fib_382,
        '0.5': fib_5, '0.618': fib_618, '0.705': fib_705,
        '0.786': fib_786, '1.0': fib_1,
    }
    result['ote_zone'] = {'high': ote_high, 'low': ote_low}
    
    # ============================================
    # MODULE 3: SIGNAL CONFIRMATION
    # ============================================
    price_in_ote = ote_low <= current_price <= ote_high
    price_near_ote = ote_low * 0.995 <= current_price <= ote_high * 1.005
    
    # Candlestick pattern
    candle_pattern = detect_candlestick_pattern(df)
    result['candlestick_pattern'] = candle_pattern
    
    has_bullish_confirmation = candle_pattern['type'] == 'BULLISH' and candle_pattern['strength'] >= 60
    has_bearish_confirmation = candle_pattern['type'] == 'BEARISH' and candle_pattern['strength'] >= 60
    
    # RSI Check (Optional)
    rsi_ok_long = rsi < 40 if use_rsi_filter else True
    rsi_ok_short = rsi > 60 if use_rsi_filter else True
    
    # ATR for SL buffer
    if 'ATR' in df.columns:
        atr = df['ATR'].iloc[-1]
    else:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = high_low.combine(high_close, max).combine(low_close, max)
        atr = tr.rolling(window=14).mean().iloc[-1]
    
    # ============================================
    # GENERATE SIGNAL
    # ============================================
    if impulse_type == 'UP':
        # Looking for LONG
        entry_price = fib_618
        sl_price = fib_786 - atr * 1.5
        tp1_price = highest['price']
        tp2_price = highest['price'] + fib_range * 0.272
        
        if price_in_ote or price_near_ote:
            if has_bullish_confirmation:
                if rsi_ok_long:
                    result['valid'] = True
                    result['action'] = 'LONG'
                    result['reason'] = f"✅ Giá trong OTE + {candle_pattern['pattern']} + RSI OK"
                else:
                    result['action'] = 'WAIT'
                    result['reason'] = f"⏳ Có nến {candle_pattern['pattern']} nhưng RSI chưa oversold ({rsi:.0f})"
            else:
                result['action'] = 'WAIT'
                result['reason'] = f"⏳ Giá trong OTE, chờ nến xác nhận"
        elif current_price > ote_high:
            result['action'] = 'WAIT'
            result['reason'] = f"⏳ Chờ giá hồi về OTE (${ote_low:,.0f} - ${ote_high:,.0f})"
        elif current_price < ote_low:
            result['action'] = 'INVALIDATED'
            result['reason'] = f"⚠️ Giá phá vỡ dưới OTE - Setup không hợp lệ"
        else:
            result['action'] = 'WAIT'
            result['reason'] = f"⏳ Chờ giá về OTE"
            
    else:  # Impulse DOWN -> SHORT
        entry_price = fib_618
        sl_price = fib_786 + atr * 1.5
        tp1_price = lowest['price']
        tp2_price = lowest['price'] - fib_range * 0.272
        
        if price_in_ote or price_near_ote:
            if has_bearish_confirmation:
                if rsi_ok_short:
                    result['valid'] = True
                    result['action'] = 'SHORT'
                    result['reason'] = f"✅ Giá trong OTE + {candle_pattern['pattern']} + RSI OK"
                else:
                    result['action'] = 'WAIT'
                    result['reason'] = f"⏳ Có nến {candle_pattern['pattern']} nhưng RSI chưa overbought ({rsi:.0f})"
            else:
                result['action'] = 'WAIT'
                result['reason'] = f"⏳ Giá trong OTE, chờ nến xác nhận"
        elif current_price < ote_low:
            result['action'] = 'WAIT'
            result['reason'] = f"⏳ Chờ giá hồi lên OTE (${ote_low:,.0f} - ${ote_high:,.0f})"
        elif current_price > ote_high:
            result['action'] = 'INVALIDATED'
            result['reason'] = f"⚠️ Giá phá vỡ trên OTE - Setup không hợp lệ"
        else:
            result['action'] = 'WAIT'
            result['reason'] = f"⏳ Chờ giá về OTE"
    
    result['entry'] = entry_price
    result['sl'] = sl_price
    result['tp1'] = tp1_price
    result['tp2'] = tp2_price
    
    # Debug output
    ema50_str = f"${ema50:,.0f}" if ema50 else "N/A"
    ema200_str = f"${ema200:,.0f}" if ema200 else "N/A"
    print(f"\n  🌊 Elliott Wave Fibo Analysis (3-Module):")
    print(f"    EMA Trend: {ema_trend} | EMA50: {ema50_str} | EMA200: {ema200_str}")
    print(f"    Impulse: {impulse_type} | Trade: {trend}")
    print(f"    Swing High: ${highest['price']:,.0f} | Swing Low: ${lowest['price']:,.0f}")
    print(f"    Fibo Range: ${fib_range:,.0f}")
    print(f"    OTE Zone: ${ote_low:,.0f} - ${ote_high:,.0f}")
    print(f"    Current Price: ${current_price:,.0f} | In OTE: {price_in_ote}")
    print(f"    RSI: {rsi:.1f} | Candle: {candle_pattern['pattern']} ({candle_pattern['type']})")
    print(f"    -> Action: {result['action']}")
    print(f"    -> Reason: {result['reason']}")
    
    return result


