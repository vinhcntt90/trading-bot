import pandas as pd
import numpy as np

# ============================================
# CONFIGURATION
# ============================================
SRCHANNEL_CONFIG = {
    'pivot_period': 10,
    'channel_width_pct': 5,
    'min_strength': 1,
    'max_channels': 6,
    'loopback': 100,
}

LUXALGO_CONFIG = {
    'length': 8,
    'mult': 50,
    'atr_length': 4,
}

def detect_pivot_points(df, period=10):
    """
    Detect Pivot High and Pivot Low points
    """
    pivots = []
    
    for i in range(period, len(df) - period):
        # Check Pivot High
        is_pivot_high = True
        high_val = df['high'].iloc[i]
        for j in range(i - period, i + period + 1):
            if j != i and df['high'].iloc[j] >= high_val:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            pivots.append({
                'type': 'high',
                'price': high_val,
                'index': i,
                'bar_index': df.index[i]
            })
        
        # Check Pivot Low
        is_pivot_low = True
        low_val = df['low'].iloc[i]
        for j in range(i - period, i + period + 1):
            if j != i and df['low'].iloc[j] <= low_val:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            pivots.append({
                'type': 'low',
                'price': low_val,
                'index': i,
                'bar_index': df.index[i]
            })
    
    return pivots

def calculate_sr_channels(df, config=None):
    """
    Calculate Support/Resistance Channels
    """
    if config is None:
        config = SRCHANNEL_CONFIG
    
    period = config['pivot_period']
    channel_width_pct = config['channel_width_pct']
    min_strength = config['min_strength']
    max_channels = config['max_channels']
    loopback = min(config['loopback'], len(df) - period * 2 - 1)
    
    if loopback < 20:
        return None
    
    highest_300 = df['high'].iloc[-min(300, len(df)):].max()
    lowest_300 = df['low'].iloc[-min(300, len(df)):].min()
    channel_width = (highest_300 - lowest_300) * channel_width_pct / 100
    
    pivots = detect_pivot_points(df, period)
    current_bar = len(df) - 1
    recent_pivots = [p for p in pivots if current_bar - p['index'] <= loopback]
    
    if len(recent_pivots) == 0:
        return None
    
    def get_channel_for_pivot(pivot_idx, all_pivots):
        base_price = all_pivots[pivot_idx]['price']
        lo = base_price
        hi = base_price
        included = []
        
        for i, p in enumerate(all_pivots):
            price = p['price']
            width = price - lo if price > hi else hi - price if price < lo else 0
            
            if width <= channel_width:
                if price < lo:
                    lo = price
                elif price > hi:
                    hi = price
                included.append(i)
        
        return {'hi': hi, 'lo': lo, 'pivot_count': len(included), 'pivots': included}
    
    channel_candidates = []
    for i in range(len(recent_pivots)):
        ch = get_channel_for_pivot(i, recent_pivots)
        strength = ch['pivot_count'] * 20
        
        touches = 0
        for j in range(max(0, len(df) - loopback), len(df)):
            h = df['high'].iloc[j]
            l = df['low'].iloc[j]
            if (h <= ch['hi'] and h >= ch['lo']) or (l <= ch['hi'] and l >= ch['lo']):
                touches += 1
        
        strength += touches
        ch['strength'] = strength
        channel_candidates.append(ch)
    
    # Sort and filter
    used_pivots = set()
    final_channels = []
    channel_candidates.sort(key=lambda x: x['strength'], reverse=True)
    
    for ch in channel_candidates:
        if ch['strength'] < min_strength * 20:
            continue
        
        overlap = False
        for p_idx in ch['pivots']:
            if p_idx in used_pivots:
                overlap = True
                break
        
        if not overlap:
            for p_idx in ch['pivots']:
                used_pivots.add(p_idx)
            final_channels.append(ch)
            if len(final_channels) >= max_channels:
                break
    
    current_price = df['close'].iloc[-1]
    for ch in final_channels:
        if ch['hi'] < current_price and ch['lo'] < current_price:
            ch['type'] = 'support'
            ch['color'] = '#26a69a'
        elif ch['hi'] > current_price and ch['lo'] > current_price:
            ch['type'] = 'resistance'
            ch['color'] = '#ef5350'
        else:
            ch['type'] = 'active'
            ch['color'] = '#808080'
            
    # Check broken
    prev_close = df['close'].iloc[-2] if len(df) > 1 else current_price
    broken_resistance = False
    broken_support = False
    
    for ch in final_channels:
        if prev_close <= ch['hi'] and current_price > ch['hi']:
            broken_resistance = True
        if prev_close >= ch['lo'] and current_price < ch['lo']:
            broken_support = True
            
    return {
        'channels': final_channels,
        'broken_resistance': broken_resistance,
        'broken_support': broken_support,
    }

def calculate_luxalgo_sr_dynamic(df, config=None):
    """
    LuxAlgo-style Support & Resistance Dynamic
    """
    if config is None:
        config = LUXALGO_CONFIG
    
    length = config['length']
    mult = config['mult'] / 100
    atr_length = config['atr_length']
    
    if len(df) < max(length * 2, atr_length) + 10:
        return None
    
    # Simple ATR
    df_calc = df.copy()
    df_calc['tr'] = np.maximum(
        df_calc['high'] - df_calc['low'],
        np.maximum(
            abs(df_calc['high'] - df_calc['close'].shift(1)),
            abs(df_calc['low'] - df_calc['close'].shift(1))
        )
    )
    df_calc['atr'] = df_calc['tr'].rolling(window=atr_length).mean()
    
    current_atr = df_calc['atr'].iloc[-1]
    zone_width = current_atr * mult
    
    zones = []
    
    for i in range(length, len(df) - length):
        # Resistance
        is_pivot_high = True
        high_val = df['high'].iloc[i]
        for j in range(i - length, i + length + 1):
            if j != i and df['high'].iloc[j] >= high_val:
                is_pivot_high = False
                break
        
        if is_pivot_high:
            zones.append({
                'type': 'resistance',
                'top': high_val,
                'bottom': high_val - zone_width,
                'mid': high_val - zone_width/2,
                'color': '#ef5350'
            })
            
        # Support
        is_pivot_low = True
        low_val = df['low'].iloc[i]
        for j in range(i - length, i + length + 1):
            if j != i and df['low'].iloc[j] <= low_val:
                is_pivot_low = False
                break
        
        if is_pivot_low:
            zones.append({
                'type': 'support',
                'top': low_val + zone_width,
                'bottom': low_val,
                'mid': low_val + zone_width/2,
                'color': '#26a69a'
            })
            
    # Return last valid zones
    return zones

def calculate_indicators(df):
    """Calculate all indicators including multi-TF analysis indicators"""
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
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Moving Averages
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['close'].rolling(window=20).mean()
    df['BB_Std'] = df['close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Volume
    df['Vol_MA20'] = df['volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['volume'] / df['Vol_MA20']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df['ATR'] = tr.rolling(window=14).mean()
    
    return df


# ============================================
# LINEAR REGRESSION OSCILLATOR [CHARTPRIME]
# ============================================
def calculate_linear_regression_oscillator(df, length=20):
    """
    Calculate Linear Regression Oscillator based on ChartPrime indicator
    """
    from .config import CHARTPRIME_CONFIG
    
    if len(df) < length:
        return None
    
    df_calc = df.copy()
    
    def linear_regression(series, length):
        """Calculate linear regression value"""
        result = []
        for i in range(len(series)):
            if i < length - 1:
                result.append(np.nan)
            else:
                y = series.iloc[i-length+1:i+1].values
                x = np.arange(length)
                coeffs = np.polyfit(x, y, 1)
                predicted = coeffs[0] * (length - 1) + coeffs[1]
                result.append(predicted)
        return pd.Series(result, index=series.index)
    
    df_calc['linreg'] = linear_regression(df_calc['close'], length)
    df_calc['stdev'] = df_calc['close'].rolling(window=length).std()
    df_calc['oscillator'] = (df_calc['close'] - df_calc['linreg']) / df_calc['stdev']
    
    current_osc = df_calc['oscillator'].iloc[-1]
    prev_osc = df_calc['oscillator'].iloc[-2] if len(df_calc) > 1 else 0
    prev_prev_osc = df_calc['oscillator'].iloc[-3] if len(df_calc) > 2 else 0
    
    upper_threshold = CHARTPRIME_CONFIG['lro_upper_threshold']
    lower_threshold = CHARTPRIME_CONFIG['lro_lower_threshold']
    
    if current_osc > upper_threshold:
        status = 'OVERBOUGHT'
    elif current_osc < lower_threshold:
        status = 'OVERSOLD'
    else:
        status = 'NEUTRAL'
    
    signal = None
    if prev_osc < 0 and current_osc >= 0:
        signal = 'BULLISH_CROSS'
    elif prev_osc > 0 and current_osc <= 0:
        signal = 'BEARISH_CROSS'
    elif prev_osc <= lower_threshold and current_osc > lower_threshold:
        signal = 'BULLISH_REVERSION'
    elif prev_osc >= upper_threshold and current_osc < upper_threshold:
        signal = 'BEARISH_REVERSION'
    
    if current_osc > prev_osc > prev_prev_osc:
        momentum = 'RISING'
    elif current_osc < prev_osc < prev_prev_osc:
        momentum = 'FALLING'
    else:
        momentum = 'NEUTRAL'
    
    return {
        'oscillator': current_osc,
        'prev_oscillator': prev_osc,
        'linreg': df_calc['linreg'].iloc[-1],
        'status': status,
        'signal': signal,
        'momentum': momentum,
        'upper_threshold': upper_threshold,
        'lower_threshold': lower_threshold,
        'invalidation_level': df_calc['linreg'].iloc[-1],
        'plot_data': df_calc,
    }


def calculate_trend_levels(df, length=10):
    """Wrapper for backward compatibility - now uses SRChannel algorithm"""
    sr_data = calculate_sr_channels(df, {
        'pivot_period': length,
        'channel_width_pct': 5,
        'min_strength': 1,
        'max_channels': 6,
        'loopback': 290,
    })
    
    if sr_data is None:
        return None
    
    current_price = df['close'].iloc[-1]
    channels = sr_data.get('channels', [])
    
    resistances = [ch for ch in channels if ch['type'] == 'resistance']
    supports = [ch for ch in channels if ch['type'] == 'support']
    
    trend_high = resistances[0]['hi'] if resistances else df['high'].iloc[-1]
    trend_low = supports[0]['lo'] if supports else df['low'].iloc[-1]
    trend_mid = (trend_high + trend_low) / 2
    
    return {
        'trend_high': trend_high,
        'trend_low': trend_low,
        'trend_mid': trend_mid,
        'trend_direction': 'UP' if current_price > trend_mid else 'DOWN',
        'reversion_signal': sr_data.get('reversion_signal'),
        'near_high': abs(current_price - trend_high) / current_price < 0.005,
        'near_mid': abs(current_price - trend_mid) / current_price < 0.005,
        'near_low': abs(current_price - trend_low) / current_price < 0.005,
        'invalidation_level': trend_mid,
        'trend_strength': 0,
        'broken_resistance': sr_data.get('broken_resistance', False),
        'broken_support': sr_data.get('broken_support', False),
    }


def get_chartprime_entry_signals(df_15m):
    """
    Combine Trend Levels and Linear Regression Oscillator for entry signals
    """
    from .config import CHARTPRIME_CONFIG
    
    length_tl = CHARTPRIME_CONFIG['trend_levels_length']
    length_lro = CHARTPRIME_CONFIG['lro_length']
    
    trend_levels = calculate_trend_levels(df_15m, length_tl)
    lro = calculate_linear_regression_oscillator(df_15m, length_lro)
    
    if trend_levels is None or lro is None:
        return None
    
    score = 0
    signals = []
    current_price = df_15m['close'].iloc[-1]
    tl_score = 0
    
    # Trend Levels scoring
    if trend_levels.get('reversion_signal') == 'BULLISH_REVERSION':
        tl_score += 3
        signals.append(('TL', 'Bullish Reversion', 'BULL'))
    elif trend_levels.get('reversion_signal') == 'BEARISH_REVERSION':
        tl_score -= 3
        signals.append(('TL', 'Bearish Reversion', 'BEAR'))
    
    if trend_levels['trend_direction'] == 'UP':
        if trend_levels['near_low']:
            tl_score += 2
            signals.append(('TL', f"At Support ${trend_levels['trend_low']:,.0f}", 'BULL'))
    else:
        if trend_levels['near_high']:
            tl_score -= 2
            signals.append(('TL', f"At Resistance ${trend_levels['trend_high']:,.0f}", 'BEAR'))
    
    # LRO scoring
    lro_score = 0
    if lro['status'] == 'OVERSOLD' and lro['momentum'] == 'RISING':
        lro_score += 3
        signals.append(('LRO', f"Oversold + Rising ({lro['oscillator']:.2f})", 'BULL'))
    elif lro['status'] == 'OVERBOUGHT' and lro['momentum'] == 'FALLING':
        lro_score -= 3
        signals.append(('LRO', f"Overbought + Falling ({lro['oscillator']:.2f})", 'BEAR'))
    elif lro['signal'] == 'BULLISH_REVERSION':
        lro_score += 2
        signals.append(('LRO', f"Left Oversold ({lro['oscillator']:.2f})", 'BULL'))
    elif lro['signal'] == 'BEARISH_REVERSION':
        lro_score -= 2
        signals.append(('LRO', f"Left Overbought ({lro['oscillator']:.2f})", 'BEAR'))
    
    score = tl_score + lro_score
    
    if score >= 4:
        entry_signal = 'STRONG_BUY'
    elif score >= 2:
        entry_signal = 'BUY'
    elif score <= -4:
        entry_signal = 'STRONG_SELL'
    elif score <= -2:
        entry_signal = 'SELL'
    else:
        entry_signal = 'WAIT'
    
    if score > 0:
        entry_price = current_price
        sl_price = trend_levels['trend_low'] * 0.99
        tp1_price = trend_levels['trend_mid']
        tp2_price = trend_levels['trend_high']
        tp3_price = tp2_price + (tp2_price - tp1_price)
    else:
        entry_price = current_price
        sl_price = trend_levels['trend_high'] * 1.01
        tp1_price = trend_levels['trend_mid']
        tp2_price = trend_levels['trend_low']
        tp3_price = tp2_price - (tp1_price - tp2_price)
    
    return {
        'score': score,
        'tl_score': tl_score,
        'lro_score': lro_score,
        'entry_signal': entry_signal,
        'signals': signals,
        'trend_levels': trend_levels,
        'lro': lro,
        'entry': {
            'price': entry_price,
            'sl': sl_price,
            'tp1': tp1_price,
            'tp2': tp2_price,
            'tp3': tp3_price,
        },
    }


def find_pivots(df, left=5, right=5):
    """Find pivot highs and lows"""
    pivots = {'highs': [], 'lows': []}
    
    for i in range(left, len(df) - right):
        is_high = True
        is_low = True
        
        for j in range(i - left, i + right + 1):
            if j != i:
                if df['high'].iloc[j] >= df['high'].iloc[i]:
                    is_high = False
                if df['low'].iloc[j] <= df['low'].iloc[i]:
                    is_low = False
        
        if is_high:
            pivots['highs'].append({'idx': i, 'price': df['high'].iloc[i]})
        if is_low:
            pivots['lows'].append({'idx': i, 'price': df['low'].iloc[i]})
    
    return pivots


def get_tf_bias(df):
    """Get bias based on EMA alignment"""
    if 'EMA50' not in df.columns:
        df = calculate_indicators(df)
    
    current = df['close'].iloc[-1]
    ema50 = df['EMA50'].iloc[-1]
    ema200 = df['EMA200'].iloc[-1]
    
    if current > ema50 > ema200:
        return 'BULLISH'
    elif current < ema50 < ema200:
        return 'BEARISH'
    else:
        return 'NEUTRAL'

