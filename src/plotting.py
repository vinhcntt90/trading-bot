import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from .config import Config
from .indicators import calculate_sr_channels

def plot_plotly_chart(df, timeframes_bias, pivots, poc_data, plan):
    """
    Generate Professional Crypto Chart using Plotly
    """
    print("[*] Generating Plotly Chart...")
    
    # Create Figure
    fig = make_subplots(rows=1, cols=1)
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='BTC/USDT',
        increasing_line_color='#26a69a', # Teal
        decreasing_line_color='#ef5350'  # Red
    ))
    
    # 2. EMAs
    colors = {'EMA9': '#2962ff', 'EMA21': '#ff6d00', 'EMA50': '#ffab00', 'EMA200': '#e91e63'}
    for ema, color in colors.items():
        if ema in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[ema], 
                mode='lines', name=ema,
                line=dict(color=color, width=1)
            ))
            
    # 3. Bollinger Bands
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Upper'],
            mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df['BB_Lower'],
            mode='lines', line=dict(width=0), fill='tonexty',
            fillcolor='rgba(0, 188, 212, 0.05)',
            showlegend=False, hoverinfo='skip'
        ))

    # 4. Support/Resistance Channels
    sr_data = calculate_sr_channels(df)
    if sr_data and sr_data.get('channels'):
        for ch in sr_data['channels']:
            color = 'rgba(38, 166, 154, 0.2)' if ch['type'] == 'support' else 'rgba(239, 83, 80, 0.2)'
            fig.add_shape(type="rect",
                x0=df.index[0], y0=ch['lo'], x1=df.index[-1], y1=ch['hi'],
                fillcolor=color, line_width=0, layer="below"
            )
            fig.add_annotation(
                x=df.index[-10], y=ch['hi'],
                text=f"{ch['type'].upper()}",
                showarrow=False, font=dict(color=color.replace('0.2', '1.0'), size=10)
            )

    # 5. Trading Plan / Golden Pocket
    if 'golden_pocket_strategy' in plan:
        gp = plan['golden_pocket_strategy']
        if gp.get('valid'):
            # Lines
            fig.add_hline(y=gp['entry'], line_dash="solid", line_color="yellow", annotation_text="ENTRY")
            fig.add_hline(y=gp['sl'], line_dash="dash", line_color="red", annotation_text="SL")
            fig.add_hline(y=gp['tp1'], line_dash="dot", line_color="green", annotation_text="TP1")
            
            # Zone
            gp_zone = gp['golden_pocket']
            color = 'rgba(255, 215, 0, 0.15)' # Gold
            fig.add_shape(type="rect",
                x0=df.index[-50], y0=gp_zone['low'], x1=df.index[-1], y1=gp_zone['high'],
                fillcolor=color, line_width=0
            )

    # 6. Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        title=dict(text=f"BTC/USDT - {plan.get('direction', 'NEUTRAL')} SETUP", font=dict(size=20, color="white")),
        xaxis=dict(
            showgrid=True, gridcolor='#2a2e39', gridwidth=1,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#2a2e39', gridwidth=1,
            side='right'
        ),
        margin=dict(l=10, r=50, t=50, b=10),
        height=1000,
        width=1600,
        showlegend=False
    )
    
    # Watermark
    fig.add_annotation(
        text="ANTIGRAVITY",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=80, color="rgba(255,255,255,0.05)")
    )

    # Save
    output_path = os.path.join(Config.ARTIFACTS_DIR, 'btc_chart_pro.png')
    try:
        fig.write_image(output_path, scale=2)
        print(f"[+] Plotly Chart Saved: {output_path}")
        return output_path
    except Exception as e:
        print(f"[!] Error saving Plotly chart: {e}")
        return None

def plot_fibo_chart(df, gp_result, ew_result=None):
    """
    Generate Fibonacci Chart for Golden Pocket Strategy using Plotly
    Now also supports Elliott Wave overlay
    """
    if not gp_result or gp_result.get('trend') == 'NEUTRAL':
        return None
        
    print("[*] Generating Fibonacci Chart...")
    
    # Use last 120 candles
    chart_df = df.tail(120).copy()
    
    fig = go.Figure()
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=chart_df.index,
        open=chart_df['open'], high=chart_df['high'],
        low=chart_df['low'], close=chart_df['close'],
        name='BTC',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # 2. Fibo Levels (from Golden Pocket)
    swing_high = gp_result['swing_high']
    swing_low = gp_result['swing_low']
    trend = gp_result['trend']
    fib_range = swing_high - swing_low
    
    levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618]
    
    fib_prices = {}
    for ratio in levels:
        if trend == 'BULLISH':
            price = swing_high - fib_range * ratio if ratio <= 1.0 else swing_high + fib_range * (ratio - 1.0)
        else:
            price = swing_low + fib_range * ratio if ratio <= 1.0 else swing_low - fib_range * (ratio - 1.0)
        fib_prices[ratio] = price
        
        # Draw Line
        color = '#00bcd4' if ratio in [0.5, 0.618] else '#787b86'
        width = 2 if ratio in [0.5, 0.618] else 1
        dash = 'solid' if ratio in [0.5, 0.618] else 'dash'
        
        fig.add_hline(y=price, line_color=color, line_width=width, line_dash=dash)
        
        # Label
        fig.add_annotation(
            x=chart_df.index[5], y=price,
            text=f"{ratio} ({price:,.2f})",
            showarrow=False, xanchor='left', yanchor='bottom',
            font=dict(color=color, size=10)
        )
        
    # 3. Golden Pocket Zone
    p05 = fib_prices[0.5]
    p618 = fib_prices[0.618]
    fig.add_shape(type="rect",
        x0=chart_df.index[0], y0=min(p05, p618),
        x1=chart_df.index[-1], y1=max(p05, p618),
        fillcolor="rgba(0, 188, 212, 0.15)", line_width=0, layer="below"
    )
    
    # 4. Entry/SL/TP Markers
    if gp_result.get('valid'):
        entry = gp_result.get('entry', 0)
        sl = gp_result.get('sl', 0)
        tp1 = gp_result.get('tp1', 0)
        
        # Entry Line
        fig.add_hline(y=entry, line_color="green", line_dash="solid", annotation_text=f"ENTRY: {entry:,.0f}")
        fig.add_hline(y=sl, line_color="red", line_dash="dash", annotation_text=f"SL: {sl:,.0f}")
        fig.add_hline(y=tp1, line_color="lime", line_dash="dot", annotation_text=f"TP1: {tp1:,.0f}")

    # ===== 5. ELLIOTT WAVE OVERLAY (NEW) =====
    if ew_result and ew_result.get('impulse_type'):
        ew_swing_high = ew_result['swing_high']
        ew_swing_low = ew_result['swing_low']
        ew_swing_high_idx = ew_result.get('swing_high_idx', 0)
        ew_swing_low_idx = ew_result.get('swing_low_idx', 0)
        ote_zone = ew_result.get('ote_zone', {})
        
        # Convert global indices to chart_df indices (approximate)
        chart_start_idx = len(df) - 120
        
        # OTE Zone (0.618 - 0.786) - Different color from Golden Pocket
        if ote_zone.get('low') and ote_zone.get('high'):
            fig.add_shape(type="rect",
                x0=chart_df.index[0], y0=ote_zone['low'],
                x1=chart_df.index[-1], y1=ote_zone['high'],
                fillcolor="rgba(255, 152, 0, 0.2)",  # Orange
                line_width=1, line_color="rgba(255, 152, 0, 0.8)",
                layer="below"
            )
            # OTE Label
            fig.add_annotation(
                x=chart_df.index[-5], y=(ote_zone['low'] + ote_zone['high']) / 2,
                text="🌊 OTE ZONE",
                showarrow=False, xanchor='right',
                font=dict(color='#ff9800', size=12, family='Arial Black'),
                bgcolor='rgba(0,0,0,0.5)'
            )
        
        # Swing High Marker
        high_chart_idx = max(0, min(ew_swing_high_idx - chart_start_idx, len(chart_df) - 1))
        if 0 <= high_chart_idx < len(chart_df):
            fig.add_trace(go.Scatter(
                x=[chart_df.index[high_chart_idx]],
                y=[ew_swing_high],
                mode='markers+text',
                marker=dict(symbol='triangle-down', size=15, color='#f44336'),
                text=['🔻 HIGH'],
                textposition='top center',
                textfont=dict(color='#f44336', size=10),
                showlegend=False
            ))
        
        # Swing Low Marker
        low_chart_idx = max(0, min(ew_swing_low_idx - chart_start_idx, len(chart_df) - 1))
        if 0 <= low_chart_idx < len(chart_df):
            fig.add_trace(go.Scatter(
                x=[chart_df.index[low_chart_idx]],
                y=[ew_swing_low],
                mode='markers+text',
                marker=dict(symbol='triangle-up', size=15, color='#4caf50'),
                text=['🔺 LOW'],
                textposition='bottom center',
                textfont=dict(color='#4caf50', size=10),
                showlegend=False
            ))
        
        # Impulse Wave Line (connects Low to High or High to Low)
        if ew_result['impulse_type'] == 'UP':
            # Low -> High
            fig.add_trace(go.Scatter(
                x=[chart_df.index[low_chart_idx], chart_df.index[high_chart_idx]],
                y=[ew_swing_low, ew_swing_high],
                mode='lines',
                line=dict(color='#2196f3', width=2, dash='dot'),
                name='Impulse Wave',
                showlegend=False
            ))
        else:
            # High -> Low
            fig.add_trace(go.Scatter(
                x=[chart_df.index[high_chart_idx], chart_df.index[low_chart_idx]],
                y=[ew_swing_high, ew_swing_low],
                mode='lines',
                line=dict(color='#ff5722', width=2, dash='dot'),
                name='Impulse Wave',
                showlegend=False
            ))
        
        # Elliott Wave Action Label
        ew_action = ew_result.get('action', 'WAIT')
        ew_color = '#4caf50' if ew_action == 'LONG' else ('#f44336' if ew_action == 'SHORT' else '#ffffff')
        fig.add_annotation(
            x=chart_df.index[10], y=chart_df['high'].max(),
            text=f"🌊 Elliott: {ew_action}",
            showarrow=False, xanchor='left', yanchor='top',
            font=dict(color=ew_color, size=14, family='Arial Black'),
            bgcolor='rgba(0,0,0,0.7)', borderpad=4
        )

    # Layout
    action_text = f"{gp_result['action']}"
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        title=dict(text=f"Fibonacci Retracement - {action_text}", font=dict(size=18, color="white")),
        xaxis=dict(showgrid=False, visible=False, rangeslider=dict(visible=False)),
        yaxis=dict(showgrid=True, gridcolor='#2a2e39', side='right'),
        margin=dict(l=10, r=60, t=50, b=10),
        height=900, width=1600, showlegend=False
    )
    
    output_path = os.path.join(Config.ARTIFACTS_DIR, 'btc_fibo_chart.png')
    try:
        fig.write_image(output_path, scale=2)
        print(f"[+] Saved Fibo Chart: {output_path}")
        return output_path
    except Exception as e:
        print(f"[!] Error saving Fibo chart: {e}")
        return None

