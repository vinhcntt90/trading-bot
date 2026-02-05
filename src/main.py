import sys
from datetime import datetime
from .config import Config, CHARTPRIME_CONFIG
from .data import get_btc_data, get_derivatives_data, get_derivatives_signal
from .indicators import (
    calculate_indicators, calculate_sr_channels, 
    calculate_linear_regression_oscillator, get_chartprime_entry_signals,
    find_pivots, get_tf_bias
)
from .analysis import (
    calculate_trading_plan, calculate_pivot_points, calculate_poc,
    get_fibonacci_levels, analyze_timeframe_detailed, 
    calculate_golden_pocket_strategy, calculate_gann_fan, get_gann_signal,
    get_moon_phase_simple, get_lunar_trading_signal, get_mercury_retrograde,
    detect_wyckoff_phase, analyze_smc,
    calculate_elliott_wave_fibo  # NEW
)
from .plotting import plot_plotly_chart, plot_fibo_chart
from .reporting import (
    print_trading_plan, print_multi_tf_analysis, 
    print_golden_pocket_strategy, print_smc_analysis, print_derivatives_info
)
from .notifications import send_telegram_photo, send_telegram_media_group, create_telegram_caption


def main():
    print("\n[*] BTC/USDT TradingView Pro Chart")
    print("    + Pivot Points (PP, R1-R3, S1-S3)")
    print("    + POC (Point of Control)")
    print("    + Multi-Timeframe Analysis (15m, 1H, 4H, 1D)")
    print("    + MACD, MA20/50, BB, Volume")
    print("    + Fibonacci Levels")
    print("    + Trading Plan (Long/Short)")
    print("=" * 50)
    
    # 1. Fetch 15m Data
    print("[*] Fetching data (15m)...")
    df = get_btc_data('15m', 150)
    if df is None:
        print("[!] Failed to fetch data. Exiting.")
        return None
    df = calculate_indicators(df)
    
    # 2. Fetch Daily data for Pivot Points
    print("[*] Fetching Daily data for Pivots...")
    df_daily = get_btc_data('1d', 60)
    
    print("[*] Calculating Pivot Points (Daily)...")
    pivots = calculate_pivot_points(df_daily)
    for name, price in pivots.items():
        print(f"    {name}: ${price:,.0f}")
    
    # 3. Calculate POC
    print("[*] Calculating POC...")
    poc_data = calculate_poc(df)
    print(f"    POC: ${poc_data['poc']:,.0f}")
    print(f"    VA High: ${poc_data['va_high']:,.0f}")
    print(f"    VA Low: ${poc_data['va_low']:,.0f}")
    
    # 4. Multi-Timeframe Detailed Analysis
    print("[*] Multi-Timeframe Detailed Analysis...")
    timeframes_config = {
        '15m': ('15m', 100),
        '1H': ('1h', 100),
        '4H': ('4h', 100),
        '1D': ('1d', 60)
    }
    
    analyses = {}
    timeframes_bias = {}
    
    for tf_name, (interval, limit) in timeframes_config.items():
        print(f"    Analyzing {tf_name}...")
        tf_df = get_btc_data(interval, limit)
        if tf_df is not None:
            analysis = analyze_timeframe_detailed(tf_df, tf_name)
            if analysis:
                analyses[tf_name] = analysis
                timeframes_bias[tf_name] = analysis['bias']
            else:
                tf_df = calculate_indicators(tf_df)
                timeframes_bias[tf_name] = get_tf_bias(tf_df)
    
    overall_bias = print_multi_tf_analysis(analyses)
    
    # 5. Gann Fan Analysis
    print("\n[*] Gann Fan Analysis...")
    gann_up, gann_up_pos, gann_up_price = calculate_gann_fan(df, 'low')
    gann_down, gann_down_pos, gann_down_price = calculate_gann_fan(df, 'high')
    gann_signal, gann_desc = get_gann_signal(df['close'].iloc[-1], gann_up, gann_up_price)
    
    print(f"    Pivot Low: ${gann_up_price:,.0f} (Position: {gann_up_pos})")
    print(f"    Pivot High: ${gann_down_price:,.0f} (Position: {gann_down_pos})")
    print(f"    Gann Signal: {gann_signal} - {gann_desc}")
    
    # 6. Lunar / Astrology Analysis
    print("\n[*] Lunar / Astrology Analysis...")
    moon_phase = get_moon_phase_simple()
    lunar_signal = get_lunar_trading_signal(moon_phase)
    mercury = get_mercury_retrograde()
    
    print(f"    Moon Phase: {moon_phase['phase_name']} ({moon_phase['percentage']:.1f}%)")
    print(f"    Lunar Signal: {lunar_signal['signal']} - {lunar_signal['description']}")
    print(f"    Lunar Sentiment: {lunar_signal['sentiment']}")
    print(f"    Mercury: {'RETROGRADE' if mercury['is_retrograde'] else 'Direct'}")
    if mercury['is_retrograde']:
        print(f"      Days remaining: {mercury['days_remaining']}")
    
    # 7. Derivatives Data
    print("\n[*] Fetching Derivatives Data...")
    derivatives_data = get_derivatives_data()
    
    if derivatives_data and derivatives_data.get('funding_rate_pct') is not None:
        fr_pct = derivatives_data['funding_rate_pct']
        print(f"    Funding Rate: {fr_pct:.4f}%")
        
        if derivatives_data.get('open_interest'):
            oi = derivatives_data['open_interest']
            oi_usd = derivatives_data.get('open_interest_usd', 0)
            print(f"    Open Interest: {oi:,.0f} BTC (${oi_usd/1e9:.2f}B)")
        
        if derivatives_data.get('ls_ratio'):
            ls = derivatives_data['ls_ratio']
            long_pct = derivatives_data.get('long_ratio', 0) * 100
            print(f"    L/S Ratio: {ls:.2f} ({long_pct:.0f}% Long / {100-long_pct:.0f}% Short)")
    else:
        print("    [!] Could not fetch derivatives data")
    
    # 8. SMC Analysis
    print("\n[*] SMC Analysis (Wyckoff, FVG, OB, Liquidity)...")
    smc_data = analyze_smc(df)
    
    wyckoff = smc_data.get('wyckoff', {})
    print(f"    Wyckoff: {wyckoff.get('phase', 'N/A')} ({wyckoff.get('confidence', 0)}%)")
    print(f"      -> {wyckoff.get('description', '')}")
    
    nearest_fvg = smc_data.get('nearest_fvg')
    if nearest_fvg:
        print(f"    FVG: {nearest_fvg['type'].title()} at ${nearest_fvg['mid']:,.0f}")
    
    nearest_ob = smc_data.get('nearest_ob')
    if nearest_ob:
        print(f"    Order Block: {nearest_ob['type'].title()} OB at ${nearest_ob['mid']:,.0f}")
    
    liquidity = smc_data.get('liquidity', {})
    if liquidity.get('nearest_bsl'):
        print(f"    BSL: ${liquidity['nearest_bsl']['level']:,.0f}")
    if liquidity.get('nearest_ssl'):
        print(f"    SSL: ${liquidity['nearest_ssl']['level']:,.0f}")
    
    print(f"    SMC Score: {smc_data.get('score', 0):+d}")
    
    # 9. ChartPrime Analysis
    print("\n[*] ChartPrime Analysis (15m Trend Levels + LRO)...")
    chartprime_data = get_chartprime_entry_signals(df)
    
    if chartprime_data:
        tl = chartprime_data['trend_levels']
        lro = chartprime_data['lro']
        
        print(f"    === Trend Levels ===")
        print(f"    Trend High: ${tl['trend_high']:,.0f}")
        print(f"    Trend Mid:  ${tl['trend_mid']:,.0f}")
        print(f"    Trend Low:  ${tl['trend_low']:,.0f}")
        print(f"    Direction:  {tl['trend_direction']}")
        
        print(f"\n    === LRO ===")
        print(f"    Oscillator: {lro['oscillator']:.2f} ({lro['status']})")
        print(f"    Momentum:   {lro['momentum']}")
        
        print(f"\n    === Combined ===")
        print(f"    Total Score: {chartprime_data['score']:+d}")
        print(f"    Entry Signal: {chartprime_data['entry_signal']}")
    else:
        print("    [!] Could not calculate ChartPrime data")
        chartprime_data = None
    
    # 10. Calculate Trading Plan
    print("\n[*] Calculating trading plan...")
    gann_data = {'signal': gann_signal, 'desc': gann_desc, 'lines_up': gann_up, 'lines_down': gann_down}
    lunar_data = {'phase': moon_phase, 'signal': lunar_signal, 'mercury': mercury}
    plan = calculate_trading_plan(df, pivots, poc_data, timeframes_bias, gann_data, lunar_data, derivatives_data, smc_data, chartprime_data)
    
    # Store additional data for charts
    plan['gann'] = gann_data
    plan['lunar'] = lunar_data
    plan['cp_data'] = chartprime_data
    
    # 11. Golden Pocket Strategy
    print("\n[*] Calculating Golden Pocket Strategy (Parallel)...")
    gp_strategy = calculate_golden_pocket_strategy(df, smc_data, pivots)
    plan['golden_pocket_strategy'] = gp_strategy
    
    print("\n" + "=" * 60)
    print("  🏆 GOLDEN POCKET STRATEGY (15m)")
    print("=" * 60)
    print(f"  Trend          : {gp_strategy['trend']}")
    print(f"  Swing High     : ${gp_strategy['swing_high']:,.0f}")
    print(f"  Swing Low      : ${gp_strategy['swing_low']:,.0f}")
    gp = gp_strategy['golden_pocket']
    print(f"  Golden Pocket  : ${gp['low']:,.0f} - ${gp['high']:,.0f}")
    if gp_strategy.get('atr'):
        print(f"  ATR Buffer     : ${gp_strategy['atr']:,.0f}")
    
    print("-" * 60)
    action_emoji = "🟢" if gp_strategy['action'] == 'LONG' else ("🔴" if gp_strategy['action'] == 'SHORT' else "⚪")
    print(f"  {action_emoji} Action: {gp_strategy['action']}")
    if gp_strategy['valid']:
        print(f"  Entry1 (50%)   : ${gp_strategy.get('entry1', gp_strategy['entry']):,.0f}")
        print(f"  Entry2 (50%)   : ${gp_strategy.get('entry2', gp_strategy['entry']):,.0f}")
        print(f"  Stop Loss      : ${gp_strategy['sl']:,.0f}")
        print(f"  TP1            : ${gp_strategy['tp1']:,.0f}")
        print(f"  TP2            : ${gp_strategy['tp2']:,.0f}")
    print(f"  Reason         : {gp_strategy['reason']}")
    print("=" * 60)
    
    # 11b. Elliott Wave Fibo Strategy (NEW INDICATOR)
    print("\n[*] Calculating Elliott Wave + Fibonacci Strategy...")
    ew_fibo = calculate_elliott_wave_fibo(df, lookback=100)
    plan['elliott_wave_fibo'] = ew_fibo
    
    print("\n" + "=" * 60)
    print("  🌊 ELLIOTT WAVE + FIBONACCI (NEW)")
    print("=" * 60)
    print(f"  Impulse Type   : {ew_fibo['impulse_type']}")
    print(f"  Wave Context   : {ew_fibo['wave_context']}")
    print(f"  Swing High     : ${ew_fibo['swing_high']:,.0f}")
    print(f"  Swing Low      : ${ew_fibo['swing_low']:,.0f}")
    ote = ew_fibo['ote_zone']
    print(f"  OTE Zone       : ${ote['low']:,.0f} - ${ote['high']:,.0f}")
    if ew_fibo.get('candlestick_pattern'):
        cp = ew_fibo['candlestick_pattern']
        print(f"  Candle Pattern : {cp.get('pattern', 'None')} ({cp.get('type', 'None')})")
    print("-" * 60)
    ew_emoji = "🟢" if ew_fibo['action'] == 'LONG' else ("🔴" if ew_fibo['action'] == 'SHORT' else "⚪")
    print(f"  {ew_emoji} Action: {ew_fibo['action']}")
    if ew_fibo['valid']:
        print(f"  Entry          : ${ew_fibo['entry']:,.0f}")
        print(f"  Stop Loss      : ${ew_fibo['sl']:,.0f}")
        print(f"  TP1            : ${ew_fibo['tp1']:,.0f}")
        print(f"  TP2            : ${ew_fibo['tp2']:,.0f}")
    print(f"  Reason         : {ew_fibo['reason']}")
    print("=" * 60)
    
    # 12. Generate Charts
    print("\n[*] Generating chart with Trading Plan...")
    chart_path = plot_plotly_chart(df, timeframes_bias, pivots, poc_data, plan)
    if chart_path:
        print(f"[+] Saved: {chart_path}")
        
    # Generate Fibo Chart
    print("\n[*] Generating Fibonacci Chart...")
    fibo_chart_path = plot_fibo_chart(df, gp_strategy, ew_fibo)
    if fibo_chart_path:
        plan['fibo_chart_path'] = fibo_chart_path
        print(f"[+] Saved: {fibo_chart_path}")
    
    # 13. Print Trading Plan
    print_trading_plan(plan, pivots, poc_data)
    
    # 14. Final Summary
    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"  Current Price  : ${df['close'].iloc[-1]:,.0f}")
    print(f"  RSI (14)       : {df['RSI'].iloc[-1]:.1f}")
    print(f"  MACD           : {df['MACD'].iloc[-1]:,.0f} ({('Bullish' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'Bearish')})")
    print(f"  MA20/50        : {('Bullish' if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else 'Bearish')}")
    print(f"  BB Position    : {('Upper' if df['close'].iloc[-1] > df['BB_Middle'].iloc[-1] else 'Lower')}")
    print(f"  Volume         : {df['Vol_Ratio'].iloc[-1]:.2f}x")
    print(f"  POC            : ${poc_data['poc']:,.0f}")
    print(f"  Pivot PP       : ${pivots['PP']:,.0f}")
    print("-" * 60)
    print(f"  Gann Fan       : {gann_signal}")
    print(f"  Lunar          : {moon_phase['phase_name']} ({lunar_signal['sentiment']})")
    print(f"  Mercury        : {'RETROGRADE' if mercury['is_retrograde'] else 'Direct'}")
    print("-" * 60)
    if chartprime_data:
        tl = chartprime_data['trend_levels']
        lro = chartprime_data['lro']
        print(f"  [15m] Trend Levels : {tl['trend_direction']} | H:${tl['trend_high']:,.0f} M:${tl['trend_mid']:,.0f} L:${tl['trend_low']:,.0f}")
        print(f"  [15m] LRO          : {lro['oscillator']:.2f} ({lro['status']})")
        print(f"  [15m] ChartPrime   : {chartprime_data['entry_signal']} (Score: {chartprime_data['score']:+d})")
    print("-" * 60)
    print(f"  Overall Bias   : {overall_bias}")
    print(f"  Direction      : {plan['direction']}")
    print("=" * 60)
    
    # 15. Send to Telegram
    if Config.SEND_TO_TELEGRAM:
        print("\n[*] Sending to Telegram...")
        caption = create_telegram_caption(plan, analyses, pivots, poc_data, df, derivatives_data, smc_data)
        
        # Check if Fibo chart exists
        if plan.get('fibo_chart_path') and chart_path:
            # Send both images as album (2 photos in 1 message)
            print("[*] Sending Album (Main + Fibo Charts)...")
            image_paths = [chart_path, plan['fibo_chart_path']]
            send_telegram_media_group(image_paths, caption)
        elif chart_path:
            # Fallback to single photo
            send_telegram_photo(chart_path, caption)
    
    print("\n[*] Done!")
    return df, timeframes_bias, pivots, poc_data, plan, analyses


if __name__ == "__main__":
    main()
