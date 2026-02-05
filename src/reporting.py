from datetime import datetime


def print_multi_tf_analysis(analyses):
    """Print detailed multi-timeframe analysis table"""
    print("\n" + "=" * 90)
    print("  MULTI-TIMEFRAME DETAILED ANALYSIS")
    print("=" * 90)
    print(f"  {'TF':<6} {'Price':>12} {'RSI':>8} {'MACD':>10} {'MA20/50':>10} {'BB':>8} {'Vol':>8} {'BIAS':>10}")
    print("  " + "-" * 88)
    
    for tf in ['15m', '1H', '4H', '1D']:
        if tf in analyses:
            d = analyses[tf]
            bias_emoji = "[+]" if d['bias'] == 'BULLISH' else ("[-]" if d['bias'] == 'BEARISH' else "[=]")
            macd_trend = d.get('macd_trend', 'N/A')
            ma_signal = d.get('ma_signal', 'N/A')
            bb_signal = d.get('bb_signal', 'N/A')
            vol_signal = d.get('vol_signal', 'N/A')
            print(f"  {tf:<6} ${d['price']:>10,.0f} {d['rsi']:>7.1f} {macd_trend:>10} {ma_signal:>10} {bb_signal:>8} {vol_signal:>8} {bias_emoji} {d['bias']}")
    
    # Overall analysis
    bull_count = sum(1 for d in analyses.values() if d['bias'] == 'BULLISH')
    bear_count = sum(1 for d in analyses.values() if d['bias'] == 'BEARISH')
    overall = 'BULLISH' if bull_count > bear_count else ('BEARISH' if bear_count > bull_count else 'NEUTRAL')
    
    print("\n  " + "-" * 88)
    print(f"  OVERALL: {overall} ({bull_count} Bull / {bear_count} Bear)")
    print("=" * 90)
    
    # Fibonacci from 1D
    if '1D' in analyses and 'fib' in analyses['1D']:
        d = analyses['1D']
        print("\n  FIBONACCI LEVELS (1D):")
        for name, value in sorted(d['fib'].items(), key=lambda x: x[1], reverse=True):
            marker = " ◄ NEAR" if abs(value - d['price']) < 1500 else ""
            print(f"    Fib {name}: ${value:,.0f}{marker}")
    
    return overall


def print_trading_plan(plan, pivots, poc_data):
    """Print formatted trading plan"""
    print("\n" + "=" * 60)
    print("  TRADING PLAN - " + datetime.now().strftime("%d/%m/%Y %H:%M"))
    print("=" * 60)
    
    # Signals summary
    print("\n  SIGNAL ANALYSIS")
    print("  " + "-" * 56)
    for signal in plan.get('signals', []):
        name, desc, bias = signal
        emoji = "[+]" if bias == 'BULL' else ("[-]" if bias == 'BEAR' else "[=]")
        print(f"  {emoji} {name:12} : {desc}")
    
    recommendation = plan.get('recommendation', 'WAIT')
    print(f"\n  SCORE: {plan['score']:+d} => {recommendation}")
    
    # Long Plan
    print("\n  " + "=" * 56)
    print("  [+] INTRADAY LONG")
    lp = plan['long']
    tp3_long = lp.get('tp3', lp['tp2'])
    print(f"  Entry: ${lp['entry']:,.0f} | SL: ${lp['sl']:,.0f} | TP1: ${lp['tp1']:,.0f} | TP2: ${lp['tp2']:,.0f} | TP3: ${tp3_long:,.0f}")
    
    print("\n  [-] INTRADAY SHORT")
    sp = plan['short']
    tp3_short = sp.get('tp3', sp['tp2'])
    print(f"  Entry: ${sp['entry']:,.0f} | SL: ${sp['sl']:,.0f} | TP1: ${sp['tp1']:,.0f} | TP2: ${sp['tp2']:,.0f} | TP3: ${tp3_short:,.0f}")
    
    # Recommendation
    print("\n  " + "=" * 56)
    print("  RECOMMENDATION")
    print("  " + "=" * 56)
    
    if plan['direction'] == 'LONG':
        print(f"  >>> LONG at ${lp['entry']:,.0f}")
        print(f"     SL: ${lp['sl']:,.0f}")
        print(f"     TP1: ${lp['tp1']:,.0f} | TP2: ${lp['tp2']:,.0f} | TP3: ${tp3_long:,.0f}")
        print(f"     R/R: 1:{lp.get('rr', 1.5):.2f} | Win: {lp.get('winrate', 50):.0f}%")
    elif plan['direction'] == 'SHORT':
        print(f"  >>> SHORT at ${sp['entry']:,.0f}")
        print(f"     SL: ${sp['sl']:,.0f}")
        print(f"     TP1: ${sp['tp1']:,.0f} | TP2: ${sp['tp2']:,.0f} | TP3: ${tp3_short:,.0f}")
        print(f"     R/R: 1:{sp.get('rr', 1.5):.2f} | Win: {sp.get('winrate', 50):.0f}%")
    elif 'ALL TARGETS HIT' in recommendation:
        print("  🎉 ALL TARGETS HIT!")
        print("     Wait for new setup...")
    else:
        print("  ... WAIT - No clear signal")
        print("     Mixed indicators, wait for confirmation")
    
    print("  " + "=" * 56)


def print_golden_pocket_strategy(gp_data):
    """Print Golden Pocket Strategy details"""
    print("\n" + "=" * 60)
    print("  GOLDEN POCKET STRATEGY (15m)")
    print("=" * 60)
    
    if not gp_data:
        print("  [!] No Golden Pocket data available")
        return
    
    gp = gp_data.get('golden_pocket', {})
    print(f"  Trend: {gp_data.get('trend', 'N/A')}")
    print(f"  Zone: ${gp.get('low', 0):,.0f} - ${gp.get('high', 0):,.0f}")
    print(f"  Action: {gp_data.get('action', 'WAIT')}")
    
    if gp_data.get('valid'):
        print(f"\n  Entry: ${gp_data.get('entry', 0):,.0f}")
        print(f"  SL: ${gp_data.get('sl', 0):,.0f}")
        print(f"  TP1: ${gp_data.get('tp1', 0):,.0f}")
        print(f"  TP2: ${gp_data.get('tp2', 0):,.0f}")
    else:
        print(f"  Reason: {gp_data.get('reason', 'N/A')}")
    
    print("=" * 60)


def print_smc_analysis(smc_data):
    """Print Smart Money Concepts analysis"""
    print("\n" + "=" * 60)
    print("  SMC ANALYSIS")
    print("=" * 60)
    
    if not smc_data:
        print("  [!] No SMC data available")
        return
    
    wyckoff = smc_data.get('wyckoff', {})
    print(f"  Wyckoff Phase: {wyckoff.get('phase', 'N/A')} ({wyckoff.get('confidence', 0)}%)")
    print(f"  Description: {wyckoff.get('description', 'N/A')}")
    
    # FVG
    nearest_fvg = smc_data.get('nearest_fvg')
    if nearest_fvg:
        print(f"\n  Nearest FVG: {nearest_fvg['type'].title()} @ ${nearest_fvg['mid']:,.0f}")
    
    # Order Blocks
    nearest_ob = smc_data.get('nearest_ob')
    if nearest_ob:
        print(f"  Nearest OB: {nearest_ob['type'].title()} @ ${nearest_ob['mid']:,.0f}")
    
    # Liquidity
    liquidity = smc_data.get('liquidity', {})
    if liquidity.get('nearest_bsl'):
        print(f"  BSL: ${liquidity['nearest_bsl']['level']:,.0f}")
    if liquidity.get('nearest_ssl'):
        print(f"  SSL: ${liquidity['nearest_ssl']['level']:,.0f}")
    
    print(f"\n  SMC Score: {smc_data.get('score', 0):+d}")
    print("=" * 60)


def print_derivatives_info(derivatives_data):
    """Print derivatives/futures data"""
    print("\n" + "=" * 60)
    print("  DERIVATIVES DATA")
    print("=" * 60)
    
    if not derivatives_data:
        print("  [!] No derivatives data available")
        return
    
    fr = derivatives_data.get('funding_rate_pct', 0)
    ls = derivatives_data.get('ls_ratio', 0)
    long_pct = derivatives_data.get('long_ratio', 0) * 100
    oi = derivatives_data.get('open_interest', 0)
    
    fr_signal = "⚠️ Long Crowded" if fr > 0.05 else ("🟢 Short Crowded" if fr < -0.01 else "Neutral")
    
    print(f"  Funding Rate: {fr:.4f}% ({fr_signal})")
    print(f"  Long/Short: {ls:.2f} ({long_pct:.0f}% Long)")
    print(f"  Open Interest: {oi:,.0f} BTC")
    print("=" * 60)
