import requests
import json
import os
from datetime import datetime
from .config import Config


def send_telegram_photo(image_path, caption=""):
    """Send photo to all configured Telegram channels"""
    if not Config.SEND_TO_TELEGRAM:
        print("[!] Telegram sending disabled.")
        return
    
    if not Config.TELEGRAM_BOT_TOKEN:
        print("[!] Telegram Bot Token not set.")
        return

    for chat_id in Config.TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendPhoto"
        try:
            with open(image_path, 'rb') as photo:
                payload = {
                    'chat_id': chat_id,
                    'caption': caption,
                    'parse_mode': 'Markdown'
                }
                files = {'photo': photo}
                response = requests.post(url, data=payload, files=files, timeout=20)
                
                if response.status_code == 200:
                    print(f"  ✅ Telegram: Photo sent to {chat_id}!")
                else:
                    print(f"  ❌ Telegram Error ({chat_id}): {response.text}")
        except Exception as e:
            print(f"  ❌ Telegram Connection Error {chat_id}: {e}")


def send_telegram_message(message):
    """Send text message to all configured Telegram channels"""
    if not Config.SEND_TO_TELEGRAM:
        return
    
    if not Config.TELEGRAM_BOT_TOKEN:
        return

    for chat_id in Config.TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"  ❌ Telegram Message Error {chat_id}: {e}")


def send_telegram_media_group(image_paths, caption=""):
    """Send multiple photos as a media group (album) to Telegram"""
    if not Config.SEND_TO_TELEGRAM:
        print("[!] Telegram sending disabled.")
        return
    
    for chat_id in Config.TELEGRAM_CHAT_IDS:
        url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMediaGroup"
        
        try:
            media = []
            files = {}
            
            for i, path in enumerate(image_paths):
                file_key = f"photo{i}"
                media_item = {
                    "type": "photo",
                    "media": f"attach://{file_key}"
                }
                if i == 0 and caption:
                    media_item["caption"] = caption
                    media_item["parse_mode"] = "Markdown"
                
                media.append(media_item)
                files[file_key] = open(path, 'rb')
            
            payload = {
                'chat_id': chat_id,
                'media': json.dumps(media)
            }
            
            response = requests.post(url, data=payload, files=files, timeout=30)
            
            for f in files.values():
                f.close()
            
            if response.status_code == 200:
                print(f"  ✅ Telegram: Album {len(image_paths)} photos sent to {chat_id}!")
            else:
                print(f"  ❌ Telegram Album Error ({chat_id}): {response.text}")
                
        except Exception as e:
            print(f"  ❌ Telegram Connection Error {chat_id}: {e}")


def create_telegram_caption(plan, analyses, pivots, poc_data, df, derivatives_data=None, smc_data=None):
    """Create caption for Telegram photo"""
    current_price = df['close'].iloc[-1]
    rsi = df['RSI'].iloc[-1]
    
    def esc(text):
        if not isinstance(text, str):
            text = str(text)
        return text.replace('_', '\\_').replace('*', '\\*').replace('[', '\\[')

    # Multi-TF Bias
    tf_lines = []
    for tf in ['15m', '1H', '4H', '1D']:
        if tf in analyses:
            bias = analyses[tf]['bias']
            emoji = "🟢" if bias == 'BULLISH' else ("🔴" if bias == 'BEARISH' else "⚪")
            tf_lines.append(f"{emoji} {tf}: {bias}")
    
    # ChartPrime Analysis
    cp_info = ""
    if plan.get('cp_data'):
        cp = plan['cp_data']
        tl = cp['trend_levels']
        lro = cp['lro']
        cp_info = f"""
💠 *ChartPrime (15m):*
• Trend: {esc(tl['trend_direction'])}
  H:${tl['trend_high']:,.0f} L:${tl['trend_low']:,.0f}
• LRO: {lro['oscillator']:.2f} ({esc(lro['status'])})
• Score: {cp['score']:+d} ({esc(cp['entry_signal'])})"""

    # Derivatives
    deriv_info = ""
    if derivatives_data and derivatives_data.get('funding_rate_pct') is not None:
        fr = derivatives_data.get('funding_rate_pct', 0)
        ls = derivatives_data.get('ls_ratio', 0)
        long_pct = derivatives_data.get('long_ratio', 0) * 100
        oi = derivatives_data.get('open_interest', 0)
        
        fr_signal = "⚠️ Long Crowded" if fr > 0.05 else ("🟢 Short Crowded" if fr < -0.01 else "Neutral")
        
        deriv_info = f"""
📊 *Derivatives:*
• Funding: {fr:.4f}% ({esc(fr_signal)})
• L/S: {ls:.2f} ({long_pct:.0f}% Long)
• OI: {oi:,.0f} BTC"""

    # SMC Info
    smc_info = ""
    if smc_data:
        wyckoff = smc_data.get('wyckoff', {})
        lines = []
        if wyckoff.get('phase'):
            lines.append(f"• Wyckoff: {esc(wyckoff['phase'])} ({wyckoff.get('confidence', 0)}%)")
        if lines:
            smc_info = "\n\n🧠 *SMC Analysis:*\n" + "\n".join(lines)

    # Summary
    last = df.iloc[-1]
    macd_status = "Bullish" if last['MACD'] > last['MACD_Signal'] else "Bearish"
    ma_status = "Bullish" if last['MA20'] > last['MA50'] else "Bearish"
    
    summary_info = f"""
📋 *FINAL SUMMARY*
• RSI (14): {rsi:.1f}
• MACD: {last['MACD']:.0f} ({macd_status})
• MA20/50: {ma_status}
• POC: ${poc_data['poc']:,.0f}
• Pivot PP: ${pivots['PP']:,.0f}"""

    # Recommendation
    dir_emoji = "🟢" if plan['direction'] == 'LONG' else ("🔴" if plan['direction'] == 'SHORT' else "⚪")
    
    # Get win probabilities for both directions
    long_prob = plan.get('long_win_prob')
    short_prob = plan.get('short_win_prob')
    
    def format_prob(prob, label):
        if prob is None:
            return f"{label}: N/A"
        pct = prob * 100
        emoji = "🟢" if pct >= 65 else ("🟡" if pct >= 50 else "🔴")
        return f"{emoji} {label}: {pct:.0f}%"
    
    rec_info = ""
    if plan['direction'] in ['LONG', 'SHORT']:
        active_plan = plan['long'] if plan['direction'] == 'LONG' else plan['short']
        entry = active_plan.get('entry', 0)
        sl = active_plan.get('sl', 0)
        tp1 = active_plan.get('tp1', 0)
        tp2 = active_plan.get('tp2', 0)
        tp3 = active_plan.get('tp3', 0)
        rr = active_plan.get('rr', 0)
        win = active_plan.get('winrate', 0)
        
        rec_info = f"""
{dir_emoji} *RECOMMENDATION*
>>> {plan['direction']} at ${entry:,.0f}
   SL: ${sl:,.0f}
   TP1: ${tp1:,.0f} | TP2: ${tp2:,.0f} | TP3: ${tp3:,.0f}
   R/R: 1:{rr:.2f} | Win: {win:.0f}%
   Score: {plan['score']:+d}
   {format_prob(long_prob, 'LONG')} | {format_prob(short_prob, 'SHORT')}"""
    else:
        rec_info = f"""
{dir_emoji} *RECOMMENDATION*
   Wait for clear setup
   Score: {plan['score']:+d}
   {format_prob(long_prob, 'LONG')} | {format_prob(short_prob, 'SHORT')}"""

    # Golden Pocket Strategy
    gp_info = ""
    gp_strat = plan.get('golden_pocket_strategy')
    if gp_strat:
        gp = gp_strat.get('golden_pocket', {})
        gp_emoji = "🟢" if gp_strat['action'] == 'LONG' else ("🔴" if gp_strat['action'] == 'SHORT' else "⚪")
        
        gp_info = f"""

🏆 *GOLDEN POCKET (15m):*
• Trend: {gp_strat['trend']}
• Zone: ${gp.get('low', 0):,.0f} - ${gp.get('high', 0):,.0f}
{gp_emoji} {gp_strat['action']}"""

        # AI Win Probability for GP (Both directions)
        gp_long = gp_strat.get('long_win_prob')
        gp_short = gp_strat.get('short_win_prob')
        if gp_long is not None and gp_short is not None:
            hypo_label = " _(Giả định)_" if not gp_strat['valid'] else ""
            gp_info += f"\n{format_prob(gp_long, 'LONG')} | {format_prob(gp_short, 'SHORT')}{hypo_label}"
        
        if gp_strat['valid']:
            gp_info += f"""
   Entry1: ${gp_strat.get('entry1', gp_strat['entry']):,.0f} (50%)
   Entry2: ${gp_strat.get('entry2', gp_strat['entry']):,.0f} (50%)
   SL: ${gp_strat['sl']:,.0f}
   TP1: ${gp_strat['tp1']:,.0f} | TP2: ${gp_strat['tp2']:,.0f}"""
        else:
            gp_info += f"\n   {gp_strat['reason']}"
            # Show hypothetical Entry/SL/TP
            if gp_strat.get('entry') and gp_strat.get('sl') and gp_strat.get('tp1'):
                gp_info += f"""
   _(Giả định nếu vào lệnh:)_
   Entry: ${gp_strat['entry']:,.0f}
   SL: ${gp_strat['sl']:,.0f}
   TP1: ${gp_strat['tp1']:,.0f} | TP2: ${gp_strat['tp2']:,.0f}"""

    # Elliott Wave Fibo Strategy (AI-Enhanced)
    ew_info = ""
    ew_strat = plan.get('elliott_wave_fibo')
    if ew_strat:
        ote = ew_strat.get('ote_zone', {})
        ew_emoji = "🟢" if ew_strat['action'] == 'LONG' else ("🔴" if ew_strat['action'] == 'SHORT' else "⚪")
        
        ew_info = f"""

🌊 *ELLIOTT WAVE FIBO (AI):*
• Impulse: {ew_strat.get('impulse_type', 'N/A')}
• OTE Zone: ${ote.get('low', 0):,.0f} - ${ote.get('high', 0):,.0f}
{ew_emoji} {ew_strat['action']}"""
        
        # AI Win Probability (Both directions)
        ew_long = ew_strat.get('long_win_prob')
        ew_short = ew_strat.get('short_win_prob')
        if ew_long is not None and ew_short is not None:
            hypo_label = " _(Giả định)_" if not ew_strat['valid'] else ""
            ew_info += f"\n{format_prob(ew_long, 'LONG')} | {format_prob(ew_short, 'SHORT')}{hypo_label}"
        
        cp = ew_strat.get('candlestick_pattern', {})
        if cp and cp.get('pattern'):
            ew_info += f"\n• Candle: {cp['pattern']} ({cp['type']})"
        
        if ew_strat['valid']:
            ew_info += f"""
   Entry: ${ew_strat['entry']:,.0f}
   SL: ${ew_strat['sl']:,.0f}
   TP1: ${ew_strat['tp1']:,.0f} | TP2: ${ew_strat['tp2']:,.0f}"""
        else:
            ew_info += f"\n   {ew_strat['reason']}"
            # Show hypothetical Entry/SL/TP
            if ew_strat.get('entry') and ew_strat.get('sl') and ew_strat.get('tp1'):
                ew_info += f"""
   _(Giả định nếu vào lệnh:)_
   Entry: ${ew_strat['entry']:,.0f}
   SL: ${ew_strat['sl']:,.0f}
   TP1: ${ew_strat['tp1']:,.0f} | TP2: ${ew_strat['tp2']:,.0f}"""

    # Final Caption
    caption = f"""📊 *BTC/USDT Analysis*
🕐 {datetime.now().strftime("%d/%m/%Y %H:%M")}

💰 *Price:* ${current_price:,.0f}

*Multi-TF Bias:*
{chr(10).join(tf_lines)}{deriv_info}
{summary_info}
{rec_info}{gp_info}{ew_info}"""
    
    return caption

