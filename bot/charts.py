"""
Chart generation logic for the financial-telegram-bot.
Uses matplotlib and seaborn to create professional financial visualizations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import requests
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from bot.config import FRED_SERIES, URLS, CHART_STYLE, DEFAULT_FIGSIZE, REGIME_COLORS

# Set style for professional-looking charts
sns.set_style(CHART_STYLE)
plt.rcParams['figure.figsize'] = DEFAULT_FIGSIZE
plt.rcParams['font.size'] = 10

def create_spy_stats_chart(stats: Dict[str, float], output_file: str = 'spy_stats.png') -> str:
    """Create SPY statistics chart with a modern card design"""
    print("Creating SPY statistics chart...")
    
    try:
        # Validate all required stats
        required_keys = ['current', 'ma_200', 'ma_200_pct', 'week_52_high', 'high_52w_pct', 'rsi_9d', 'return_3y_pct']
        for key in required_keys:
            if key not in stats or not np.isfinite(stats[key]):
                raise ValueError(f"Invalid or missing stat: {key}")
                
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        color_text = '#2d2d2d'
        color_gray = '#888888'
        
        rect = patches.FancyBboxPatch(
            (0.05, 0.05), 0.9, 0.9,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            edgecolor='#d4d4d4',
            facecolor='#ffffff',
            linewidth=1.5,
            zorder=0
        )
        ax.add_patch(rect)
        
        ax.text(0.12, 0.82, 'SPY', fontsize=40, color=color_text, fontweight='bold', va='center', ha='left')
        ax.text(0.5, 0.82, f"{stats['current']:.2f}", fontsize=56, color=color_text, fontweight='bold', va='center', ha='center')
        ax.plot([0.1, 0.9], [0.68, 0.68], color='#e0e0e0', linewidth=1.5)
        
        slider_y = 0.52
        ax.text(0.1, 0.60, '200D ', fontsize=16, color=color_gray, fontweight='bold', ha='left', va='bottom')
        ax.text(0.18, 0.60, f"{stats['ma_200']:.2f}", fontsize=18, color=color_text, fontweight='bold', ha='left', va='bottom')
        ax.text(0.78, 0.60, '52wH ', fontsize=16, color=color_gray, fontweight='bold', ha='right', va='bottom')
        ax.text(0.9, 0.60, f"{stats['week_52_high']:.2f}", fontsize=18, color=color_text, fontweight='bold', ha='right', va='bottom')
        
        bar_height = 0.025
        ax.add_patch(patches.Rectangle((0.1, slider_y - bar_height/2), 0.8, bar_height, facecolor='#d4d4d4', edgecolor='none', zorder=1))
        
        range_span = stats['week_52_high'] - stats['ma_200']
        position = (stats['current'] - stats['ma_200']) / range_span if range_span > 0 else 0.5
        position = max(0, min(1, position))
        
        marker_x = 0.1 + (0.8 * position)
        ax.add_patch(patches.Rectangle((marker_x - 0.005, slider_y - bar_height - 0.01), 0.01, bar_height * 2 + 0.02, facecolor='#333333', edgecolor='none', zorder=2))
        
        ax.text(0.1, 0.45, f"{stats['ma_200_pct']:.2f}%", fontsize=14, color=color_gray, ha='left', va='top')
        ax.text(0.9, 0.45, f"{stats['high_52w_pct']:.2f}%", fontsize=14, color=color_gray, ha='right', va='top')
        ax.plot([0.1, 0.9], [0.38, 0.38], color='#e0e0e0', linewidth=1.5)
        ax.plot([0.48, 0.48], [0.1, 0.35], color='#e0e0e0', linewidth=1.5)
        
        # RSI Gauge
        gauge_center_x, gauge_center_y = 0.28, 0.18
        gauge_radius_outer, gauge_radius_inner = 0.13, 0.09
        rsi_segments = [(0, 30, '#5a9f5a'), (30, 70, '#d4d4d4'), (70, 100, '#c75450')]
        
        for start, end, color in rsi_segments:
            theta_start, theta_end = np.pi - (start / 100) * np.pi, np.pi - (end / 100) * np.pi
            angles = np.linspace(theta_start, theta_end, 50)
            x_outer, y_outer = gauge_center_x + gauge_radius_outer * np.cos(angles), gauge_center_y + gauge_radius_outer * np.sin(angles)
            x_inner, y_inner = gauge_center_x + gauge_radius_inner * np.cos(angles[::-1]), gauge_center_y + gauge_radius_inner * np.sin(angles[::-1])
            verts = list(zip(x_outer, y_outer)) + list(zip(x_inner, y_inner))
            ax.add_patch(patches.Polygon(verts, facecolor=color, edgecolor='white', linewidth=1.5))
            
        for val in [30, 70]:
            theta_val = np.pi - (val / 100) * np.pi
            ax.plot([gauge_center_x + gauge_radius_inner * np.cos(theta_val), gauge_center_x + (gauge_radius_outer + 0.015) * np.cos(theta_val)],
                    [gauge_center_y + gauge_radius_inner * np.sin(theta_val), gauge_center_y + (gauge_radius_outer + 0.015) * np.sin(theta_val)], color='#333333', linewidth=2, zorder=3)
            ax.text(gauge_center_x + (gauge_radius_outer + 0.035) * np.cos(theta_val), gauge_center_y + (gauge_radius_outer + 0.035) * np.sin(theta_val), str(val), ha='center', va='center', fontsize=10, fontweight='bold', color=color_gray)
        
        val_rsi = min(max(stats['rsi_9d'], 0), 100)
        theta_needle = np.pi - (val_rsi / 100) * np.pi
        ax.plot([gauge_center_x, gauge_center_x + (gauge_radius_inner + 0.03) * np.cos(theta_needle)], [gauge_center_y, gauge_center_y + (gauge_radius_inner + 0.03) * np.sin(theta_needle)], color='#333333', linewidth=4, zorder=3)
        ax.add_patch(patches.Circle((gauge_center_x, gauge_center_y), 0.015, facecolor='#333333', zorder=4))
        ax.text(gauge_center_x - 0.02, 0.08, '9D RSI ', fontsize=14, color=color_gray, fontweight='bold', ha='right')
        ax.text(gauge_center_x, 0.08, f"{stats['rsi_9d']:.2f}", fontsize=20, color=color_text, fontweight='bold', ha='left')
        
        # 3Y Return
        ax.text(0.53, 0.30, '3Y Return ', fontsize=16, color=color_text, fontweight='normal', ha='left')
        ax.text(0.70, 0.30, f"{stats['return_3y_pct']:.2f}%", fontsize=18, color=color_text, fontweight='bold', ha='left')
        ax.add_patch(patches.Rectangle((0.53, 0.20), 0.35, 0.035, facecolor='#d4d4d4', edgecolor='none'))
        fill_pct = min(max(stats['return_3y_pct'] / 85.0, 0), 1.0)
        ax.add_patch(patches.Rectangle((0.53, 0.20), 0.35 * fill_pct, 0.035, facecolor='#0052cc', edgecolor='none'))
        ax.plot([0.53 + 0.35, 0.53 + 0.35], [0.19, 0.245], color='#333333', linewidth=3)
        ax.text(0.53 + 0.35, 0.18, 'Target 85%', fontsize=12, color=color_gray, ha='center', va='top')
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ SPY chart saved: {output_file}")
        return output_file
    except Exception as e:
        print(f"ERROR: Failed to create SPY chart: {str(e)}")
        raise

def create_yield_curve_chart(fred: Any, output_file: str = 'yield_curve.png') -> Tuple[str, float, datetime]:
    """Create yield curve inversion chart (10Y-2Y Treasury spread)"""
    print("Fetching yield curve data (T10Y2Y)...")
    try:
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=20*365)
        data = fred.get_series(FRED_SERIES['YIELD_10Y2Y'], start_date=start_date, end_date=end_date)
        if data.empty: raise ValueError("No data returned for T10Y2Y")
        latest_value = float(data.dropna().iloc[-1])
        latest_date = data.dropna().index[-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
            recessions = fred.get_series(FRED_SERIES['RECESSIONS'])
            recessions = recessions[recessions.index >= data.index.min()]
            in_recession, start = False, None
            for date, value in recessions.items():
                if value == 1 and not in_recession: start, in_recession = date, True
                elif value == 0 and in_recession:
                    ax.axvspan(start, date, alpha=0.2, color='gray', zorder=0)
                    in_recession = False
            if in_recession: ax.axvspan(start, recessions.index[-1], alpha=0.2, color='gray', zorder=0)
        except Exception as e: print(f"  Warning: No recession shading: {e}")
        ax.plot(data.index, data.values, color='blue', linewidth=2, label='10Y-2Y Spread')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Inversion Threshold')
        ax.set_title('US Treasury Yield Curve Spread (10Y - 2Y)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=11); ax.set_ylabel('Percentage Points', fontsize=11)
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
        fig.text(0.99, 0.01, 'Source: FRED', ha='right', va='bottom', fontsize=8, style='italic', alpha=0.7)
        plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
        return output_file, latest_value, latest_date
    except Exception as e:
        print(f"ERROR: Failed yield curve chart: {str(e)}"); raise

def create_profit_margin_chart(fred: Any, output_file: str = 'profit_margin.png') -> Tuple[str, float, datetime]:
    """Create US economy-wide profit margin chart"""
    print("Fetching profit margin data...")
    try:
        nos = fred.get_series(FRED_SERIES['CORP_PROFITS'])
        gdp = fred.get_series(FRED_SERIES['GDP'])
        df = pd.DataFrame({'nos': nos, 'gdp': gdp}).dropna()
        df['margin'] = (df['nos'] / df['gdp']) * 100
        latest_value, latest_date = float(df['margin'].iloc[-1]), df.index[-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['margin'], color='green', linewidth=2, label='Profit Margin')
        ax.set_title('US Economy-Wide Profit Margin', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Date'); ax.set_ylabel('Percentage (%)')
        ax.legend(loc='best'); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
        return output_file, latest_value, latest_date
    except Exception as e:
        print(f"ERROR: Failed profit margin chart: {str(e)}"); raise

def create_fear_greed_chart(output_file: str = 'fear_greed.png') -> Tuple[str, float, str]:
    """Create Fear & Greed Index chart from CNN data"""
    print("Fetching Fear & Greed Index data...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://edition.cnn.com/', 'Accept': 'application/json'}
        r = requests.get(URLS['FEAR_GREED'], headers=headers, timeout=10)
        r.raise_for_status(); data = r.json()
        fg = data['fear_and_greed']
        score, rating = float(fg['score']), fg['rating'].upper()
        
        fig = plt.figure(figsize=(10, 6), facecolor='white')
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
        ax.add_patch(patches.FancyBboxPatch((0.05, 0.05), 0.9, 0.9, boxstyle="round,pad=0.02,rounding_size=0.05", edgecolor='#d4d4d4', facecolor='#ffffff', linewidth=1.5, zorder=0))
        ax.text(0.5, 0.85, 'Fear & Greed Index', fontsize=32, fontweight='bold', ha='center')
        ax.plot([0.1, 0.9], [0.70, 0.70], color='#e0e0e0', linewidth=1.5)
        
        cx, cy, r_out, r_in = 0.5, 0.35, 0.25, 0.16
        segments = [(0, 25, 'Extreme Fear', '#c75450'), (25, 45, 'Fear', '#e8a798'), (45, 55, 'Neutral', '#d4d4d4'), (55, 75, 'Greed', '#a8c5a2'), (75, 100, 'Extreme Greed', '#5a9f5a')]
        for s, e, l, c in segments:
            ts, te = np.pi - (s/100)*np.pi, np.pi - (e/100)*np.pi
            ang = np.linspace(ts, te, 50)
            v = list(zip(cx + r_out*np.cos(ang), cy + r_out * np.sin(ang))) + list(zip(cx + r_in*np.cos(ang[::-1]), cy + r_in*np.sin(ang[::-1])))
            ax.add_patch(patches.Polygon(v, facecolor=c, edgecolor='white', linewidth=2))
            ax.text(cx + (r_out+0.04)*np.cos((ts+te)/2), cy + (r_out+0.04)*np.sin((ts+te)/2), l, ha='center', fontsize=10, fontweight='bold', color='#888888')

        val_fg = min(max(score, 0), 100)
        tn = np.pi - (val_fg/100)*np.pi
        ax.plot([cx, cx + (r_in+0.06)*np.cos(tn)], [cy, cy + (r_in+0.06)*np.sin(tn)], color='#333333', linewidth=5, zorder=3)
        ax.add_patch(patches.Circle((cx, cy), 0.025, facecolor='#333333', zorder=4))
        ax.text(cx, cy - 0.12, f"{int(round(score))}", fontsize=48, fontweight='bold', ha='center')
        ax.text(cx, cy - 0.20, rating, fontsize=20, fontweight='bold', ha='center')
        
        hist = f"Prev Close: {int(round(fg['previous_close']))}  |  1W: {int(round(fg['previous_1_week']))}  |  1M: {int(round(fg['previous_1_month']))}  |  1Y: {int(round(fg['previous_1_year']))}"
        ax.text(0.5, 0.10, hist, ha='center', fontsize=12, color='#888888', fontweight='bold')
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white'); plt.close()
        return output_file, score, rating
    except Exception as e:
        print(f"ERROR: Failed Fear & Greed chart: {str(e)}"); raise

def create_indicators_table(fred: Any, output_file: str = 'indicators_table.png') -> Tuple[str, List[Dict[str, str]]]:
    """Create a table of advanced economic indicators"""
    print("Creating indicators table...")
    try:
        indicators_data = []
        # Sahm Rule
        unrate = fred.get_series(FRED_SERIES['UNEMPLOYMENT']).dropna().tail(12)
        sahm = float(unrate.tail(3).mean() - unrate.min())
        indicators_data.append({'Indicator': 'Sahm Rule', 'Value': f'{sahm:.2f}', 'Change': '🔴 Signal' if sahm >= 0.5 else '🟢 Safe', 'Status': '⚠️' if sahm >= 0.5 else '✅'})
        # Consumer Sentiment
        sent = fred.get_series(FRED_SERIES['SENTIMENT']).dropna()
        indicators_data.append({'Indicator': 'Consumer Sentiment', 'Value': f'{sent.iloc[-1]:.1f}', 'Change': '🟢 Strong' if sent.iloc[-1] > 80 else '🔴 Weak', 'Status': '🛒'})
        # Initial Claims
        claims = float(fred.get_series(FRED_SERIES['CLAIMS']).dropna().tail(4).mean() / 1000)
        indicators_data.append({'Indicator': 'Initial Claims (4wk)', 'Value': f'{claims:.0f}K', 'Change': '🟢 Healthy' if claims < 250 else '🔴 Weak', 'Status': '📋'})
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight'); ax.axis('off')
        table_data = [[item['Status'] + ' ' + item['Indicator'], item['Value'], item['Change']] for item in indicators_data]
        table = ax.table(cellText=table_data, colLabels=['Indicator', 'Value', 'Status'], cellLoc='left', loc='center', colWidths=[0.4, 0.3, 0.3])
        table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1, 3)
        for i in range(3):
            cell = table[(0, i)]; cell.set_facecolor('#4472C4'); cell.set_text_props(weight='bold', color='white')
        plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
        return output_file, indicators_data
    except Exception as e:
        print(f"ERROR: Failed indicators table: {str(e)}"); raise

def create_bull_market_checklist(fred: Any, output_file: str = 'bull_checklist.png') -> Tuple[str, int, int]:
    """Create a comprehensive Bull Market Checklist table"""
    print("Creating Bull Market Checklist...")
    try:
        checklist_data = []
        sp500 = fred.get_series(FRED_SERIES['SP500']).dropna()
        current_sp = float(sp500.iloc[-1])
        ma_200_sp = float(sp500.tail(200).mean())
        trend_bullish = current_sp > ma_200_sp
        checklist_data.append({'Indicator': 'Market Trend', 'Criteria': 'S&P 500 > 200d MA', 'Reading': f'{current_sp:.0f}', 'Status': '✅ Bullish' if trend_bullish else '❌ Bearish', 'Bullish': trend_bullish})
        
        bullish_count = sum(1 for item in checklist_data if item['Bullish'])
        total_count = len(checklist_data)
        
        fig, ax = plt.subplots(figsize=(14, 10)); ax.axis('off')
        table_data = [[item['Indicator'], item['Criteria'], item['Reading'], item['Status']] for item in checklist_data]
        table = ax.table(cellText=table_data, colLabels=['Indicator', 'Criteria', 'Reading', 'Status'], cellLoc='left', loc='center')
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 2.5)
        for i in range(4):
            cell = table[(0, i)]; cell.set_facecolor('#2C3E50'); cell.set_text_props(weight='bold', color='white')
        plt.savefig(output_file, dpi=300, bbox_inches='tight'); plt.close()
        return output_file, bullish_count, total_count
    except Exception as e:
        print(f"ERROR: Failed checklist: {e}"); raise
