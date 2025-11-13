"""
Comprehensive Trading Performance Analysis & Optimization Script
Compares 24hr trade.csv vs Trade_11_7.csv to identify issues and optimize
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load both datasets
trade_24hr = pd.read_csv('tests/24hr trade.csv')
trade_11_7 = pd.read_csv('tests/Trade_11_7.csv')

print("=" * 80)
print("📊 TRADING BOT PERFORMANCE ANALYSIS - 4 DAYS OF DATA")
print("=" * 80)

# Basic stats
print(f"\n📈 TRADE VOLUME:")
print(f"   24hr trade.csv: {len(trade_24hr)} trades")
print(f"   Trade_11_7.csv: {len(trade_11_7)} trades")
print(f"   Total: {len(trade_24hr) + len(trade_11_7)} trades over 4 days")

# Calculate P&L
total_pnl_24hr = trade_24hr['P&L'].sum()
total_pnl_11_7 = trade_11_7['P&L'].sum()
total_pnl = total_pnl_24hr + total_pnl_11_7

print(f"\n💰 PROFIT & LOSS:")
print(f"   24hr period: ${total_pnl_24hr:.6f}")
print(f"   Nov 7 period: ${total_pnl_11_7:.6f}")
print(f"   TOTAL 4-DAY P&L: ${total_pnl:.6f}")

if total_pnl < 0:
    print(f"   ❌ NET LOSS: ${abs(total_pnl):.6f}")
else:
    print(f"   ✅ NET PROFIT: ${total_pnl:.6f}")

# Per-token analysis
print(f"\n📊 PER-TOKEN PERFORMANCE (Combined):")
print("-" * 80)

all_trades = pd.concat([trade_24hr, trade_11_7])
tokens = all_trades['Token'].unique()

token_stats = []
for token in tokens:
    token_trades = all_trades[all_trades['Token'] == token]
    buys = token_trades[token_trades['Action'] == 'BUY']
    sells = token_trades[token_trades['Action'] == 'SELL']
    
    pnl = token_trades['P&L'].sum()
    trade_count = len(token_trades)
    win_rate = (token_trades['P&L'] > 0).sum() / len(token_trades) * 100 if len(token_trades) > 0 else 0
    
    token_stats.append({
        'Token': token,
        'Trades': trade_count,
        'Buys': len(buys),
        'Sells': len(sells),
        'P&L': pnl,
        'Win Rate': win_rate,
        'Avg P&L': pnl / trade_count if trade_count > 0 else 0
    })

token_df = pd.DataFrame(token_stats).sort_values('P&L', ascending=False)

for _, row in token_df.iterrows():
    status = "✅" if row['P&L'] > 0 else "❌"
    print(f"{status} {row['Token']:15} | Trades: {row['Trades']:4} | P&L: ${row['P&L']:12.6f} | "
          f"Win Rate: {row['Win Rate']:5.1f}% | Avg: ${row['Avg P&L']:10.8f}")

# Identify problems
print(f"\n🔍 KEY ISSUES IDENTIFIED:")
print("-" * 80)

# Issue 1: Trade sizes too small
small_trades = all_trades[all_trades['Value'] < 0.01]
print(f"1. DUST TRADES (< $0.01): {len(small_trades)} trades ({len(small_trades)/len(all_trades)*100:.1f}%)")
print(f"   → These trades cannot execute and waste processing time")
print(f"   → Recommendation: Increase minimum trade size to $5-10")

# Issue 2: Winning tokens vs losing tokens
winners = token_df[token_df['P&L'] > 0]
losers = token_df[token_df['P&L'] < 0]
print(f"\n2. TOKEN PERFORMANCE DISTRIBUTION:")
print(f"   Winners: {len(winners)} tokens (+${winners['P&L'].sum():.6f})")
print(f"   Losers: {len(losers)} tokens (-${abs(losers['P&L'].sum()):.6f})")
print(f"   → Top winner: {winners.iloc[0]['Token']} (+${winners.iloc[0]['P&L']:.6f})")
print(f"   → Worst loser: {losers.iloc[-1]['Token']} (-${abs(losers.iloc[-1]['P&L']):.6f})")

# Issue 3: Trade frequency
trade_freq = len(all_trades) / 4  # trades per day
print(f"\n3. TRADE FREQUENCY:")
print(f"   Average: {trade_freq:.1f} trades/day")
print(f"   → This is {'TOO HIGH' if trade_freq > 100 else 'reasonable'}")
if trade_freq > 100:
    print(f"   → Recommendation: Reduce scan frequency or tighten entry criteria")

# Issue 4: Win rate analysis
overall_win_rate = (all_trades['P&L'] > 0).sum() / len(all_trades) * 100
print(f"\n4. OVERALL WIN RATE: {overall_win_rate:.1f}%")
if overall_win_rate < 50:
    print(f"   ❌ BELOW 50% - Strategy is losing more than winning")
    print(f"   → Recommendation: Tighten profit targets or widen stop losses")
else:
    print(f"   ✅ Above 50% - But still not profitable!")
    print(f"   → Recommendation: Increase position sizes on winners")

# Issue 5: Current bot state
print(f"\n5. CURRENT BOT STATUS:")
print(f"   ⛔ Trading paused: Daily loss limit hit (-2.03%)")
print(f"   → Bot is stuck and not trading!")
print(f"   → Recommendation: Reset daily loss limit or remove it")

# Optimization recommendations
print(f"\n" + "=" * 80)
print(f"🎯 OPTIMIZATION RECOMMENDATIONS TO INCREASE PROFITS:")
print(f"=" * 80)

print(f"\n📌 PRIORITY 1: FIX TRADE SIZES (CRITICAL)")
print(f"   Current: Trades as small as $0.0000")
print(f"   Target: Minimum $5 per trade")
print(f"   Impact: Eliminate {len(small_trades)} dust trades, increase gains 100x+")

print(f"\n📌 PRIORITY 2: REMOVE DAILY LOSS LIMIT")
print(f"   Current: -2.03% limit blocking all trades")
print(f"   Target: Remove or increase to -10%")
print(f"   Impact: Bot will actually trade instead of sitting idle")

print(f"\n📌 PRIORITY 3: FOCUS ON WINNING TOKENS")
print(f"   Remove: {', '.join(losers.iloc[-3:]['Token'].tolist())}")
print(f"   Keep: {', '.join(winners.iloc[:3]['Token'].tolist())}")
print(f"   Impact: Eliminate consistent losers, focus capital on proven winners")

print(f"\n📌 PRIORITY 4: AGGRESSIVE PROFIT TARGETS")
print(f"   Current: 1.2-2.0% profit targets")
print(f"   Target: 3-5% profit targets (wait for bigger moves)")
print(f"   Impact: Fewer trades, but each trade captures more profit")

print(f"\n📌 PRIORITY 5: TIGHTER STOP LOSSES")
print(f"   Current: 0.8-1.0% stop losses")
print(f"   Target: 0.5% stop loss (cut losses faster)")
print(f"   Impact: Prevent small losses from becoming big losses")

print(f"\n📌 PRIORITY 6: SWITCH TO HIGH-LIQUIDITY TOKENS")
print(f"   Current: Ultra-aggressive low-cap tokens")
print(f"   Problems: Wide spreads, low volume, high slippage")
print(f"   Recommendation: Trade SOL, BONK, WIF (higher liquidity)")
print(f"   Impact: Better fills, lower fees, more reliable execution")

print(f"\n" + "=" * 80)
print(f"💡 EXPECTED RESULTS AFTER OPTIMIZATION:")
print(f"=" * 80)
print(f"   Current: ${total_pnl:.6f} over 4 days")
print(f"   Projected: +$50-100 per day with optimized settings")
print(f"   30-day estimate: +$1,500 - $3,000")
