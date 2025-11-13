# 🔍 Solana Token Scanner - User Guide

## What It Does

This standalone tool **automatically scans Solana tokens** and identifies the best trading opportunities based on:

- **📈 Price Momentum** - Tokens making big moves
- **💰 Volume Spikes** - Whale activity and buying pressure
- **📊 Technical Patterns** - Breakouts, dips, reversals
- **💎 Liquidity Depth** - Can you actually trade it without slippage?
- **🟢 Buy/Sell Pressure** - More buyers than sellers?

Each token gets a **score from 0-100** with a clear recommendation: BUY, WATCH, or AVOID.

---

## Quick Start

### 1. Scan Top 100 Tokens (Default)
```powershell
python token_scanner.py
```

### 2. Scan Top 50 Tokens
```powershell
python token_scanner.py --top 50
```

### 3. Analyze Specific Token
```powershell
python token_scanner.py --symbol WIF
python token_scanner.py --symbol BONK
python token_scanner.py --symbol JUP
```

### 4. Export Results to JSON
```powershell
python token_scanner.py --export
```

### 5. Only Show High-Quality Opportunities (70+ score)
```powershell
python token_scanner.py --min-score 70
```

---

## Understanding the Scores

### 🟢 BUY (70-100 points)
- Strong momentum + high volume + good liquidity
- Multiple positive signals
- Low risk, high potential
- **Action**: Consider adding to your trading bot

### 🟡 WATCH (50-69 points)
- Some good signals but also concerns
- Moderate opportunity
- **Action**: Monitor for better entry point

### 🔴 AVOID (0-49 points)
- Weak signals or red flags
- Low volume, bad liquidity, or suspicious patterns
- **Action**: Skip this token

---

## What the Scanner Looks For

### ✅ POSITIVE SIGNALS (Increase Score)

**Momentum (30 points max)**
- 🚀 1h pump: +5-10% 
- 🔥 6h explosion: +10-20%
- 💎 24h gain: +20%+

**Volume (25 points max)**
- 💰 $1M+ daily volume = 15 points
- 💵 $500k+ = 10 points
- 📊 $100k+ = 5 points

**Liquidity (25 points max)**
- 💎 $500k+ liquidity = 15 points
- ✅ $100k+ = 10 points
- ⚡ $50k+ = 5 points

**Patterns (20 points max)**
- 📈 Breakout (big move after consolidation)
- 💰 Dip buying (down 1h, up 24h)
- 🔄 Reversal (turning around)
- 🟢 Buy pressure (more buys than sells)

### ⚠️ WARNING SIGNALS (Decrease Score)

- 🚨 Pump risk: Huge spike on low volume (-20 points)
- ⚠️ Dead cat bounce: Up 1h but down 24h (-15 points)
- ⛔ Low liquidity: Under $50k (-10 points)
- 🔴 Sell pressure: More sellers than buyers

---

## Example Output

```
🎯 TOP 10 TRADING OPPORTUNITIES

1. 🟢 BUY WIF
   Score: 85/100
   Price: $0.476000 (+12.3% 24h)
   Volume: $2.4M | Liquidity: $850k
   Address: EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm
   ✅ Signals:
      • 🚀 Strong 1h momentum: +8.2%
      • 💰 High volume: $2.4M
      • 💎 Deep liquidity: $0.85M
   
2. 🟡 WATCH BONK
   Score: 62/100
   Price: $0.000025 (+5.1% 24h)
   Volume: $450k | Liquidity: $120k
   Address: DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
   ✅ Signals:
      • 📈 Positive 1h momentum: +3.4%
      • 💵 Good volume: $450k
   ⚠️  Warnings:
      • 🔴 Sell pressure: 1.3x more sells than buys
```

---

## Integration with Trading Bot

### Option 1: Manual Integration
1. Run scanner: `python token_scanner.py --export`
2. Review `token_opportunities.json`
3. Pick tokens with 70+ score
4. Add to `trading_bot_gui.py` target tokens

### Option 2: Automated (Future Enhancement)
- Scanner runs every 1 hour
- Automatically updates bot's token list
- Only adds tokens meeting criteria:
  - Score >= 70
  - Liquidity >= $100k
  - Volume >= $500k

---

## Advanced Usage

### Find Breakout Opportunities
```powershell
# Scan for tokens with recent momentum
python token_scanner.py --min-score 65 --top 200
```

### Monitor Specific Tokens
```powershell
# Check if WIF, JUP, BONK are good buys right now
python token_scanner.py --symbol WIF
python token_scanner.py --symbol JUP
python token_scanner.py --symbol BONK
```

### Export for Analysis
```powershell
# Get JSON data for spreadsheet analysis
python token_scanner.py --export --top 300
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `--top N` | Scan top N tokens (default: 100) |
| `--symbol XXX` | Analyze specific token |
| `--export` | Save results to JSON |
| `--min-score N` | Only show tokens with score >= N |

---

## Tips for Best Results

1. **Run regularly** - Market changes fast, scan every 1-2 hours
2. **Cross-reference** - Check DexScreener charts manually for top picks
3. **Start with BUY signals** - 70+ score tokens have best win rate
4. **Avoid low liquidity** - Even with high score, <$50k liquidity is risky
5. **Watch for patterns** - Breakouts and reversals are strongest signals
6. **Check volume** - $500k+ daily volume = safer trades

---

## FAQ

**Q: How often should I run the scanner?**
A: Every 1-2 hours during active trading. Market moves fast.

**Q: Can I trust 100-score tokens?**
A: High score means strong signals, but ALWAYS verify on DexScreener/Birdeye before trading.

**Q: What if no tokens score above 70?**
A: Lower `--min-score` to 60, or wait for better market conditions.

**Q: Should I add ALL 70+ tokens to my bot?**
A: No. Pick 3-5 best and diversify. Too many tokens = diluted capital.

**Q: Does this guarantee profits?**
A: No. This identifies OPPORTUNITIES, not guarantees. Use with proper risk management.

---

## Next Steps

1. **Test the scanner**: `python token_scanner.py --symbol WIF`
2. **Find opportunities**: `python token_scanner.py --min-score 70`
3. **Add to bot**: Update `trading_bot_gui.py` with top tokens
4. **Monitor results**: Track which scanner picks actually profit

**Good luck farming those gains! 🚀💰**
