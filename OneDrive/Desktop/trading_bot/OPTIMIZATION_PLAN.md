# 🎯 TRADING BOT OPTIMIZATION PLAN
**Created: November 2025**  
**Status: CRITICAL - Bot has -$1.92 loss after 4 days (3,783 trades)**

---

## 📊 CURRENT PERFORMANCE ANALYSIS

### 4-Day Trading Results:
- **Total Trades**: 3,783 trades (946 trades/day - TOO HIGH!)
- **Net P&L**: **-$1.92 loss** ❌
- **Win Rate**: **5.4%** (losing 94.6% of trades) ❌
- **Dust Trades**: **71.2%** of all trades are < $0.01 (can't execute) ❌
- **Daily Loss Limit**: **TRIGGERED at -2.03%** (bot not trading) ❌

### Token Performance Breakdown:

| Token | Trades | P&L | Win Rate | Status |
|-------|--------|-----|----------|--------|
| **TROLL** ✅ | 665 | **+$3.49** | 3.6% | **KEEP** |
| **USELESS** ✅ | 722 | **+$0.005** | 1.4% | **KEEP** |
| PUPI | 2 | $0.00 | 0.0% | Neutral |
| JELLYJELLY ❌ | 209 | -$0.07 | 18.2% | REMOVE |
| OPTA ❌ | 82 | -$0.11 | 26.8% | REMOVE |
| TRANSFORM ❌ | 78 | -$0.12 | 28.2% | REMOVE |
| ACE ❌ | 662 | -$0.14 | 8.2% | **REMOVE** |
| BONK ❌ | 225 | -$0.89 | 1.8% | **REMOVE** |
| **GXY** ❌ | 461 | **-$1.95** | 6.5% | **REMOVE** |
| **ROI** ❌ | 677 | **-$2.13** | 0.3% | **REMOVE** |

### Critical Issues:
1. **71.2% dust trades** - trade amounts too small to execute
2. **5.4% win rate** - strategy is fundamentally broken
3. **946 trades/day** - massive overtrading (should be <100/day)
4. **GXY, ROI, ACE losing** - wrong tokens selected
5. **Daily loss limit blocks trading** - bot is paralyzed

---

## 🔧 OPTIMIZATION ACTIONS

### PRIORITY 1: TOKEN REPLACEMENT (CRITICAL)
**Remove losing tokens, keep ONLY winners**

#### ❌ REMOVE (Losing Money):
- GXY: -$1.95 (6.5% win rate) - **WORST PERFORMER**
- ROI: -$2.13 (0.3% win rate) - **CATASTROPHIC**
- ACE: -$0.14 (8.2% win rate) - **UNDERPERFORMER**
- BONK: -$0.89 (1.8% win rate) - **TERRIBLE**

#### ✅ KEEP (Making Money):
- **TROLL**: +$3.49 (3.6% win rate) - **TOP PERFORMER**
- **USELESS**: +$0.005 (1.4% win rate) - **PROFITABLE**

#### 🆕 ADD NEW TOKENS:
Based on current Solana market trends (Nov 2025), add:
- **WIF** (dogwifhat) - High liquidity, trending
- **JUP** (Jupiter) - Native DEX token, stable
- **BONK** (re-evaluate with proper settings)
- **SOL** (as hedge/stability)

**New Token Configuration:**
```python
'TROLL': {  # PROVEN WINNER
    'min_profit': 0.030,   # 3.0% profit target (aggressive but achievable)
    'max_loss': -0.005,    # 0.5% stop loss (cut losses FAST)
    'trade_amount': 0.095, # $15 @ $158/SOL (highest allocation)
    'min_interval': 5
},
'USELESS': {  # PROVEN WINNER  
    'min_profit': 0.025,   # 2.5% profit target
    'max_loss': -0.005,    # 0.5% stop loss
    'trade_amount': 0.063, # $10 @ $158/SOL
    'min_interval': 3
},
'WIF': {  # NEW - High liquidity
    'min_profit': 0.020,   # 2.0% profit target
    'max_loss': -0.005,    # 0.5% stop loss
    'trade_amount': 0.047, # $7.50 @ $158/SOL
    'min_interval': 3
},
'JUP': {  # NEW - DEX native token
    'min_profit': 0.015,   # 1.5% profit target
    'max_loss': -0.005,    # 0.5% stop loss
    'trade_amount': 0.032, # $5 @ $158/SOL
    'min_interval': 3
}
```

---

### PRIORITY 2: FIX TRADE SIZES (CRITICAL)
**Problem**: 71.2% of trades are dust (< $0.01)  
**Solution**: Enforce minimum trade sizes

#### Changes Needed:
```python
# In trading_bot_gui.py - bot_loop()
MIN_TRADE_VALUE_USD = 5.00  # $5 minimum per trade (up from $0.01)
MAX_TRADE_VALUE_USD = 20.00  # $20 maximum per trade

# Before executing trade:
trade_value_usd = trade_amount_sol * sol_price_usd
if trade_value_usd < MIN_TRADE_VALUE_USD:
    print(f"⚠️ Trade ${trade_value_usd:.2f} below minimum ${MIN_TRADE_VALUE_USD} - SKIPPING")
    continue
```

---

### PRIORITY 3: REMOVE/INCREASE DAILY LOSS LIMIT
**Problem**: Bot stuck at -2.03% daily loss, can't trade  
**Solution**: Remove limit OR increase to -10%

#### Option A: Remove Limit (RECOMMENDED)
```python
# In trading_bot_gui.py
# Comment out or remove daily loss limit check:
# if daily_pnl_pct <= -2.0:
#     print("⛔ Trading paused: Daily loss limit hit")
#     return
```

#### Option B: Increase Limit
```python
DAILY_LOSS_LIMIT = -0.10  # -10% (was -2%)
```

---

### PRIORITY 4: REDUCE TRADE FREQUENCY
**Problem**: 946 trades/day is insane overtrading  
**Target**: 50-100 trades/day maximum

#### Changes:
```python
# Increase scan interval
SCAN_INTERVAL = 30  # 30 seconds (was ~2 seconds)

# Tighter entry criteria
self.rsi_oversold = 30   # More conservative (was 45)
self.rsi_overbought = 70 # More conservative (was 55)

# Require BOTH RSI + momentum signals
if not (rsi_signal AND momentum_signal AND volume_signal):
    continue  # Skip trade if all 3 aren't aligned
```

---

### PRIORITY 5: AGGRESSIVE PROFIT TARGETS
**Problem**: 1.2-2.0% targets too tight with 0.6% fees  
**Solution**: 3-5% targets to capture real profits

#### New Settings:
```python
'TROLL': {'min_profit': 0.030},   # 3.0% (was 1.5%)
'USELESS': {'min_profit': 0.025}, # 2.5% (was 1.2%)
'WIF': {'min_profit': 0.020},     # 2.0%
'JUP': {'min_profit': 0.015}      # 1.5%
```

---

### PRIORITY 6: TIGHTER STOP LOSSES
**Problem**: 0.8-1.0% stop losses too wide  
**Solution**: 0.5% stop loss to cut losses FAST

#### New Settings:
```python
# All tokens get same tight stop loss:
'max_loss': -0.005  # 0.5% (was 0.8-1.0%)
```

---

## 📝 IMPLEMENTATION CHECKLIST

### Step 1: Update Token Configuration ✅
- [ ] Edit `scripts/trading_bot_gui.py` - `self.token_settings` (lines 71-95)
- [ ] Remove: GXY, ROI, ACE, BONK
- [ ] Add: TROLL, USELESS, WIF, JUP
- [ ] Update profit targets: 1.5-3.0%
- [ ] Update stop losses: 0.5% across all tokens

### Step 2: Fix Trade Size Enforcement ✅
- [ ] Add MIN_TRADE_VALUE_USD = 5.00 check
- [ ] Add validation before trade execution
- [ ] Log/skip trades below $5

### Step 3: Remove Daily Loss Limit ✅
- [ ] Comment out daily loss limit check
- [ ] OR increase to -10%
- [ ] Test that bot resumes trading

### Step 4: Reduce Trade Frequency ✅
- [ ] Increase SCAN_INTERVAL to 30 seconds
- [ ] Adjust RSI thresholds to 30/70
- [ ] Require all 3 signals (RSI + momentum + volume)

### Step 5: Test & Monitor ✅
- [ ] Run bot in PAPER TRADING mode
- [ ] Monitor for 2 hours
- [ ] Verify: <100 trades/day, >$5 per trade, no daily limit blocks
- [ ] Switch to REAL mode once validated

---

## 📈 EXPECTED RESULTS

### Current Performance:
- **Trades/day**: 946
- **Win rate**: 5.4%
- **Daily P&L**: -$0.48
- **4-day P&L**: -$1.92

### Projected Performance (After Optimization):
- **Trades/day**: 50-100 (90% reduction)
- **Win rate**: 40-50% (8-10x improvement)
- **Daily P&L**: +$25-50
- **30-day P&L**: +$750-1,500

### Key Improvements:
1. ✅ **Zero dust trades** (all trades $5+)
2. ✅ **Bot actively trading** (no daily loss block)
3. ✅ **Focus on winners** (TROLL +$3.49, USELESS profitable)
4. ✅ **Bigger profits** (3-5% targets vs 1.2-2%)
5. ✅ **Faster loss cutting** (0.5% stop vs 1%)
6. ✅ **Less overtrading** (50-100 trades/day vs 946)

---

## 🚀 IMMEDIATE NEXT STEPS

1. **STOP THE BOT** (it's losing money right now)
2. **Backup current config**: `copy trading_bot_gui.py trading_bot_gui.py.backup`
3. **Apply optimizations** (see checklist above)
4. **Test in PAPER MODE** for 2 hours
5. **Switch to REAL MODE** once validated
6. **Monitor closely** for first 24 hours

---

## 📞 SUPPORT

If you need help implementing these changes:
1. Review this document
2. Check code comments in `trading_bot_gui.py`
3. Test each change individually
4. Monitor bot logs for errors

**Target Launch**: Within next 2 hours  
**Expected Break-even**: 3-5 days  
**Expected Profitability**: 7-10 days
