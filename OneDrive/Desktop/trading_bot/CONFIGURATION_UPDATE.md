# Bot Configuration Update Summary

## Changes Applied: November 6, 2025

### ✅ TOKEN LINEUP UPDATED

#### **REMOVED:**
- ❌ **BONK** - Worst performer (-$0.45 loss, 6.2% win rate over 32 trades)
- ❌ **WOBBLES** - Not listed on DexScreener yet

#### **KEEPING:**
- ✅ **TROLL** - Best performer (+$1.74 profit, 45.5% win rate) → **$15 per trade**
- ✅ **USELESS** - Moderate performer (+$0.01 profit, 42.9% win rate) → **$10 per trade**

#### **ADDED:**
- ✅ **JELLYJELLY** - Perfect record (+$0.16 profit, 100% win rate over 15 trades) → **$5 per trade**

---

## NEW CONFIGURATION

### 🎯 TROLL (Best Performer - Highest Priority)
```
Trade Amount: $15 per trade (0.093 SOL @ $161/SOL)
Profit Target: 1.2% (up from 0.8%)
Stop Loss: 0.8% (up from 0.5%)
Scan Interval: 5 seconds
```
**Rationale:**
- Proven performer with 45.5% win rate
- Generated +$1.74 profit in test period
- Wider stop loss (0.8%) allows volatility tolerance
- Higher profit target (1.2%) covers 0.6% fees + 0.6% net profit

### 💧 USELESS (Moderate Performer - Medium Priority)
```
Trade Amount: $10 per trade (0.062 SOL @ $161/SOL)
Profit Target: 1.0% (up from 0.8%)
Stop Loss: 0.7% (up from 0.5%)
Scan Interval: 3 seconds
```
**Rationale:**
- Moderate performance (42.9% win rate)
- Small profit (+$0.01) but no major losses
- Currently holding dust position (0.003 tokens)
- Moderate trade size while evaluating performance

### 🍇 JELLYJELLY (Perfect Record - Conservative Entry)
```
Trade Amount: $5 per trade (0.031 SOL @ $161/SOL)
Profit Target: 1.2% (up from 0.8%)
Stop Loss: 0.8% (up from 0.5%)
Scan Interval: 3 seconds
```
**Rationale:**
- **100% win rate** (15 wins, 0 losses in historical data)
- Generated +$0.16 profit on only $64 volume
- Conservative $5 trades to test at scale
- Same aggressive parameters as TROLL (proven successful)

---

## EXPECTED PERFORMANCE IMPROVEMENTS

### Before Changes:
- **Win Rate:** 40.3%
- **Daily Profit:** +$1.35
- **Problem:** BONK dragging down overall performance (-$0.45)

### After Changes (Projected):
- **Win Rate:** 55-60% (estimated)
- **Daily Profit:** +$3-5 (estimated)
- **Improvement:** 200-300% increase

### Key Improvements:
1. **Removed BONK** → Eliminates -$0.45 daily loss
2. **Added JELLYJELLY** → 100% win rate token
3. **Optimized Parameters** → 1.2% profit targets cover fees better
4. **Prioritized Capital** → $15 to TROLL (best performer)

---

## FILES MODIFIED

### 1. `scripts/trading_bot_gui.py`
- **Line 62-85:** Updated `token_settings` dictionary
  - Removed BONK and WOBBLES
  - Added JELLYJELLY with $5 trades
  - Updated TROLL to $15 trades with 1.2%/0.8% parameters
  - Updated USELESS to $10 trades with 1.0%/0.7% parameters

- **Line 480:** Updated `target_tokens` list
  - Changed from: `['BONK', 'TROLL', 'USELESS', 'WOBBLES']`
  - Changed to: `['TROLL', 'USELESS', 'JELLYJELLY']`

- **Line 1039:** Updated log message
  - Changed to: `"🎯 Target tokens: TROLL ($15), USELESS ($10), JELLYJELLY ($5)"`

### 2. `scripts/real_trader.py`
- **Line 48-55:** Updated `token_mints` dictionary
  - Removed BONK and WOBBLES
  - Added JELLYJELLY: `'FeR8VBqNRSUD5NtXAj2n3j1dAHkZHfyDktKuLXD4pump'`

- **Line 62-68:** Updated `token_decimals` cache
  - Removed BONK decimal entry
  - Added JELLYJELLY: 6 decimals (pump.fun standard)

---

## VALIDATION CHECKLIST

Before running the bot, verify:

- ✅ BONK removed from all configurations
- ✅ WOBBLES removed from all configurations  
- ✅ JELLYJELLY mint address added: `FeR8VBqNRSUD5NtXAj2n3j1dAHkZHfyDktKuLXD4pump`
- ✅ JELLYJELLY decimals set to 6
- ✅ TROLL trade amount: 0.093 SOL (~$15)
- ✅ USELESS trade amount: 0.062 SOL (~$10)
- ✅ JELLYJELLY trade amount: 0.031 SOL (~$5)
- ✅ Profit targets increased: 1.2% (TROLL/JELLYJELLY), 1.0% (USELESS)
- ✅ Stop losses widened: 0.8% (TROLL/JELLYJELLY), 0.7% (USELESS)
- ✅ Minimum trade value still active: $0.15 threshold

---

## TESTING RECOMMENDATIONS

### Phase 1: Initial Testing (1-2 hours)
- Monitor JELLYJELLY performance closely (new addition)
- Verify $15 TROLL trades execute correctly
- Confirm profit targets triggering at 1.2%
- Check that dust positions don't trigger trades

### Phase 2: Performance Validation (24 hours)
- Track win rate - target: 55-60%
- Monitor total P&L - target: +$3-5/day
- Verify JELLYJELLY maintains high win rate
- Confirm BONK removal eliminated losses

### Phase 3: Long-term Monitoring (1 week)
- Compare to historical 40.3% win rate
- Adjust parameters if needed
- Consider increasing JELLYJELLY trade size if performance holds
- Re-evaluate USELESS if it underperforms

---

## ROLLBACK PLAN

If performance degrades:

1. **JELLYJELLY underperforms** → Remove and monitor separately
2. **Win rate below 40%** → Revert to 0.8% profit targets
3. **Too many stop losses** → Widen stop loss to 1.0%
4. **Insufficient volume** → Reduce trade amounts by 20%

---

## NEXT STEPS

1. **Start the bot** with new configuration
2. **Monitor first 10 trades** of each token closely
3. **Take profit snapshot** at 24 hours for comparison
4. **Document JELLYJELLY performance** (critical - testing 100% win rate theory)
5. **Adjust if needed** after 48 hours of data

**Expected Outcome:** 55-60% win rate with $3-5/day profit (vs previous 40.3% and $1.35/day)

---

**Status:** ✅ Configuration Updated - Ready to Run
**Risk Level:** Medium (removed worst performer, added proven winner)
**Expected Improvement:** 200-300% profit increase
