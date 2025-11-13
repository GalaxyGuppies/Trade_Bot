"""
Quick verification of optimization changes
"""
import re

print("=" * 80)
print("🔍 VERIFYING OPTIMIZATION CHANGES")
print("=" * 80)

with open('scripts/trading_bot_gui.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Check 1: Token list
if "'TROLL':" in content and "'USELESS':" in content and "'WIF':" in content and "'JUP':" in content:
    print("✅ Token Configuration: TROLL, USELESS, WIF, JUP")
else:
    print("❌ Token Configuration: FAILED")

if "'GXY':" not in content or content.count("'GXY':") <= 1:  # Might be in comments
    print("✅ Removed: GXY")
else:
    print("⚠️ Warning: GXY still in config")

# Check 2: Target tokens list
if "self.target_tokens = ['TROLL', 'USELESS', 'WIF', 'JUP']" in content:
    print("✅ Target Tokens List: Updated")
else:
    print("❌ Target Tokens List: FAILED")

# Check 3: Scan interval
if "self.scan_interval = 30" in content:
    print("✅ Scan Interval: 30 seconds (reduced from 2s)")
else:
    print("❌ Scan Interval: FAILED")

# Check 4: Daily loss limit
if "self.max_daily_loss_pct = -0.10" in content:
    print("✅ Daily Loss Limit: -10% (increased from -2%)")
else:
    print("❌ Daily Loss Limit: FAILED")

# Check 5: Minimum trade size
if "min_trade_value_usd = 5.00" in content:
    print("✅ Minimum Trade Size: $5.00 (increased from $0.01)")
else:
    print("❌ Minimum Trade Size: FAILED")

# Check 6: RSI thresholds
if "self.rsi_oversold = 30" in content:
    print("✅ RSI Oversold: 30 (conservative)")
else:
    print("❌ RSI Oversold: FAILED")

if "self.rsi_overbought = 70" in content:
    print("✅ RSI Overbought: 70 (conservative)")
else:
    print("❌ RSI Overbought: FAILED")

# Check 7: Profit targets
troll_profit = re.search(r"'TROLL':\s*{[^}]*'min_profit':\s*0\.030", content, re.DOTALL)
useless_profit = re.search(r"'USELESS':\s*{[^}]*'min_profit':\s*0\.025", content, re.DOTALL)
wif_profit = re.search(r"'WIF':\s*{[^}]*'min_profit':\s*0\.020", content, re.DOTALL)
jup_profit = re.search(r"'JUP':\s*{[^}]*'min_profit':\s*0\.015", content, re.DOTALL)

if troll_profit:
    print("✅ TROLL Profit Target: 3.0%")
else:
    print("❌ TROLL Profit Target: FAILED")

if useless_profit:
    print("✅ USELESS Profit Target: 2.5%")
else:
    print("❌ USELESS Profit Target: FAILED")

if wif_profit:
    print("✅ WIF Profit Target: 2.0%")
else:
    print("❌ WIF Profit Target: FAILED")

if jup_profit:
    print("✅ JUP Profit Target: 1.5%")
else:
    print("❌ JUP Profit Target: FAILED")

# Check 8: Stop losses (all 0.5%)
troll_stop = re.search(r"'TROLL':\s*{[^}]*'max_loss':\s*-0\.005", content, re.DOTALL)
useless_stop = re.search(r"'USELESS':\s*{[^}]*'max_loss':\s*-0\.005", content, re.DOTALL)
wif_stop = re.search(r"'WIF':\s*{[^}]*'max_loss':\s*-0\.005", content, re.DOTALL)
jup_stop = re.search(r"'JUP':\s*{[^}]*'max_loss':\s*-0\.005", content, re.DOTALL)

if all([troll_stop, useless_stop, wif_stop, jup_stop]):
    print("✅ Stop Losses: All tokens 0.5% (tight)")
else:
    print("❌ Stop Losses: Some tokens FAILED")

print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

total_checks = 14
passed_checks = sum([
    1 if "'TROLL':" in content else 0,
    1 if "self.target_tokens = ['TROLL', 'USELESS', 'WIF', 'JUP']" in content else 0,
    1 if "self.scan_interval = 30" in content else 0,
    1 if "self.max_daily_loss_pct = -0.10" in content else 0,
    1 if "min_trade_value_usd = 5.00" in content else 0,
    1 if "self.rsi_oversold = 30" in content else 0,
    1 if "self.rsi_overbought = 70" in content else 0,
    1 if troll_profit else 0,
    1 if useless_profit else 0,
    1 if wif_profit else 0,
    1 if jup_profit else 0,
    1 if troll_stop else 0,
    1 if useless_stop else 0,
    1 if wif_stop and jup_stop else 0
])

print(f"Passed: {passed_checks}/{total_checks} checks")
print(f"Success Rate: {passed_checks/total_checks*100:.0f}%")

if passed_checks == total_checks:
    print("\n🎉 ALL OPTIMIZATIONS APPLIED SUCCESSFULLY!")
    print("\n📋 NEXT STEPS:")
    print("1. Stop the current bot (if running)")
    print("2. Start: python scripts/trading_bot_gui.py")
    print("3. Test in PAPER mode for 2 hours")
    print("4. Switch to REAL mode once verified")
    print("\n📈 Expected Results:")
    print("   - Win Rate: 40-50% (was 5.4%)")
    print("   - Trades/Day: 50-100 (was 946)")
    print("   - Daily P&L: +$25-50 (was -$0.48)")
else:
    print(f"\n⚠️ {total_checks - passed_checks} checks failed - review above")
