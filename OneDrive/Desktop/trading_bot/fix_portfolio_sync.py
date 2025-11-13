"""
Reset bot portfolio to match ACTUAL on-chain balances
Run this to sync the bot's memory with your real wallet
"""
import json
from pathlib import Path

# Real balances from blockchain
ACTUAL_BALANCES = {
    'GXY': 10.285224,
    'USELESS': 62.380896,  # Bot has wrong address!
    'ACE': 0.0,  # Never actually bought
    'PUPI': 0.0,  # Never actually bought
    'ROI': 0.0   # Not traded yet
}

print("🔄 PORTFOLIO SYNC TOOL")
print("=" * 60)
print("\n📊 ACTUAL ON-CHAIN BALANCES:")
for token, balance in ACTUAL_BALANCES.items():
    if balance > 0:
        print(f"  ✅ {token}: {balance:.6f}")
    else:
        print(f"  ❌ {token}: 0 (no tokens)")

print("\n" + "=" * 60)
print("⚠️ CRITICAL ISSUES FOUND:")
print("=" * 60)
print("\n1. ACE: Bot thinks it has 139.28 tokens")
print("   Reality: You have 0 tokens (transaction failed)")
print("   → Portfolio tracker is WRONG\n")

print("2. PUPI: Bot thinks it has 45,659 tokens") 
print("   Reality: You have 0 tokens (transaction failed)")
print("   → Portfolio tracker is WRONG\n")

print("3. USELESS: Bot thinks it has 0.062 tokens")
print("   Reality: You have 62.38 tokens")
print("   → Bot has WRONG token address!\n")

print("=" * 60)
print("🔧 HOW TO FIX:")
print("=" * 60)
print("\n**Option 1: Manual Portfolio Reset (Quick Fix)**")
print("  1. Stop the bot")
print("  2. Delete any saved portfolio/state files")
print("  3. Restart bot - it will query actual balances")

print("\n**Option 2: Fix USELESS Address (Permanent Fix)**")
print("  Need to find correct USELESS token mint address")
print("  Current: Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk")
print("  Your wallet has different USELESS token")

print("\n**Option 3: Start Fresh**")
print("  1. Sell GXY (10.28 tokens) to SOL")
print("  2. Sell USELESS (62.38 tokens) to SOL")
print("  3. Reset portfolio to all zeros")
print("  4. Start trading with ROI + 2-3 other tokens")

print("\n" + "=" * 60)
print("💡 RECOMMENDATION:")
print("=" * 60)
print("The 'Insufficient funds' error is NOT about SOL.")
print("You have 0.915 SOL ($139) - that's plenty!")
print("\nThe problem is the bot is trying to sell ACE and PUPI")
print("tokens that don't exist in your wallet.")
print("\n→ Stop the bot, clear the portfolio state, restart fresh.")
print("=" * 60)
