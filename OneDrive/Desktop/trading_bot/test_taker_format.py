#!/usr/bin/env python3
"""
Test Jupiter Ultra API taker parameter format
"""
import httpx
import sys

# Your wallet address
WALLET = "GgDZS5HuWPZ58JdyPgfiYqUL98oiThabswPQNdeGJZao"

# Test endpoints
JUPITER_ORDER_API = "https://lite-api.jup.ag/ultra/v1/order"

# Token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"

print("="*60)
print("Testing Jupiter Ultra API taker parameter")
print("="*60)

# Test 1: With taker parameter
print("\n🧪 Test 1: With taker parameter")
params1 = {
    'inputMint': SOL_MINT,
    'outputMint': USDC_MINT,
    'amount': '5000000',  # 0.005 SOL
    'taker': WALLET
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params1, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
    else:
        print(f"   ❌ Failed: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Without taker parameter
print("\n🧪 Test 2: Without taker parameter")
params2 = {
    'inputMint': SOL_MINT,
    'outputMint': USDC_MINT,
    'amount': '5000000'
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params2, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
        print(f"   Has transaction: {'transaction' in data}")
    else:
        print(f"   ❌ Failed: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Check if transaction field exists
print("\n🧪 Test 3: Checking response structure without taker")
params3 = {
    'inputMint': SOL_MINT,
    'outputMint': USDC_MINT,
    'amount': '10000000'  # 0.01 SOL
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params3, timeout=10.0)
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Response keys: {list(data.keys())}")
        if 'transaction' not in data:
            print(f"   ℹ️ No 'transaction' field - taker might be required for executable order")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("💡 Analysis:")
print("   If Test 1 fails with 'Invalid taker', the wallet address format is wrong")
print("   If Test 2 succeeds, taker might not be required for quote")
print("   If Test 3 shows no transaction field, taker is needed to get executable tx")
print("="*60)
