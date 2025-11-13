#!/usr/bin/env python3
"""
Test Jupiter Ultra API /order endpoint with various pairs
"""
import httpx

JUPITER_ORDER_API = "https://lite-api.jup.ag/ultra/v1/order"

# Token mints
SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
BONK_MINT = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
TRANSFORM_MINT = "FBne7KRJJgFYBUzgBHPEJgU7pzpDzMRwEBNxMC9rPump"
OPTA_MINT = "GYMbJRmB8rRaTmz9MHNu3nRx3jKLDqQeC3s3rLYkpump"

print("="*70)
print("Testing Jupiter Ultra API /order endpoint")
print("="*70)

# Test 1: SOL → USDC (most liquid pair)
print("\n🧪 Test 1: SOL → USDC (0.005 SOL)")
params1 = {
    'inputMint': SOL_MINT,
    'outputMint': USDC_MINT,
    'amount': '5000000'  # 0.005 SOL
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params1, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
        print(f"   Request ID: {data.get('requestId', 'N/A')}")
        print(f"   Has transaction: {'transaction' in data}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: SOL → BONK
print("\n🧪 Test 2: SOL → BONK (0.005 SOL)")
params2 = {
    'inputMint': SOL_MINT,
    'outputMint': BONK_MINT,
    'amount': '5000000'
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params2, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: SOL → TRANSFORM (your failing token)
print("\n🧪 Test 3: SOL → TRANSFORM (0.015 SOL)")
params3 = {
    'inputMint': SOL_MINT,
    'outputMint': TRANSFORM_MINT,
    'amount': '15000000'
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params3, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: SOL → OPTA (your failing token)
print("\n🧪 Test 4: SOL → OPTA (0.005 SOL)")
params4 = {
    'inputMint': SOL_MINT,
    'outputMint': OPTA_MINT,
    'amount': '5000000'
}

try:
    response = httpx.get(JUPITER_ORDER_API, params=params4, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success!")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Try regular Jupiter Quote API (v6) for comparison
print("\n🧪 Test 5: Jupiter Quote API v6 (SOL → USDC for comparison)")
QUOTE_API = "https://quote-api.jup.ag/v6/quote"
params5 = {
    'inputMint': SOL_MINT,
    'outputMint': USDC_MINT,
    'amount': '5000000',
    'slippageBps': '50'
}

try:
    response = httpx.get(QUOTE_API, params=params5, timeout=10.0)
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Success! Quote API v6 still works")
        print(f"   Out Amount: {data.get('outAmount', 'N/A')}")
    else:
        print(f"   ❌ Failed: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("💡 Analysis:")
print("   - If Test 1 (SOL→USDC) works, API is functional")
print("   - If Tests 3-4 fail, TRANSFORM/OPTA might not be supported by Ultra API")
print("   - If Test 5 works, we might need to use Quote API v6 instead")
print("="*70)
