#!/usr/bin/env python3
"""
Test official Dextools library with your original tokens
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.official_dextools_provider import (
    get_dextools_token_price,
    get_dextools_token_info,
    get_official_dextools_provider
)

async def test_official_dextools():
    """Test official Dextools library"""
    print("🛠️ Testing Official Dextools Library")
    print("=" * 50)
    
    # Your original tokens
    your_tokens = {
        'JELLY': '3bC2e2RxcfvF9oP22LvbaNsVwoS2T98q6ErCRoayQYdq',
        'MATRIX': 'AaasmYsdaFLP5ctnWc5TKQZg2yuPpsf6QMAS7xzkT5vm',
        'RYS': 'BuX9TN5doE5hCqpcmqMKYkidXC8zgBK5wHHKujdaAbiQ',
        'AIT': 'GL5ujRvPU3FXJjta88goA6rfHuBn7zKgX1L2LCyJXTw1'
    }
    
    # Also test some known tokens
    known_tokens = {
        'SOL': 'So11111111111111111111111111111111111111112',
        'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    }
    
    working_tokens = []
    
    print("\n📊 Testing Known Tokens First:")
    print("-" * 30)
    
    for symbol, address in known_tokens.items():
        try:
            print(f"\nTesting {symbol} ({address[:8]}...):")
            
            price = await get_dextools_token_price(address, "solana")
            if price:
                print(f"  ✅ Price: ${price}")
                working_tokens.append({'symbol': symbol, 'address': address, 'price': price})
            else:
                print(f"  ❌ No price data")
            
            info = await get_dextools_token_info(address, "solana")
            if info:
                print(f"  ✅ Info: {list(info.keys())[:5]}...")  # Show first 5 keys
            else:
                print(f"  ❌ No info data")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🎯 Testing Your Original Tokens:")
    print("-" * 30)
    
    for symbol, address in your_tokens.items():
        try:
            print(f"\nTesting {symbol} ({address[:8]}...):")
            
            price = await get_dextools_token_price(address, "solana")
            if price:
                print(f"  ✅ Price: ${price}")
                working_tokens.append({'symbol': symbol, 'address': address, 'price': price})
            else:
                print(f"  ❌ No price data")
            
            info = await get_dextools_token_info(address, "solana")
            if info:
                print(f"  ✅ Info available")
                # Extract useful info
                name = info.get('name', 'Unknown')
                symbol_info = info.get('symbol', 'Unknown')
                print(f"    Name: {name}")
                print(f"    Symbol: {symbol_info}")
            else:
                print(f"  ❌ No info data")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n🔥 RESULTS:")
    print("=" * 40)
    
    if working_tokens:
        print(f"✅ Found {len(working_tokens)} working tokens:")
        for token in working_tokens:
            print(f"  {token['symbol']}: {token['address']} - ${token.get('price', 'N/A')}")
        
        print("\n🚨 READY FOR AGGRESSIVE TRADING!")
        print("Official Dextools integration successful.")
        
        return working_tokens
    else:
        print("❌ No tokens found working with Dextools")
        print("This might be due to:")
        print("  - API key required for full functionality")
        print("  - Tokens not tracked by Dextools")
        print("  - Rate limiting on free tier")
        
        return []

async def test_with_free_tier_limitations():
    """Test what we can do with free tier"""
    print(f"\n📋 Free Tier Limitations Test:")
    print("-" * 40)
    
    provider = await get_official_dextools_provider()
    
    print(f"Provider plan: {provider.plan}")
    print(f"Has API key: {provider.api_key is not None}")
    
    # Test with a very common token
    sol_address = "So11111111111111111111111111111111111111112"
    
    try:
        print(f"\nTesting SOL token access:")
        price = await provider.get_token_price(sol_address, "solana")
        
        if price:
            print(f"✅ SOL Price: ${price}")
            print("🎯 Basic price fetching works!")
        else:
            print("❌ SOL price failed - may need API key")
            
    except Exception as e:
        print(f"❌ Free tier test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_official_dextools())
    asyncio.run(test_with_free_tier_limitations())