#!/usr/bin/env python3
"""
Test script for new low-cap DeFi token endpoints
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dexscreener_provider import (
    get_clash_price, 
    get_aura_price, 
    get_believe_price,
    get_all_lowcap_prices
)

async def test_lowcap_endpoints():
    """Test all low-cap token endpoints"""
    print("🚀 Testing Low-Cap DeFi Token Endpoints")
    print("=" * 50)
    
    try:
        # Test individual endpoints
        print("\n📊 Testing Individual Endpoints:")
        
        clash_price = await get_clash_price()
        print(f"GET CLASH_PRICE: ${clash_price}" if clash_price else "GET CLASH_PRICE: Failed")
        
        aura_price = await get_aura_price()
        print(f"GET AURA_PRICE: ${aura_price}" if aura_price else "GET AURA_PRICE: Failed")
        
        believe_price = await get_believe_price()
        print(f"GET BELIEVE_PRICE: ${believe_price}" if believe_price else "GET BELIEVE_PRICE: Failed")
        
        # Test batch endpoint
        print("\n🔥 Testing Batch Endpoint:")
        all_prices = await get_all_lowcap_prices()
        
        if all_prices:
            print("GET ALL_LOWCAP_PRICES:")
            for token, price in all_prices.items():
                print(f"  {token}: ${price}")
        else:
            print("GET ALL_LOWCAP_PRICES: Failed")
        
        # Calculate potential volatility indicators
        if all_prices and len(all_prices) >= 2:
            prices = list(all_prices.values())
            avg_price = sum(prices) / len(prices)
            print(f"\n💰 Average Price: ${avg_price:.8f}")
            print("⚡ Ready for aggressive low-cap trading!")
        
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")

if __name__ == "__main__":
    asyncio.run(test_lowcap_endpoints())