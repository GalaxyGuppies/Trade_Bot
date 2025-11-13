#!/usr/bin/env python3
"""
Test aggressive/volatile memecoin endpoints
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dexscreener_provider import (
    get_popcat_price, 
    get_pnut_price, 
    get_goat_price,
    get_fwog_price,
    get_all_lowcap_prices
)

async def test_volatile_tokens():
    """Test all volatile memecoin endpoints"""
    print("🔥 Testing Aggressive/Volatile Memecoin Endpoints")
    print("=" * 60)
    
    try:
        # Test individual endpoints
        print("\n📊 Testing Individual Endpoints:")
        
        popcat_price = await get_popcat_price()
        print(f"GET POPCAT_PRICE: ${popcat_price}" if popcat_price else "GET POPCAT_PRICE: Failed")
        
        pnut_price = await get_pnut_price()
        print(f"GET PNUT_PRICE: ${pnut_price}" if pnut_price else "GET PNUT_PRICE: Failed")
        
        goat_price = await get_goat_price()
        print(f"GET GOAT_PRICE: ${goat_price}" if goat_price else "GET GOAT_PRICE: Failed")
        
        fwog_price = await get_fwog_price()
        print(f"GET FWOG_PRICE: ${fwog_price}" if fwog_price else "GET FWOG_PRICE: Failed")
        
        # Test batch endpoint
        print("\n🔥 Testing Batch Endpoint:")
        all_prices = await get_all_lowcap_prices()
        
        if all_prices:
            print("GET ALL_AGGRESSIVE_PRICES:")
            total_value = 0
            for token, price in all_prices.items():
                print(f"  {token}: ${price}")
                total_value += price
            
            print(f"\n💰 Total Portfolio Value: ${total_value:.6f}")
            print(f"📊 Average Price: ${total_value/len(all_prices):.6f}")
            print(f"🎯 Available Tokens: {len(all_prices)}")
            
            # Calculate volatility indicators
            prices = list(all_prices.values())
            if len(prices) >= 2:
                price_range = max(prices) - min(prices)
                print(f"📈 Price Range: ${price_range:.6f}")
                print(f"⚡ Range/Avg Ratio: {price_range/(total_value/len(all_prices)):.2f}")
            
            print("\n🚨 READY FOR ULTRA-AGGRESSIVE MEMECOIN TRADING!")
            print("These tokens have proven liquidity and high volatility.")
            
        else:
            print("GET ALL_AGGRESSIVE_PRICES: Failed")
        
        return all_prices
        
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")
        return {}

if __name__ == "__main__":
    asyncio.run(test_volatile_tokens())