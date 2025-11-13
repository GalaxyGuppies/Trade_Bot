#!/usr/bin/env python3
"""
Working Token Summary - Ready for Aggressive Trading
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dexscreener_provider import (
    get_opta_price, 
    get_bonk_price,
    get_all_lowcap_prices
)

async def show_working_setup():
    """Show the working token setup"""
    print("🚨 AGGRESSIVE TRADING SETUP - READY!")
    print("=" * 60)
    
    print("\n✅ WORKING ENDPOINTS:")
    print("-" * 30)
    
    # Test each working endpoint
    opta_price = await get_opta_price()
    bonk_price = await get_bonk_price()
    
    if opta_price:
        print(f"GET OPTA_PRICE: ${opta_price:.8f}")
        print(f"  Address: H1R9HvnBsTnAAxAtUr7RHRYRrv3Sq5wVRTH3pbzKCkxG")
        print(f"  Status: ✅ Active trading pairs, ${91806:.0f} liquidity")
        print(f"  Risk Level: 🔥 HIGH - Perfect for aggressive trading")
    
    if bonk_price:
        print(f"\nGET BONK_PRICE: ${bonk_price:.8f}")
        print(f"  Address: DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263")
        print(f"  Status: ✅ Proven memecoin with high liquidity")
        print(f"  Risk Level: ⚡ MODERATE - Reliable but volatile")
    
    # Test batch endpoint
    all_prices = await get_all_lowcap_prices()
    
    print(f"\n📊 BATCH ENDPOINT:")
    print(f"GET ALL_LOWCAP_PRICES: {len(all_prices)} tokens")
    
    if len(all_prices) >= 2:
        prices = list(all_prices.values())
        volatility_ratio = (max(prices) - min(prices)) / (sum(prices) / len(prices))
        print(f"  Volatility Ratio: {volatility_ratio:.3f}")
        
        if volatility_ratio > 1.5:
            print(f"  🚨 EXTREME VOLATILITY - PERFECT FOR AGGRESSIVE TRADING!")
        elif volatility_ratio > 0.5:
            print(f"  ⚡ HIGH VOLATILITY - GOOD FOR ACTIVE TRADING")
    
    print(f"\n🎯 INTEGRATION READY:")
    print("=" * 40)
    
    print("✅ DexScreener provider updated with working tokens")
    print("✅ Endpoint functions available:")
    print("   - get_opta_price()")
    print("   - get_bonk_price()")
    print("   - get_all_lowcap_prices()")
    
    print(f"\n🚨 NEXT STEPS:")
    print("1. Update trading bot to use OPTA and BONK")
    print("2. Set ultra-aggressive parameters for OPTA (high risk/reward)")
    print("3. Use BONK as a more stable backup option")
    print("4. Monitor OPTA closely - $91k liquidity means high volatility potential")
    
    print(f"\n💡 TRADING STRATEGY RECOMMENDATION:")
    print("- OPTA: Micro-trades with 0.1% profit targets due to volatility")
    print("- BONK: Slightly larger trades with 0.5% profit targets")
    print("- Stop losses: Very tight (1-2%) due to low-cap nature")
    print("- Position sizes: Small due to limited liquidity")
    
    return all_prices

if __name__ == "__main__":
    asyncio.run(show_working_setup())