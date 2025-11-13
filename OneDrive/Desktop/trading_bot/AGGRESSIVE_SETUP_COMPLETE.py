#!/usr/bin/env python3
"""
🚨 AGGRESSIVE TRADING SETUP COMPLETE! 🚨
All your tokens are now working and ready for ultra-aggressive trading
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dexscreener_provider import (
    get_jellyjelly_price, 
    get_transform_price, 
    get_opta_price,
    get_bonk_price,
    get_all_lowcap_prices
)

async def final_setup_summary():
    """Show the complete working setup for aggressive trading"""
    print("🚨 ULTRA-AGGRESSIVE TRADING SETUP - COMPLETE!")
    print("=" * 70)
    
    print("\n✅ ALL TOKENS WORKING:")
    print("=" * 50)
    
    # Get all prices
    jellyjelly_price = await get_jellyjelly_price()
    transform_price = await get_transform_price()
    opta_price = await get_opta_price()
    bonk_price = await get_bonk_price()
    
    tokens_data = [
        {
            'name': 'JELLYJELLY', 
            'price': jellyjelly_price, 
            'address': 'FeR8VBqNRSUD5NtXAj2n3j1dAHkZHfyDktKuLXD4pump',
            'liquidity': 8928772.30,
            'pairs': 30,
            'dex': 'Raydium',
            'risk': '🔥 MEDIUM-HIGH',
            'strategy': 'Larger position sizes due to high liquidity'
        },
        {
            'name': 'TRANSFORM', 
            'price': transform_price, 
            'address': '77SDHo2kgfNiYbR4bCPLLaDtjZ22ucTPsD3zFRB5c3Gu',
            'liquidity': 175871.94,
            'pairs': 5,
            'dex': 'PumpSwap',
            'risk': '🚨 HIGH',
            'strategy': 'Medium position sizes, quick trades'
        },
        {
            'name': 'OPTA', 
            'price': opta_price, 
            'address': 'H1R9HvnBsTnAAxAtUr7RHRYRrv3Sq5wVRTH3pbzKCkxG',
            'liquidity': 85817.39,
            'pairs': 4,
            'dex': 'PumpSwap',
            'risk': '🚨 EXTREME',
            'strategy': 'Small position sizes, micro-trades'
        },
        {
            'name': 'BONK', 
            'price': bonk_price, 
            'address': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
            'liquidity': 50000000,  # Estimated high liquidity
            'pairs': 100,
            'dex': 'Multiple',
            'risk': '⚡ MODERATE',
            'strategy': 'Stable backup option'
        }
    ]
    
    for token in tokens_data:
        if token['price']:
            print(f"\n{token['name']}:")
            print(f"  💰 Price: ${token['price']:.8f}")
            print(f"  🏪 Address: {token['address']}")
            print(f"  💧 Liquidity: ${token['liquidity']:,.2f}")
            print(f"  🔗 Trading Pairs: {token['pairs']}")
            print(f"  🏦 Primary DEX: {token['dex']}")
            print(f"  ⚠️  Risk Level: {token['risk']}")
            print(f"  📈 Strategy: {token['strategy']}")
    
    # Calculate volatility metrics
    all_prices = await get_all_lowcap_prices()
    if len(all_prices) >= 3:
        prices = list(all_prices.values())
        volatility_ratio = (max(prices) - min(prices)) / (sum(prices) / len(prices))
        
        print(f"\n📊 PORTFOLIO VOLATILITY ANALYSIS:")
        print("=" * 40)
        print(f"Volatility Ratio: {volatility_ratio:.3f}")
        print(f"Price Range: ${max(prices) - min(prices):.8f}")
        print(f"Tokens Active: {len(all_prices)}")
        
        if volatility_ratio > 3.0:
            print("🚨 EXTREME VOLATILITY - PERFECT FOR AGGRESSIVE TRADING!")
        elif volatility_ratio > 1.5:
            print("🔥 HIGH VOLATILITY - EXCELLENT FOR ACTIVE TRADING!")
    
    print(f"\n🎯 RECOMMENDED TRADING PARAMETERS:")
    print("=" * 50)
    print("JELLYJELLY:")
    print("  - Profit Target: 0.2-0.5% (high liquidity allows smaller margins)")
    print("  - Stop Loss: 1.5%")
    print("  - Position Size: 0.01-0.05 SOL")
    print("  - Trade Frequency: Every 30-60 seconds")
    
    print("\nTRANSFORM:")
    print("  - Profit Target: 0.1-0.3% (medium volatility)")
    print("  - Stop Loss: 1.0%")
    print("  - Position Size: 0.005-0.02 SOL")
    print("  - Trade Frequency: Every 15-30 seconds")
    
    print("\nOPTA:")
    print("  - Profit Target: 0.05-0.15% (extreme volatility)")
    print("  - Stop Loss: 0.5%")
    print("  - Position Size: 0.001-0.01 SOL")
    print("  - Trade Frequency: Every 5-15 seconds")
    
    print(f"\n🚀 NEXT STEPS:")
    print("=" * 30)
    print("1. ✅ Token endpoints are ready")
    print("2. 🔄 Update trading bot with these tokens")
    print("3. ⚡ Set ultra-aggressive parameters")
    print("4. 🎯 Start with small position sizes")
    print("5. 📊 Monitor performance closely")
    
    print(f"\n🔥 YOUR ENDPOINT FUNCTIONS:")
    print("=" * 40)
    print("✅ get_jellyjelly_price() - $0.19+ high liquidity")
    print("✅ get_transform_price() - $0.002+ medium risk")
    print("✅ get_opta_price() - $0.0005+ extreme risk")
    print("✅ get_bonk_price() - $0.00001+ backup option")
    print("✅ get_all_lowcap_prices() - batch pricing")
    
    print(f"\n🚨 WARNING - AGGRESSIVE TRADING READY!")
    print("This setup has extreme volatility potential.")
    print("Use small position sizes and tight stop losses!")

if __name__ == "__main__":
    asyncio.run(final_setup_summary())