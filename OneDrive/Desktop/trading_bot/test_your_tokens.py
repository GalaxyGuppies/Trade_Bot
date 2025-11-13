#!/usr/bin/env python3
"""
Test your specific tokens: JELLYJELLY, TRANSFORM, OPTA using DexScreener
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

async def test_your_specific_tokens():
    """Test your specific token endpoints with DexScreener"""
    print("🎯 Testing Your Specific Tokens with DexScreener")
    print("=" * 60)
    
    try:
        # Test individual endpoints
        print("\n📊 Testing Individual Token Endpoints:")
        print("-" * 40)
        
        jellyjelly_price = await get_jellyjelly_price()
        if jellyjelly_price:
            print(f"✅ GET JELLYJELLY_PRICE: ${jellyjelly_price:.8f}")
        else:
            print("❌ GET JELLYJELLY_PRICE: Failed")
        
        transform_price = await get_transform_price()
        if transform_price:
            print(f"✅ GET TRANSFORM_PRICE: ${transform_price:.8f}")
        else:
            print("❌ GET TRANSFORM_PRICE: Failed")
        
        opta_price = await get_opta_price()
        if opta_price:
            print(f"✅ GET OPTA_PRICE: ${opta_price:.8f}")
        else:
            print("❌ GET OPTA_PRICE: Failed")
        
        # Test BONK as a control (we know this one works)
        bonk_price = await get_bonk_price()
        if bonk_price:
            print(f"✅ GET BONK_PRICE: ${bonk_price:.8f} (Control)")
        else:
            print("❌ GET BONK_PRICE: Failed (Control)")
        
        # Test batch endpoint
        print("\n🔥 Testing Batch Endpoint:")
        print("-" * 30)
        
        all_prices = await get_all_lowcap_prices()
        
        working_tokens = []
        
        if all_prices:
            print("✅ GET ALL_AGGRESSIVE_PRICES:")
            total_value = 0
            
            for token, price in all_prices.items():
                print(f"  {token}: ${price:.8f}")
                working_tokens.append({'symbol': token, 'price': price})
                total_value += price
            
            if working_tokens:
                print(f"\n🎯 TRADING SUMMARY:")
                print(f"  Working Tokens: {len(working_tokens)}")
                print(f"  Total Value: ${total_value:.8f}")
                print(f"  Average Price: ${total_value/len(working_tokens):.8f}")
                
                # Calculate price range for volatility assessment
                prices = [t['price'] for t in working_tokens]
                price_range = max(prices) - min(prices)
                avg_price = sum(prices) / len(prices)
                volatility_ratio = price_range / avg_price if avg_price > 0 else 0
                
                print(f"  Price Range: ${price_range:.8f}")
                print(f"  Volatility Ratio: {volatility_ratio:.3f}")
                
                if volatility_ratio > 0.5:
                    print("  🚨 HIGH VOLATILITY DETECTED - PERFECT FOR AGGRESSIVE TRADING!")
                elif volatility_ratio > 0.1:
                    print("  ⚡ MODERATE VOLATILITY - GOOD FOR ACTIVE TRADING")
                else:
                    print("  📊 LOW VOLATILITY - CONSIDER MORE AGGRESSIVE PARAMETERS")
            
        else:
            print("❌ GET ALL_AGGRESSIVE_PRICES: Failed")
        
        print(f"\n🎯 ENDPOINT SETUP RESULTS:")
        print("=" * 50)
        
        endpoints_ready = []
        
        if jellyjelly_price:
            endpoints_ready.append("GET JELLYJELLY_PRICE")
        if transform_price:
            endpoints_ready.append("GET TRANSFORM_PRICE")
        if opta_price:
            endpoints_ready.append("GET OPTA_PRICE")
        if bonk_price:
            endpoints_ready.append("GET BONK_PRICE")
        
        if endpoints_ready:
            print(f"✅ {len(endpoints_ready)} endpoints ready:")
            for endpoint in endpoints_ready:
                print(f"  - {endpoint}")
            
            print(f"\n🚨 READY FOR ULTRA-AGGRESSIVE LOW-CAP TRADING!")
            print("You can now use these endpoints in your trading bot.")
            
            return working_tokens
        else:
            print("❌ No endpoints working")
            print("Possible issues:")
            print("  - Tokens may not have active trading pairs")
            print("  - Very new tokens not yet indexed")
            print("  - Low liquidity tokens")
            
            return []
        
    except Exception as e:
        print(f"❌ Error testing endpoints: {e}")
        return []

async def test_direct_api_calls():
    """Test direct API calls to see what's happening"""
    print(f"\n🔍 Direct API Testing:")
    print("=" * 40)
    
    tokens = {
        'JELLYJELLY': 'FeR8VBqNRSUD5NtXAj2n3j1dAHkZHfyDktKuLXD4pump',
        'TRANSFORM': '77SDHo2kgfNiYbR4bCPLLaDtjZ22ucTPsD3zFRB5c3Gu',
        'OPTA': 'H1R9HvnBsTnAAxAtUr7RHRYRrv3Sq5wVRTH3pbzKCkxG'
    }
    
    import aiohttp
    base_url = "https://api.dexscreener.com/latest/dex"
    
    for token_name, address in tokens.items():
        try:
            print(f"\n{token_name} ({address[:8]}...):")
            url = f"{base_url}/tokens/{address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'pairs' in data:
                            pairs = data['pairs']
                            if pairs and len(pairs) > 0:
                                print(f"  ✅ Found {len(pairs)} trading pairs")
                                
                                # Show best pair info
                                best_pair = max(pairs, key=lambda x: x.get('liquidity', {}).get('usd', 0))
                                price = best_pair.get('priceUsd')
                                liquidity = best_pair.get('liquidity', {}).get('usd', 0)
                                
                                if price:
                                    print(f"    Price: ${float(price):.8f}")
                                    print(f"    Liquidity: ${liquidity:,.2f}")
                                    print(f"    DEX: {best_pair.get('dexId', 'Unknown')}")
                                else:
                                    print(f"    ❌ No price in pair data")
                            else:
                                print(f"  ❌ No trading pairs found")
                        else:
                            print(f"  ❌ No pairs data in response")
                    else:
                        print(f"  ❌ API returned status: {response.status}")
                        
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_your_specific_tokens())
    asyncio.run(test_direct_api_calls())