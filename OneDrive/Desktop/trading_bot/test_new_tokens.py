#!/usr/bin/env python3
"""
Test script for new low-cap DeFi token endpoints: JELLY, MATRIX, RYS, AIT
"""

import asyncio
import aiohttp
import json

async def test_tokens_directly():
    """Test token addresses directly with DexScreener API"""
    tokens = {
        'JELLY': '3bC2e2RxcfvF9oP22LvbaNsVwoS2T98q6ErCRoayQYdq',
        'MATRIX': 'AaasmYsdaFLP5ctnWc5TKQZg2yuPpsf6QMAS7xzkT5vm',
        'RYS': 'BuX9TN5doE5hCqpcmqMKYkidXC8zgBK5wHHKujdaAbiQ',
        'AIT': 'GL5ujRvPU3FXJjta88goA6rfHuBn7zKgX1L2LCyJXTw1'
    }
    
    print("🚀 Testing New Low-Cap DeFi Tokens")
    print("=" * 50)
    
    base_url = "https://api.dexscreener.com/latest/dex"
    successful_tokens = []
    
    for token_name, address in tokens.items():
        print(f"\n🔍 Testing {token_name} ({address[:8]}...):")
        
        try:
            url = f"{base_url}/tokens/{address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            print(f"  ✅ Found {len(pairs)} trading pairs")
                            
                            # Get best pair by liquidity
                            valid_pairs = [p for p in pairs if p.get('priceUsd')]
                            if valid_pairs:
                                best_pair = max(valid_pairs, 
                                              key=lambda x: x.get('liquidity', {}).get('usd', 0))
                                
                                price = best_pair.get('priceUsd')
                                liquidity = best_pair.get('liquidity', {}).get('usd', 0)
                                volume_24h = best_pair.get('volume', {}).get('h24', 0)
                                
                                print(f"  💰 Price: ${float(price):.8f}")
                                print(f"  💧 Liquidity: ${liquidity:,.2f}")
                                print(f"  📊 24h Volume: ${volume_24h:,.2f}")
                                print(f"  🏪 DEX: {best_pair.get('dexId', 'Unknown')}")
                                
                                successful_tokens.append({
                                    'name': token_name,
                                    'address': address,
                                    'price': float(price),
                                    'liquidity': liquidity,
                                    'volume_24h': volume_24h
                                })
                            else:
                                print("  ❌ No valid pairs with price data")
                        else:
                            print("  ❌ No trading pairs found")
                    else:
                        print(f"  ❌ API error: {response.status}")
                        
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print(f"\n🎯 SUMMARY: {len(successful_tokens)}/{len(tokens)} tokens have active trading")
    if successful_tokens:
        print("\n🔥 SUCCESSFUL TOKENS FOR AGGRESSIVE TRADING:")
        for token in successful_tokens:
            print(f"  {token['name']}: ${token['price']:.8f} (Liq: ${token['liquidity']:,.0f})")
        
        # Calculate volatility potential
        avg_liquidity = sum(t['liquidity'] for t in successful_tokens) / len(successful_tokens)
        print(f"\n⚡ Average Liquidity: ${avg_liquidity:,.2f}")
        if avg_liquidity < 100000:  # Low liquidity = high volatility potential
            print("🚨 LOW LIQUIDITY DETECTED - HIGH VOLATILITY POTENTIAL!")
        
        return successful_tokens
    else:
        print("❌ No tokens available for trading")
        return []

async def test_endpoints():
    """Test our updated endpoint functions"""
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from data.dexscreener_provider import (
            get_jelly_price, 
            get_matrix_price, 
            get_rys_price,
            get_ait_price,
            get_all_lowcap_prices
        )
        
        print("\n📊 Testing Endpoint Functions:")
        print("-" * 30)
        
        # Test individual endpoints
        jelly_price = await get_jelly_price()
        print(f"GET JELLY_PRICE: ${jelly_price}" if jelly_price else "GET JELLY_PRICE: Failed")
        
        matrix_price = await get_matrix_price()
        print(f"GET MATRIX_PRICE: ${matrix_price}" if matrix_price else "GET MATRIX_PRICE: Failed")
        
        rys_price = await get_rys_price()
        print(f"GET RYS_PRICE: ${rys_price}" if rys_price else "GET RYS_PRICE: Failed")
        
        ait_price = await get_ait_price()
        print(f"GET AIT_PRICE: ${ait_price}" if ait_price else "GET AIT_PRICE: Failed")
        
        # Test batch endpoint
        all_prices = await get_all_lowcap_prices()
        if all_prices:
            print(f"\nGET ALL_LOWCAP_PRICES: {len(all_prices)} tokens")
            for token, price in all_prices.items():
                print(f"  {token}: ${price}")
        else:
            print("GET ALL_LOWCAP_PRICES: Failed")
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Endpoint test error: {e}")

async def main():
    successful_tokens = await test_tokens_directly()
    if successful_tokens:
        await test_endpoints()
        
        print("\n🎯 READY FOR AGGRESSIVE LOW-CAP TRADING!")
        print("These tokens should provide high volatility for better trading opportunities.")

if __name__ == "__main__":
    asyncio.run(main())