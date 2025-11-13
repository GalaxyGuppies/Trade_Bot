#!/usr/bin/env python3
"""
Test Dextools API and find real low-cap tokens for aggressive trading
"""

import asyncio
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dextools_provider import (
    get_dextools_provider,
    get_dextools_trending,
    search_dextools_tokens
)

async def test_dextools_connection():
    """Test basic Dextools API connection"""
    print("🛠️ Testing Dextools API Connection")
    print("=" * 50)
    
    provider = await get_dextools_provider()
    
    # Test with a known Solana token (BONK)
    bonk_address = "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"
    
    try:
        print(f"Testing with BONK token ({bonk_address[:8]}...):")
        
        # Test price fetch
        price = await provider.get_token_price(bonk_address)
        if price:
            print(f"✅ Price fetch successful: ${price:.8f}")
        else:
            print("❌ Price fetch failed")
        
        # Test token info
        info = await provider.get_token_info(bonk_address)
        if info:
            print(f"✅ Token info successful: {list(info.keys())}")
        else:
            print("❌ Token info failed")
        
        return price is not None or info is not None
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

async def find_trending_lowcap_tokens():
    """Find trending low-cap tokens for aggressive trading"""
    print("\n🔥 Finding Trending Low-Cap Tokens")
    print("=" * 50)
    
    try:
        trending = await get_dextools_trending()
        
        if trending:
            print(f"Found {len(trending)} trending tokens")
            
            viable_tokens = []
            
            for i, token in enumerate(trending[:15]):  # Check first 15
                try:
                    symbol = token.get('symbol', 'Unknown')
                    address = token.get('id', token.get('address', ''))
                    
                    if not address:
                        continue
                    
                    print(f"\n{i+1}. Checking {symbol} ({address[:8]}...):")
                    
                    # Get detailed info
                    provider = await get_dextools_provider()
                    price = await provider.get_token_price(address)
                    info = await provider.get_token_info(address)
                    
                    if price and info:
                        # Extract key metrics
                        market_cap = info.get('marketCap', 0)
                        liquidity = info.get('liquidity', 0)
                        volume_24h = info.get('volume24h', 0)
                        
                        print(f"   Price: ${price:.8f}")
                        print(f"   Market Cap: ${market_cap:,.0f}")
                        print(f"   Liquidity: ${liquidity:,.0f}")
                        print(f"   24h Volume: ${volume_24h:,.0f}")
                        
                        # Filter for low-cap with decent activity
                        if (market_cap < 10000000 and  # Under $10M market cap
                            liquidity > 5000 and      # At least $5k liquidity
                            volume_24h > 1000):       # Some trading volume
                            
                            vol_liq_ratio = volume_24h / max(liquidity, 1)
                            print(f"   🎯 VIABLE! Vol/Liq Ratio: {vol_liq_ratio:.3f}")
                            
                            viable_tokens.append({
                                'symbol': symbol,
                                'address': address,
                                'price': price,
                                'market_cap': market_cap,
                                'liquidity': liquidity,
                                'volume_24h': volume_24h,
                                'vol_liq_ratio': vol_liq_ratio
                            })
                        else:
                            print("   ❌ Doesn't meet criteria")
                    else:
                        print("   ❌ Failed to get price/info")
                    
                except Exception as e:
                    print(f"   ❌ Error: {e}")
            
            # Sort by volatility potential (volume/liquidity ratio)
            viable_tokens.sort(key=lambda x: x['vol_liq_ratio'], reverse=True)
            
            print(f"\n🎯 FOUND {len(viable_tokens)} VIABLE LOW-CAP TOKENS:")
            print("=" * 80)
            
            for i, token in enumerate(viable_tokens[:8]):  # Show top 8
                print(f"{i+1}. {token['symbol']}")
                print(f"   Address: {token['address']}")
                print(f"   Price: ${token['price']:.8f}")
                print(f"   Market Cap: ${token['market_cap']:,.0f}")
                print(f"   Liquidity: ${token['liquidity']:,.0f}")
                print(f"   Vol/Liq Ratio: {token['vol_liq_ratio']:.3f} ⚡")
                print()
            
            return viable_tokens[:4]  # Return top 4
            
        else:
            print("❌ No trending tokens found")
            return []
            
    except Exception as e:
        print(f"❌ Error finding trending tokens: {e}")
        return []

async def search_specific_tokens():
    """Search for specific token types that might be good for trading"""
    print("\n🔍 Searching for Specific Token Types")
    print("=" * 50)
    
    search_terms = ['meme', 'ai', 'defi', 'gaming', 'new']
    found_tokens = []
    
    for term in search_terms:
        try:
            print(f"\nSearching for '{term}' tokens:")
            results = await search_dextools_tokens(term)
            
            if results:
                print(f"  Found {len(results)} results")
                
                for token in results[:3]:  # Check first 3 from each search
                    symbol = token.get('symbol', 'Unknown')
                    address = token.get('id', token.get('address', ''))
                    
                    if address:
                        print(f"    {symbol}: {address}")
                        found_tokens.append({'symbol': symbol, 'address': address})
            else:
                print(f"  No results for '{term}'")
                
        except Exception as e:
            print(f"  ❌ Error searching '{term}': {e}")
    
    return found_tokens

async def test_your_tokens():
    """Test the tokens you originally wanted to use"""
    print("\n🎯 Testing Your Original Tokens via Dextools")
    print("=" * 50)
    
    your_tokens = {
        'JELLY': '3bC2e2RxcfvF9oP22LvbaNsVwoS2T98q6ErCRoayQYdq',
        'MATRIX': 'AaasmYsdaFLP5ctnWc5TKQZg2yuPpsf6QMAS7xzkT5vm',
        'RYS': 'BuX9TN5doE5hCqpcmqMKYkidXC8zgBK5wHHKujdaAbiQ',
        'AIT': 'GL5ujRvPU3FXJjta88goA6rfHuBn7zKgX1L2LCyJXTw1'
    }
    
    provider = await get_dextools_provider()
    working_tokens = []
    
    for symbol, address in your_tokens.items():
        try:
            print(f"\nTesting {symbol} ({address[:8]}...):")
            
            price = await provider.get_token_price(address)
            info = await provider.get_token_info(address)
            
            if price:
                print(f"  ✅ Price: ${price:.8f}")
                working_tokens.append({'symbol': symbol, 'address': address, 'price': price})
            else:
                print(f"  ❌ No price data")
            
            if info:
                print(f"  ✅ Info available: {list(info.keys())}")
            else:
                print(f"  ❌ No token info")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    return working_tokens

async def main():
    print("🚀 DEXTOOLS LOW-CAP TOKEN DISCOVERY")
    print("=" * 60)
    
    # Test connection first
    connection_ok = await test_dextools_connection()
    
    if not connection_ok:
        print("\n❌ Dextools connection failed. Trying alternative approach...")
        
        # Try your original tokens with Dextools
        working_tokens = await test_your_tokens()
        
        if working_tokens:
            print(f"\n✅ Found {len(working_tokens)} working tokens:")
            for token in working_tokens:
                print(f"  {token['symbol']}: {token['address']}")
        else:
            print("\n❌ Your original tokens don't work with Dextools either")
        
        return working_tokens
    
    # If connection works, find trending tokens
    trending_tokens = await find_trending_lowcap_tokens()
    search_tokens = await search_specific_tokens()
    your_tokens = await test_your_tokens()
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("=" * 60)
    
    all_recommendations = []
    
    if trending_tokens:
        print("🔥 Trending low-cap tokens (best volatility):")
        for token in trending_tokens:
            print(f"  {token['symbol']}: {token['address']}")
            all_recommendations.append(token)
    
    if your_tokens:
        print("\n✅ Your original tokens that work:")
        for token in your_tokens:
            print(f"  {token['symbol']}: {token['address']}")
            all_recommendations.append(token)
    
    if search_tokens:
        print(f"\n🔍 Additional discovered tokens ({len(search_tokens)}):")
        for token in search_tokens[:4]:
            print(f"  {token['symbol']}: {token['address']}")
    
    if all_recommendations:
        print("\n🚨 READY FOR ULTRA-AGGRESSIVE TRADING!")
        print("These tokens should provide the volatility you need.")
        return all_recommendations[:4]  # Return top 4 for trading
    else:
        print("\n❌ Need to find alternative data sources")
        return []

if __name__ == "__main__":
    asyncio.run(main())