#!/usr/bin/env python3
"""
Find real low-cap tokens with active trading on Solana
"""

import asyncio
import aiohttp
import json

async def find_active_lowcap_tokens():
    """Search for real low-cap tokens with active trading"""
    print("🔍 Searching for Active Low-Cap Tokens")
    print("=" * 50)
    
    try:
        # Search for recent Solana tokens with low market cap
        url = "https://api.dexscreener.com/latest/dex/tokens/trending"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            # Try trending tokens first
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Found trending data: {list(data.keys()) if data else 'None'}")
            
            # Search for Solana pairs with specific criteria
            search_url = "https://api.dexscreener.com/latest/dex/search/?q=solana"
            
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    # Filter for Solana pairs with low market cap but active trading
                    solana_pairs = []
                    for pair in pairs:
                        if (pair.get('chainId') == 'solana' and 
                            pair.get('priceUsd') and
                            pair.get('liquidity', {}).get('usd', 0) > 1000 and  # At least $1k liquidity
                            pair.get('volume', {}).get('h24', 0) > 100):  # Some trading volume
                            
                            solana_pairs.append(pair)
                    
                    print(f"Found {len(solana_pairs)} active Solana pairs")
                    
                    # Sort by lowest market cap but with decent liquidity
                    viable_tokens = []
                    for pair in solana_pairs[:20]:  # Check first 20
                        token = pair.get('baseToken', {})
                        liquidity = pair.get('liquidity', {}).get('usd', 0)
                        volume_24h = pair.get('volume', {}).get('h24', 0)
                        price = float(pair.get('priceUsd', 0))
                        
                        # Look for tokens with reasonable trading activity
                        if liquidity < 500000 and volume_24h > 500:  # Under $500k liquidity, some volume
                            viable_tokens.append({
                                'symbol': token.get('symbol', 'Unknown'),
                                'address': token.get('address', ''),
                                'price': price,
                                'liquidity': liquidity,
                                'volume_24h': volume_24h,
                                'dex': pair.get('dexId', 'Unknown'),
                                'pair_address': pair.get('pairAddress', '')
                            })
                    
                    # Sort by volume to liquidity ratio (more volatile = better for aggressive trading)
                    viable_tokens.sort(key=lambda x: x['volume_24h'] / max(x['liquidity'], 1), reverse=True)
                    
                    print(f"\n🎯 Found {len(viable_tokens)} viable low-cap tokens:")
                    print("=" * 80)
                    
                    for i, token in enumerate(viable_tokens[:10]):  # Show top 10
                        vol_liq_ratio = token['volume_24h'] / max(token['liquidity'], 1)
                        print(f"{i+1}. {token['symbol']}")
                        print(f"   Address: {token['address']}")
                        print(f"   Price: ${token['price']:.8f}")
                        print(f"   Liquidity: ${token['liquidity']:,.0f}")
                        print(f"   24h Volume: ${token['volume_24h']:,.0f}")
                        print(f"   Vol/Liq Ratio: {vol_liq_ratio:.3f} (Higher = More Volatile)")
                        print(f"   DEX: {token['dex']}")
                        print()
                    
                    if viable_tokens:
                        print("🔥 TOP 4 RECOMMENDATIONS FOR AGGRESSIVE TRADING:")
                        top_4 = viable_tokens[:4]
                        for token in top_4:
                            print(f"{token['symbol']}: {token['address']}")
                        
                        return top_4
                    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    return []

async def test_known_memecoins():
    """Test some known Solana memecoins that should have trading"""
    print("\n🐕 Testing Known Solana Memecoins")
    print("=" * 50)
    
    # These are known Solana tokens that should have active trading
    known_tokens = {
        'POPCAT': '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',
        'PNUT': '2qEHjDLDLbuBgRYvsxhc5D6uDWAivNFZGan56P1tpump',
        'GOAT': 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump',
        'MEW': 'MEW1gQWJ3nEXg2qgERiKu7FAFj79PHvQVREQUzScPP5',
    }
    
    successful_tokens = []
    
    for symbol, address in known_tokens.items():
        try:
            url = f"https://api.dexscreener.com/latest/dex/tokens/{address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            best_pair = max(pairs, 
                                          key=lambda x: x.get('liquidity', {}).get('usd', 0))
                            
                            price = best_pair.get('priceUsd')
                            liquidity = best_pair.get('liquidity', {}).get('usd', 0)
                            volume_24h = best_pair.get('volume', {}).get('h24', 0)
                            
                            print(f"✅ {symbol}: ${float(price):.8f} (Liq: ${liquidity:,.0f}, Vol: ${volume_24h:,.0f})")
                            
                            successful_tokens.append({
                                'symbol': symbol,
                                'address': address,
                                'price': float(price),
                                'liquidity': liquidity,
                                'volume_24h': volume_24h
                            })
                        else:
                            print(f"❌ {symbol}: No pairs found")
                    else:
                        print(f"❌ {symbol}: API error {response.status}")
                        
        except Exception as e:
            print(f"❌ {symbol}: Error {e}")
    
    return successful_tokens

async def main():
    lowcap_tokens = await find_active_lowcap_tokens()
    memecoin_tokens = await test_known_memecoins()
    
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    print("=" * 50)
    
    if lowcap_tokens:
        print("🔥 Low-cap tokens for aggressive trading:")
        for token in lowcap_tokens[:4]:
            print(f"  {token['symbol']}: {token['address']}")
    
    if memecoin_tokens:
        print("\n🐕 Working memecoins (proven liquidity):")
        for token in memecoin_tokens:
            print(f"  {token['symbol']}: {token['address']}")
    
    if lowcap_tokens or memecoin_tokens:
        print("\n✅ Ready to update trading bot with working tokens!")
    else:
        print("\n❌ Need to find alternative token sources")

if __name__ == "__main__":
    asyncio.run(main())