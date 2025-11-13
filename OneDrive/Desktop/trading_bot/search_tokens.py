#!/usr/bin/env python3
"""
Search for low-cap tokens using Jupiter API and other methods
"""

import asyncio
import aiohttp
import json

async def search_jupiter_tokens():
    """Search Jupiter token list for our tokens"""
    print("🔍 Searching Jupiter Token List")
    print("=" * 40)
    
    try:
        # Jupiter token list API
        url = "https://token.jup.ag/all"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    tokens = await response.json()
                    print(f"Found {len(tokens)} tokens on Jupiter")
                    
                    # Search for our tokens
                    target_addresses = {
                        'EuAKdgnHKKe9ETSx7wd4NVb3EhbUptFemRrLdQaSdrkH': 'CLASH',
                        '9ViX1VductEoC2wERTSp2TuDxXPwAf69aeET8ENPJpsN': 'AURA',
                        'BJsSymifLkwUq8r3KMSiFH6t5JbLLBd1VJDgzD7qYNxF': 'BELIEVE'
                    }
                    
                    found_tokens = []
                    for token in tokens:
                        if token.get('address') in target_addresses:
                            found_tokens.append(token)
                            print(f"✅ Found {target_addresses[token['address']]}: {token}")
                    
                    if not found_tokens:
                        print("❌ No target tokens found in Jupiter list")
                        
                        # Search by symbol names
                        print("\n🔍 Searching by symbol names:")
                        for token in tokens:
                            symbol = token.get('symbol', '').upper()
                            if symbol in ['CLASH', 'AURA', 'BELIEVE']:
                                print(f"🎯 Found {symbol}: {token}")
                
    except Exception as e:
        print(f"❌ Error searching Jupiter: {e}")

async def test_birdeye_api():
    """Test Birdeye API for token data"""
    print("\n🐦 Testing Birdeye API")
    print("=" * 40)
    
    tokens = {
        'CLASH': 'EuAKdgnHKKe9ETSx7wd4NVb3EhbUptFemRrLdQaSdrkH',
        'AURA': '9ViX1VductEoC2wERTSp2TuDxXPwAf69aeET8ENPJpsN',
        'BELIEVE': 'BJsSymifLkwUq8r3KMSiFH6t5JbLLBd1VJDgzD7qYNxF'
    }
    
    for name, address in tokens.items():
        try:
            url = f"https://public-api.birdeye.so/public/price?address={address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    print(f"{name}: Status {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"  Data: {data}")
                    else:
                        text = await response.text()
                        print(f"  Error: {text[:100]}")
                        
        except Exception as e:
            print(f"❌ {name} error: {e}")

async def search_dexscreener_by_name():
    """Search DexScreener by token names"""
    print("\n🔍 Searching DexScreener by token names")
    print("=" * 40)
    
    token_names = ['clash', 'aura', 'believe']
    
    for name in token_names:
        try:
            url = f"https://api.dexscreener.com/latest/dex/search/?q={name}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        print(f"{name.upper()}: Found {len(pairs)} pairs")
                        
                        # Filter for Solana pairs
                        solana_pairs = [p for p in pairs if p.get('chainId') == 'solana']
                        print(f"  Solana pairs: {len(solana_pairs)}")
                        
                        for pair in solana_pairs[:3]:  # Show first 3
                            print(f"    {pair.get('baseToken', {}).get('symbol')} - ${pair.get('priceUsd', 'N/A')}")
                    
        except Exception as e:
            print(f"❌ Error searching {name}: {e}")

async def main():
    await search_jupiter_tokens()
    await test_birdeye_api()
    await search_dexscreener_by_name()

if __name__ == "__main__":
    asyncio.run(main())