#!/usr/bin/env python3
"""
Debug test for low-cap token addresses
"""

import asyncio
import aiohttp
import json

async def test_token_addresses():
    """Test token addresses directly with DexScreener API"""
    tokens = {
        'CLASH': 'EuAKdgnHKKe9ETSx7wd4NVb3EhbUptFemRrLdQaSdrkH',
        'AURA': '9ViX1VductEoC2wERTSp2TuDxXPwAf69aeET8ENPJpsN',
        'BELIEVE': 'BJsSymifLkwUq8r3KMSiFH6t5JbLLBd1VJDgzD7qYNxF'
    }
    
    base_url = "https://api.dexscreener.com/latest/dex"
    
    for token_name, address in tokens.items():
        print(f"\n🔍 Testing {token_name} ({address}):")
        
        try:
            url = f"{base_url}/tokens/{address}"
            print(f"URL: {url}")
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url) as response:
                    print(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        print(f"Response keys: {list(data.keys()) if data else 'None'}")
                        
                        if data and 'pairs' in data:
                            pairs = data['pairs']
                            print(f"Pairs found: {len(pairs) if pairs else 0}")
                            
                            if pairs and len(pairs) > 0:
                                pair = pairs[0]
                                print(f"First pair: {pair.get('pairAddress', 'No address')}")
                                print(f"Price USD: {pair.get('priceUsd', 'No price')}")
                                print(f"Liquidity: {pair.get('liquidity', {})}")
                            else:
                                print("No pairs available")
                        else:
                            print("No pairs data in response")
                            if data:
                                print(f"Full response: {json.dumps(data, indent=2)}")
                    else:
                        text = await response.text()
                        print(f"Error response: {text[:200]}...")
                        
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_token_addresses())