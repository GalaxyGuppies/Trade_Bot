"""
Quick script to verify correct Solana token addresses from Jupiter
"""
import aiohttp
import asyncio

async def verify_jupiter_tokens():
    """Verify WIF and JUP addresses from Jupiter token list"""
    print("🔍 Fetching Jupiter token list...")
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://token.jup.ag/strict"  # Jupiter strict token list
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    tokens = await response.json()
                    print(f"✅ Found {len(tokens)} tokens in Jupiter list\n")
                    
                    # Search for our tokens
                    search_symbols = ['WIF', 'JUP', 'TROLL', 'USELESS', 'BONK']
                    
                    for symbol in search_symbols:
                        found = [t for t in tokens if t.get('symbol', '').upper() == symbol.upper()]
                        if found:
                            for token in found:
                                print(f"✅ {symbol}:")
                                print(f"   Address: {token.get('address')}")
                                print(f"   Name: {token.get('name')}")
                                print(f"   Decimals: {token.get('decimals')}")
                                print()
                        else:
                            print(f"❌ {symbol}: NOT FOUND in Jupiter strict list")
                            print(f"   (May need to use all tokens list or token is not tradeable)\n")
                else:
                    print(f"❌ Failed to fetch token list: {response.status}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")

# Run
asyncio.run(verify_jupiter_tokens())
