import asyncio
import aiohttp

async def find_one_more():
    async with aiohttp.ClientSession() as session:
        urls = [
            "https://api.dexscreener.com/latest/dex/search?chainId=solana&q=pump",
            "https://api.dexscreener.com/latest/dex/search?chainId=solana&q=cat", 
            "https://api.dexscreener.com/latest/dex/search?chainId=solana&q=moon"
        ]
        
        for url in urls:
            try:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'pairs' in data:
                            for pair in data['pairs']:
                                if pair.get('chainId', '').lower() == 'solana':
                                    mc = float(pair.get('marketCap', 0))
                                    vol = float(pair.get('volume', {}).get('h24', 0))
                                    liq = float(pair.get('liquidity', {}).get('usd', 0))
                                    if 500000 <= mc <= 3000000 and vol > 100 and liq > 1000:
                                        token = pair.get('baseToken', {})
                                        print(f"✅ {token.get('symbol', 'N/A')} - MC: ${mc:,.0f}")
                                        print(f"   Address: {token.get('address', '')}")
                                        return
            except: pass

asyncio.run(find_one_more())