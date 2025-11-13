import asyncio
import aiohttp
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def find_medium_cap_solana():
    """Find Solana tokens in the 500k-5M range"""
    
    base_url = "https://api.dexscreener.com"
    endpoints = [
        "/latest/dex/search?chainId=solana&q=usd",
        "/latest/dex/search?q=solana", 
        "/latest/dex/search?chainId=solana&q=meme",
        "/latest/dex/search?chainId=solana&q=coin",
        "/latest/dex/search?chainId=solana&q=token",
        "/latest/dex/search?chainId=solana&q=pepe",
        "/latest/dex/search?chainId=solana&q=doge"
    ]
    
    good_tokens = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                logger.info(f"\n🔍 Checking: {endpoint}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            
                            for pair in pairs:
                                chain_id = pair.get('chainId', '').lower()
                                if chain_id != 'solana':
                                    continue
                                
                                base_token = pair.get('baseToken', {})
                                market_cap = float(pair.get('marketCap', 0))
                                volume_24h = float(pair.get('volume', {}).get('h24', 0))
                                liquidity_usd = float(pair.get('liquidity', {}).get('usd', 0))
                                
                                # Look for tokens in 500k-5M range with some activity
                                if (500000 <= market_cap <= 5000000 and 
                                    volume_24h > 100 and 
                                    liquidity_usd > 1000):
                                    
                                    token_info = {
                                        'address': base_token.get('address', ''),
                                        'symbol': base_token.get('symbol', ''),
                                        'name': base_token.get('name', ''),
                                        'market_cap': market_cap,
                                        'volume_24h': volume_24h,
                                        'liquidity_usd': liquidity_usd,
                                        'pair_address': pair.get('pairAddress', '')
                                    }
                                    
                                    # Avoid duplicates
                                    if not any(t['address'] == token_info['address'] for t in good_tokens):
                                        good_tokens.append(token_info)
                                        print(f"✅ {token_info['symbol']} - MC: ${market_cap:,.0f}, Vol: ${volume_24h:,.0f}")
                                        print(f"   Address: {token_info['address']}")
                                        
                                        if len(good_tokens) >= 10:  # Get max 10 tokens
                                            break
                            
                            if len(good_tokens) >= 10:
                                break
                        
            except Exception as e:
                logger.error(f"Error: {e}")
    
    print(f"\n🎯 Found {len(good_tokens)} suitable tokens")
    
    if good_tokens:
        print("\n📋 Updated curated tokens list:")
        for token in good_tokens[:3]:  # Show top 3
            print(f"'{token['address']}': {{")
            print(f"    'symbol': '{token['symbol']}',")
            print(f"    'market_cap': {int(token['market_cap'])},")
            print(f"    'volume_score': 0.7,")
            print(f"    'liquidity_score': 0.7,")
            print(f"    'rugpull_risk': 0.25")
            print(f"}},")

if __name__ == "__main__":
    asyncio.run(find_medium_cap_solana())