import asyncio
import aiohttp
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_dexscreener():
    """Debug DexScreener API to see what data is actually returned"""
    
    base_url = "https://api.dexscreener.com"
    endpoints = [
        "/latest/dex/search?chainId=solana&q=usd",
        "/latest/dex/search?q=solana", 
        "/latest/dex/tokens/trending",
        "/latest/dex/search?chainId=solana&q=meme"
    ]
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                logger.info(f"\n🔍 Testing: {endpoint}")
                
                async with session.get(url) as response:
                    logger.info(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            logger.info(f"Found {len(pairs)} pairs")
                            
                            # Show first few Solana pairs without filtering
                            solana_count = 0
                            for pair in pairs[:20]:  # Only check first 20
                                chain_id = pair.get('chainId', '').lower()
                                if chain_id == 'solana':
                                    solana_count += 1
                                    base_token = pair.get('baseToken', {})
                                    market_cap = float(pair.get('marketCap', 0))
                                    volume_24h = float(pair.get('volume', {}).get('h24', 0))
                                    liquidity_usd = float(pair.get('liquidity', {}).get('usd', 0))
                                    
                                    print(f"  {base_token.get('symbol', 'N/A')} - MC: ${market_cap:,.0f}, Vol: ${volume_24h:,.0f}, Liq: ${liquidity_usd:,.0f}")
                                    
                                    if solana_count >= 5:  # Show max 5 examples
                                        break
                            
                            logger.info(f"Solana pairs found: {solana_count}")
                        else:
                            logger.info("No pairs data in response")
                            if 'schemaVersion' in data:
                                logger.info(f"Schema version: {data['schemaVersion']}")
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        
            except Exception as e:
                logger.error(f"Error with {endpoint}: {e}")

if __name__ == "__main__":
    asyncio.run(debug_dexscreener())