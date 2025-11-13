#!/usr/bin/env python3
"""
Test DexScreener Solana Token Discovery
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_dexscreener_solana():
    """Test DexScreener API for Solana tokens"""
    
    base_url = "https://api.dexscreener.com/latest"
    
    # Test Solana-specific endpoints
    test_endpoints = [
        "/dex/tokens/trending",
        "/dex/search?q=solana",
        "/dex/pairs/solana",
        "/dex/search?chainId=solana&q=usd",
        "/dex/pairs/solana/58oQChx4yWmvKdwLLZzBi4ChoCkp4yaexFkTdLJAp6rT"  # Known Solana pair
    ]
    
    logger.info("🔍 Testing DexScreener API for Solana tokens...")
    
    async with aiohttp.ClientSession() as session:
        for endpoint in test_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                logger.info(f"\nTesting: {endpoint}")
                
                async with session.get(url) as response:
                    logger.info(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            logger.info(f"✅ Found {len(pairs)} pairs")
                            
                            # Show Solana tokens specifically
                            solana_tokens = [p for p in pairs if p.get('chainId', '').lower() == 'solana']
                            logger.info(f"🟡 Solana tokens: {len(solana_tokens)}")
                            
                            # Show sample token
                            if solana_tokens:
                                sample = solana_tokens[0]
                                base_token = sample.get('baseToken', {})
                                logger.info(f"📊 Sample Solana token:")
                                logger.info(f"   Symbol: {base_token.get('symbol', 'N/A')}")
                                logger.info(f"   Address: {base_token.get('address', 'N/A')}")
                                logger.info(f"   Price: ${float(sample.get('priceUsd', 0)):,.6f}")
                                logger.info(f"   Volume 24h: ${float(sample.get('volume', {}).get('h24', 0)):,.0f}")
                        else:
                            logger.info("⚠️ No pairs found")
                            logger.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    else:
                        logger.info(f"❌ Failed with status {response.status}")
                        
            except Exception as e:
                logger.error(f"Error testing {endpoint}: {e}")
    
    logger.info("\n=== Test Complete ===")
    logger.info("If Solana tokens were found, your bot can discover real tradeable tokens!")

if __name__ == "__main__":
    asyncio.run(test_dexscreener_solana())