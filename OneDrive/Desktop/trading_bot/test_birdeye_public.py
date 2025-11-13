#!/usr/bin/env python3
"""
Test Birdeye Public Endpoints
Check if any Birdeye endpoints work without API key
"""

import asyncio
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_birdeye_public():
    """Test various Birdeye endpoints to see which work without API key"""
    
    base_url = "https://public-api.birdeye.so"
    headers = {
        'accept': 'application/json',
        'x-chain': 'solana'
    }
    
    # Test endpoints that might work without API key
    test_endpoints = [
        "/defi/tokenlist?sort_by=v24hUSD&sort_type=desc&offset=0&limit=10",
        "/defi/price?address=So11111111111111111111111111111111111111112",  # SOL
        "/defi/search?keyword=solana",
        "/public/token_trending",
        "/v1/public/token_trending",
        "/defi/history_price?address=So11111111111111111111111111111111111111112&type=1H&time_from=1730358000&time_to=1730444400"
    ]
    
    logger.info("🦅 Testing Birdeye public endpoints...")
    
    async with aiohttp.ClientSession() as session:
        for endpoint in test_endpoints:
            try:
                url = f"{base_url}{endpoint}"
                logger.info(f"\nTesting: {endpoint}")
                
                async with session.get(url, headers=headers) as response:
                    logger.info(f"Status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ SUCCESS! Endpoint works without API key")
                        logger.info(f"Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                        
                        # If we got data, show some sample content
                        if isinstance(data, dict) and 'data' in data:
                            sample_data = data['data']
                            if isinstance(sample_data, list) and sample_data:
                                logger.info(f"Sample item keys: {list(sample_data[0].keys()) if isinstance(sample_data[0], dict) else 'Not a dict'}")
                        
                    elif response.status == 401:
                        logger.info("❌ Requires authentication")
                    elif response.status == 429:
                        logger.info("⚠️ Rate limited")
                    else:
                        logger.info(f"❓ Other status: {response.status}")
                        
            except Exception as e:
                logger.error(f"Error testing {endpoint}: {e}")
    
    logger.info("\n=== Test Complete ===")
    logger.info("If any endpoints worked, we can use them without API key!")
    logger.info("Otherwise, you'll need to get a Birdeye API key from https://birdeye.so/")

if __name__ == "__main__":
    asyncio.run(test_birdeye_public())