#!/usr/bin/env python3
"""
Test Birdeye API Integration with Authentication
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.birdeye_provider import BirdeyeProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_birdeye_auth():
    """Test Birdeye API with authentication"""
    
    # Load config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        birdeye_api_key = config.get('api_keys', {}).get('birdeye')
    except:
        logger.error("Could not load config.json")
        birdeye_api_key = None
    
    logger.info("🦅 Testing Birdeye API Integration")
    logger.info(f"API Key Available: {'Yes' if birdeye_api_key and birdeye_api_key != 'YOUR_BIRDEYE_API_KEY_HERE' else 'No'}")
    
    # Test without API key first
    logger.info("\n=== Testing WITHOUT API Key ===")
    provider_no_key = BirdeyeProvider()
    provider_with_key = None
    
    # Test trending tokens
    trending = await provider_no_key.get_trending_tokens(limit=5)
    logger.info(f"Trending tokens (no key): {len(trending) if trending else 0} found")
    
    # Test with API key if available
    if birdeye_api_key and birdeye_api_key != 'YOUR_BIRDEYE_API_KEY_HERE':
        logger.info("\n=== Testing WITH API Key ===")
        provider_with_key = BirdeyeProvider(api_key=birdeye_api_key)
        
        # Test trending tokens with auth
        trending_auth = await provider_with_key.get_trending_tokens(limit=5)
        logger.info(f"Trending tokens (with key): {len(trending_auth) if trending_auth else 0} found")
        
        if trending_auth:
            logger.info("✅ Birdeye API working with authentication!")
            for i, token in enumerate(trending_auth[:3]):
                logger.info(f"  {i+1}. {token.get('symbol', 'N/A')} - ${token.get('price', 0):.6f}")
        else:
            logger.warning("⚠️ No tokens returned with API key")
    else:
        logger.info("\n=== API Key Setup ===")
        logger.info("To get full Birdeye functionality:")
        logger.info("1. Sign up at https://birdeye.so/")
        logger.info("2. Get your API key")
        logger.info("3. Replace 'YOUR_BIRDEYE_API_KEY_HERE' in config.json")
        logger.info("4. Run this test again")
    
    logger.info("\n=== Testing Token Security Analysis ===")
    # Test with a known token (PEPE)
    pepe_address = "0x6982508145454Ce325dDbE47a25d4ec3d2311933"
    active_provider = provider_with_key if provider_with_key else provider_no_key
    security_data = await active_provider.get_token_security(pepe_address)
    
    if security_data:
        logger.info("✅ Token security analysis working")
        logger.info(f"Security data keys: {list(security_data.keys())}")
    else:
        logger.info("ℹ️ Security analysis requires authentication or token not found")

if __name__ == "__main__":
    asyncio.run(test_birdeye_auth())