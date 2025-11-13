#!/usr/bin/env python3
"""
Debug script to test token discovery directly
"""

import asyncio
import json
import logging
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_discovery():
    """Test the token discovery system"""
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    logger.info("🔍 Starting token discovery test...")
    
    # Initialize analyzer
    analyzer = AlternativeBlockchainAnalyzer(config)
    
    # Test scalping targets discovery
    logger.info("⚡ Testing scalping targets discovery...")
    try:
        scalping_tokens = await analyzer.search_scalping_targets(
            min_volume=config['thresholds']['min_volume']
        )
        logger.info(f"✅ Found {len(scalping_tokens)} scalping targets")
        
        for i, token in enumerate(scalping_tokens[:5]):  # Show first 5
            logger.info(f"   {i+1}. {token.get('symbol', 'UNKNOWN')} - MC: ${token.get('market_cap', 0):,.0f}, Vol: ${token.get('volume_24h', 0):,.0f}")
            
    except Exception as e:
        logger.error(f"❌ Scalping discovery failed: {e}")
    
    # Test mixed discovery
    logger.info("🔄 Testing mixed discovery...")
    try:
        mixed_tokens = await analyzer.find_trending_tokens(max_count=20)
        logger.info(f"✅ Found {len(mixed_tokens)} mixed targets")
        
        for i, token in enumerate(mixed_tokens[:5]):  # Show first 5
            logger.info(f"   {i+1}. {token.get('symbol', 'UNKNOWN')} - MC: ${token.get('market_cap', 0):,.0f}, Source: {token.get('discovery_source', 'UNKNOWN')}")
            
    except Exception as e:
        logger.error(f"❌ Mixed discovery failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_discovery())