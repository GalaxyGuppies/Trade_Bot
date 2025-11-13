#!/usr/bin/env python3
"""
Test script for token discovery functionality
"""

import asyncio
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src path
sys.path.append('src')

async def test_token_discovery():
    """Test the token discovery system"""
    try:
        from src.data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer
        
        print("📊 Testing Alternative Blockchain Analyzer...")
        logger.info("Initializing analyzer...")
        
        analyzer = AlternativeBlockchainAnalyzer()
        print("✅ Analyzer initialized")
        
        print("🔍 Searching for trending tokens...")
        tokens = await analyzer.search_trending_tokens(min_volume=50000, min_liquidity=25000)
        
        print(f"✅ Search completed: Found {len(tokens)} tokens")
        logger.info(f"Token discovery result: {len(tokens)} tokens found")
        
        if tokens:
            print("\n📈 Top tokens found:")
            for i, token in enumerate(tokens[:5]):
                symbol = token.get('symbol', 'N/A')
                volume = token.get('volume_24h', 0)
                liquidity = token.get('liquidity_usd', 0)
                price_change = token.get('price_change_24h', 0)
                
                print(f"  {i+1}. {symbol}:")
                print(f"     Volume: ${volume:,.0f}")
                print(f"     Liquidity: ${liquidity:,.0f}")
                print(f"     Price Change 24h: {price_change:+.1f}%")
                print(f"     Address: {token.get('address', 'N/A')[:10]}...")
                print()
        else:
            print("❌ No tokens found from any source")
            
        return tokens
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    """Main test function"""
    print("🚀 Starting Token Discovery Test...")
    result = asyncio.run(test_token_discovery())
    print(f"\n🏁 Test Complete: {len(result)} tokens discovered")

if __name__ == "__main__":
    main()