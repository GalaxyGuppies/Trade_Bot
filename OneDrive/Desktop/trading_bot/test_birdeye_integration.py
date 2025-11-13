"""
Test the enhanced alternative analyzer with Birdeye integration
"""
import asyncio
import sys
import json
sys.path.append('.')
from src.data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer

async def test_enhanced_with_birdeye():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print('🔍 Testing enhanced analyzer with Birdeye integration...')
    analyzer = AlternativeBlockchainAnalyzer(config)
    
    print(f'🔗 Moralis provider: {analyzer.moralis_provider is not None}')
    print(f'🦅 Birdeye provider: {analyzer.birdeye_provider is not None}')
    print(f'📊 Curated tokens: {len(analyzer.curated_tokens)}')
    
    # Test token discovery with new failover sequence
    print('\n🔍 Testing token discovery with Birdeye failover...')
    tokens = await analyzer.search_trending_tokens(min_volume=10000, min_liquidity=5000)
    print(f'📊 Total tokens found: {len(tokens)}')
    
    for i, token in enumerate(tokens[:5]):
        symbol = token.get('symbol', 'UNKNOWN')
        source = token.get('discovery_source', 'unknown')
        volume = token.get('volume_24h', 0)
        print(f'{i+1}. {symbol} ({source}) - Vol: ${volume:,.0f}')

if __name__ == "__main__":
    asyncio.run(test_enhanced_with_birdeye())