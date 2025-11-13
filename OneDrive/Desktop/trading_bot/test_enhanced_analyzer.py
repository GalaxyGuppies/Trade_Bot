"""
Test the enhanced alternative blockchain analyzer with Moralis integration
"""
import asyncio
import sys
import json
sys.path.append('.')
from src.data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer

async def test_enhanced():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print('🔍 Testing enhanced alternative blockchain analyzer...')
    analyzer = AlternativeBlockchainAnalyzer(config)
    print(f'🔗 Moralis provider initialized: {analyzer.moralis_provider is not None}')
    print(f'📊 Curated tokens available: {len(analyzer.curated_tokens)}')
    
    # Test trending tokens search with failover
    print('\n🔍 Testing token discovery with failover...')
    tokens = await analyzer.search_trending_tokens(min_volume=10000, min_liquidity=5000)
    print(f'📊 Total tokens found: {len(tokens)}')
    
    for i, token in enumerate(tokens[:5]):
        symbol = token.get('symbol', 'UNKNOWN')
        source = token.get('discovery_source', 'unknown')
        price = token.get('price_usd', 0)
        volume = token.get('volume_24h', 0)
        print(f'{i+1}. {symbol} ({source}) - ${price:.6f} | Vol: ${volume:,.0f}')

if __name__ == "__main__":
    asyncio.run(test_enhanced())