"""
Test the enhanced Moralis discovery methods
"""
import asyncio
import sys
import json
sys.path.append('.')
from src.data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer
from src.data.moralis_provider import MoralisProvider

async def test_moralis_discovery_methods():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print('🔍 Testing enhanced Moralis discovery methods...')
    
    # Test direct Moralis provider methods
    moralis_api_key = config.get('api_keys', {}).get('moralis') or config.get('moralis')
    
    if moralis_api_key:
        print('🔗 Testing direct Moralis provider methods...')
        provider = MoralisProvider(moralis_api_key)
        
        # Test getDiscoveryToken
        print('\n1️⃣ Testing getDiscoveryToken...')
        try:
            discovery_tokens = await provider.getDiscoveryToken(limit=10)
            print(f'   Found {len(discovery_tokens)} discovery tokens')
            for i, token in enumerate(discovery_tokens[:3]):
                symbol = token.get('symbol', 'UNKNOWN')
                print(f'   {i+1}. {symbol}')
        except Exception as e:
            print(f'   ❌ getDiscoveryToken failed: {e}')
        
        # Test getFilteredTokens
        print('\n2️⃣ Testing getFilteredTokens...')
        try:
            filtered_tokens = await provider.getFilteredTokens(
                min_volume=10000, 
                min_market_cap=500000,
                limit=10
            )
            print(f'   Found {len(filtered_tokens)} filtered tokens')
            for i, token in enumerate(filtered_tokens[:3]):
                symbol = token.get('symbol', 'UNKNOWN')
                volume = token.get('volume_24h', 0)
                print(f'   {i+1}. {symbol} - Vol: ${volume:,.0f}')
        except Exception as e:
            print(f'   ❌ getFilteredTokens failed: {e}')
    
    # Test enhanced alternative analyzer
    print('\n3️⃣ Testing enhanced alternative analyzer...')
    analyzer = AlternativeBlockchainAnalyzer(config)
    
    tokens = await analyzer.search_trending_tokens(min_volume=10000, min_liquidity=5000)
    print(f'📊 Total tokens found via analyzer: {len(tokens)}')
    
    for i, token in enumerate(tokens[:5]):
        symbol = token.get('symbol', 'UNKNOWN')
        source = token.get('discovery_source', 'unknown')
        volume = token.get('volume_24h', 0)
        print(f'{i+1}. {symbol} ({source}) - Vol: ${volume:,.0f}')

if __name__ == "__main__":
    asyncio.run(test_moralis_discovery_methods())