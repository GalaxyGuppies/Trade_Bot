"""
Test Moralis endpoint fix
"""
import asyncio
import sys
import json
sys.path.append('.')
from src.data.alternative_blockchain_analyzer import AlternativeBlockchainAnalyzer

async def test_moralis_fix():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    print('🔍 Testing Moralis endpoint fix...')
    analyzer = AlternativeBlockchainAnalyzer(config)
    
    if analyzer.moralis_provider:
        print('🔗 Moralis provider available - testing endpoint...')
        # Test just the Moralis method directly
        moralis_tokens = await analyzer._get_moralis_trending(min_volume=10000, min_liquidity=5000)
        print(f'📊 Moralis tokens found: {len(moralis_tokens)}')
        
        for i, token in enumerate(moralis_tokens[:3]):
            symbol = token.get('symbol', 'UNKNOWN')
            source = token.get('discovery_source', 'unknown')
            volume = token.get('volume_24h', 0)
            print(f'{i+1}. {symbol} ({source}) - Vol: ${volume:,.0f}')
    else:
        print('❌ Moralis provider not available')

if __name__ == "__main__":
    asyncio.run(test_moralis_fix())