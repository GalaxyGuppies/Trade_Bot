"""
Simple test to verify Moralis methods exist
"""
import sys
import json
sys.path.append('.')
from src.data.moralis_provider import MoralisProvider

def test_methods_exist():
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    moralis_api_key = config.get('api_keys', {}).get('moralis') or config.get('moralis')
    
    if moralis_api_key:
        provider = MoralisProvider(moralis_api_key)
        
        print('🔍 Checking if methods exist...')
        print(f'✅ getDiscoveryToken: {hasattr(provider, "getDiscoveryToken")}')
        print(f'✅ getFilteredTokens: {hasattr(provider, "getFilteredTokens")}')
        print(f'✅ getMultipleTokenAnalytics: {hasattr(provider, "getMultipleTokenAnalytics")}')
        
        # Also check method signatures
        if hasattr(provider, "getFilteredTokens"):
            print(f'📋 getFilteredTokens callable: {callable(getattr(provider, "getFilteredTokens"))}')
    else:
        print('❌ No Moralis API key found')

if __name__ == "__main__":
    test_methods_exist()