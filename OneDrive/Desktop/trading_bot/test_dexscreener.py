import asyncio
import sys
import os
import json

# Add the src directory to Python path
sys.path.append('src')

from src.execution.exchange_manager import ExchangeManager

async def test_dexscreener_integration():
    """Test the integrated DexScreener functionality"""
    print("Testing DexScreener integration in ExchangeManager...")
    
    # Load config
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create exchange manager
    manager = ExchangeManager(config)
    
    # Test various Solana tokens
    test_tokens = ['SOL', 'BONK', 'WIF', 'JUP', 'RAY']
    
    for token in test_tokens:
        try:
            print(f"\n--- Testing {token} ---")
            
            # Test token info (which now uses DexScreener)
            token_info = await manager.get_token_info(token)
            if token_info:
                print(f"Symbol: {token_info.symbol}")
                print(f"Address: {token_info.address}")
                print(f"Price: ${token_info.price_usd:.6f}")
                print(f"Decimals: {token_info.decimals}")
            else:
                print(f"Failed to get info for {token}")
                
        except Exception as e:
            print(f"Error testing {token}: {e}")
    
    print("\n=== DexScreener Integration Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_dexscreener_integration())