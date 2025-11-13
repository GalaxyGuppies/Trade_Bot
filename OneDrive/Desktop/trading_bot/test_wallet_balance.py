#!/usr/bin/env python3
"""
Test wallet balance functionality independently
"""

import requests
import json

def test_wallet_balance():
    """Test wallet balance fetching"""
    print("🔍 Testing Wallet Balance Functionality")
    print("=" * 50)
    
    # Load config
    try:
        with open('../config.json', 'r') as f:
            config = json.load(f)
        
        wallet_address = config.get('wallet', {}).get('solana_address', '')
        
        if not wallet_address:
            print("❌ No wallet address in config")
            return
        
        print(f"📍 Wallet: {wallet_address[:8]}...{wallet_address[-8:]}")
        
        # Test Solana RPC connection
        rpc_url = "https://api.mainnet-beta.solana.com"
        
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [wallet_address]
        }
        
        print("🌐 Connecting to Solana mainnet...")
        response = requests.post(rpc_url, json=payload, timeout=15)
        
        print(f"📡 HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"📊 RPC Response: {json.dumps(data, indent=2)}")
            
            if 'result' in data:
                lamports = data['result']['value']
                sol_balance = lamports / 1_000_000_000
                
                print(f"✅ Success!")
                print(f"💰 Balance: {lamports:,} lamports = {sol_balance:.6f} SOL")
                
                # Try to get SOL price too
                print(f"\n🔍 Testing SOL price fetch...")
                try:
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                    
                    from execution.exchange_manager import ExchangeManager
                    
                    exchange_manager = ExchangeManager(config)
                    
                    import asyncio
                    
                    async def get_sol_price():
                        sol_info = await exchange_manager.get_token_info('SOL')
                        if sol_info:
                            return sol_info.price_usd
                        return None
                    
                    sol_price = asyncio.run(get_sol_price())
                    
                    if sol_price:
                        usd_value = sol_balance * sol_price
                        print(f"💵 SOL Price: ${sol_price:.2f}")
                        print(f"💰 Wallet USD Value: ${usd_value:.2f}")
                    else:
                        print("❌ Could not fetch SOL price")
                        
                except Exception as e:
                    print(f"❌ Error fetching SOL price: {e}")
                
            elif 'error' in data:
                print(f"❌ RPC Error: {data['error']}")
            else:
                print(f"❌ Unexpected response format")
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_wallet_balance()