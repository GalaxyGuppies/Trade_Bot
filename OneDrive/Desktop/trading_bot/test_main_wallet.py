import asyncio
import httpx
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient

async def test_main_config_wallet():
    """Test the wallet address from main config"""
    
    # Wallet from main config
    wallet_address = "6zpXi3eJSDVxBJUBaK9gs72hGZ8ViYjBFBrqT3Hpxk8x"
    rpc_url = "https://api.mainnet-beta.solana.com"
    
    print(f"Testing main config wallet: {wallet_address}")
    print(f"Using RPC URL: {rpc_url}")
    
    try:
        # Test Solana RPC connection
        print("\n1. Testing Solana RPC connection...")
        client = AsyncClient(rpc_url)
        
        wallet_pubkey = Pubkey.from_string(wallet_address)
        balance_response = await client.get_balance(wallet_pubkey)
        
        if balance_response.value is not None:
            sol_balance = balance_response.value / 1e9  # Convert lamports to SOL
            print(f"   ✅ SOL Balance: {sol_balance:.6f} SOL")
        else:
            print("   ❌ Failed to get SOL balance")
            return
        
        await client.close()
        
        # Test SOL price fetching
        print("\n2. Testing SOL price fetching...")
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get("https://api.dexscreener.com/latest/dex/tokens/So11111111111111111111111111111111111111112")
            
            if response.status_code == 200:
                data = response.json()
                if data.get('pairs'):
                    sol_price = float(data['pairs'][0]['priceUsd'])
                    print(f"   ✅ SOL Price: ${sol_price:.2f}")
                    
                    usd_balance = sol_balance * sol_price
                    print(f"   ✅ Wallet USD Value: ${usd_balance:.2f}")
                else:
                    print("   ❌ No SOL price data found")
            else:
                print(f"   ❌ Failed to fetch SOL price: {response.status_code}")
        
        print("\n✅ Main config wallet test completed!")
        
    except Exception as e:
        print(f"❌ Error testing main config wallet: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_main_config_wallet())