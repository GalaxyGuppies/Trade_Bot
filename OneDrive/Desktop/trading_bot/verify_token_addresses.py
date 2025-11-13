"""
Quick token address verification for BANGERS, TRUMP, BASED
Checks if addresses are valid Solana token mints
"""

import asyncio
import aiohttp

async def verify_token_address(address: str, name: str):
    """Verify a token address using Solana RPC"""
    rpc_url = "https://api.mainnet-beta.solana.com"
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            address,
            {"encoding": "jsonParsed"}
        ]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(rpc_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                data = await response.json()
                
                result = data.get('result')
                if result and result.get('value'):
                    account_data = result['value']
                    print(f"\n✅ {name} ({address[:8]}...)")
                    print(f"   Owner: {account_data.get('owner', 'Unknown')}")
                    print(f"   Lamports: {account_data.get('lamports', 0)}")
                    
                    parsed = account_data.get('data', {}).get('parsed')
                    if parsed:
                        print(f"   Type: {parsed.get('type', 'Unknown')}")
                        info = parsed.get('info', {})
                        if 'decimals' in info:
                            print(f"   Decimals: {info['decimals']}")
                        if 'supply' in info:
                            print(f"   Supply: {info['supply']}")
                    return True
                else:
                    print(f"\n❌ {name} ({address[:8]}...)")
                    print(f"   ERROR: Account not found or invalid")
                    return False
                    
    except Exception as e:
        print(f"\n❌ {name} - Error: {e}")
        return False

async def main():
    print("🔍 Verifying Token Addresses")
    print("=" * 60)
    
    tokens = {
        'BANGERS': '3wppuwUMAGgxnX75Aqr4W91xYWaN6RjxjCUFiPZUpump',
        'TRUMP': '6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN',
        'BASED': 'EMAGfmV5bMzYEtgda43ZmCYwmLL7SaMi2RVqaRPjpump'
    }
    
    results = {}
    for name, address in tokens.items():
        results[name] = await verify_token_address(address, name)
        await asyncio.sleep(0.5)  # Be nice to RPC
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for name, valid in results.items():
        status = "✅ VALID" if valid else "❌ INVALID"
        print(f"  {name}: {status}")
    
    print("\nNOTE: If a token shows as INVALID, the address might be:")
    print("  1. Wrong/typo")
    print("  2. Not a token mint (could be a wallet address)")
    print("  3. Token doesn't exist on mainnet")

if __name__ == "__main__":
    asyncio.run(main())
