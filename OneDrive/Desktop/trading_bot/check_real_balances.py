"""
Quick script to check ACTUAL on-chain token balances
This bypasses the bot's portfolio tracker and queries the blockchain directly
"""
import httpx
import json
from pathlib import Path

# Load wallet address from config
config_path = Path(__file__).parent / 'config.json'
with open(config_path, 'r') as f:
    config = json.load(f)

wallet_address = config['wallet']['solana_address']

print(f"🔍 Checking wallet: {wallet_address}\n")

# Token addresses
tokens = {
    'GXY': 'PKikg1HNZinFvMgqk76aBDY4fF1fgGYQ3tv9kKypump',
    'USELESS': 'Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk',
    'ACE': 'GEuuznWpn6iuQAJxLKQDVGXPtrqXHNWTk3gZqqvJpump',
    'PUPI': '5m2yk8Vx5EpsTR8TVojnZDiJfP66rgog5mzGffZvpump',
    'ROI': 'vEHiuRmd8WvCkswH8Xy4VXTEMXA7JScik47XZkDbonk'
}

decimals = {
    'GXY': 6,
    'USELESS': 9,
    'ACE': 6,
    'PUPI': 6,
    'ROI': 9
}

# Check SOL balance first
rpc_url = "https://api.mainnet-beta.solana.com"
payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "getBalance",
    "params": [wallet_address]
}

with httpx.Client(timeout=30.0) as client:
    response = client.post(rpc_url, json=payload)
    sol_balance = response.json()['result']['value'] / 1e9
    print(f"💰 SOL: {sol_balance:.6f} SOL (${sol_balance * 152:.2f} @ $152/SOL)\n")

# Check each token
print("📊 TOKEN BALANCES (ACTUAL ON-CHAIN):")
print("-" * 60)

for token_name, token_mint in tokens.items():
    try:
        # Get token accounts for this wallet
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenAccountsByOwner",
            "params": [
                wallet_address,
                {"mint": token_mint},
                {"encoding": "jsonParsed"}
            ]
        }
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(rpc_url, json=payload)
            result = response.json()
            
            if 'result' in result and result['result']['value']:
                account = result['result']['value'][0]
                balance = float(account['account']['data']['parsed']['info']['tokenAmount']['uiAmount'])
                print(f"✅ {token_name}: {balance:.6f} tokens")
            else:
                print(f"❌ {token_name}: 0 tokens (no token account)")
                
    except Exception as e:
        print(f"⚠️ {token_name}: Error checking balance - {e}")

print("\n" + "=" * 60)
print("💡 Compare these ACTUAL balances with what the bot thinks it has!")
print("=" * 60)
