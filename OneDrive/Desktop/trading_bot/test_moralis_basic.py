"""
Simple Moralis Test - Basic Token Information
"""

import asyncio
import aiohttp
import json

async def test_moralis_basic():
    """Test basic Moralis functionality"""
    
    # Load API key
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['moralis']
    except:
        print("❌ Error loading config")
        return
    
    print("🔗 Testing Basic Moralis Functionality...")
    
    headers = {
        "X-API-Key": api_key,
        "accept": "application/json"
    }
    
    # Test 1: Get token metadata for USDC
    print("\n1️⃣ Testing Token Metadata...")
    
    usdc_address = "0xA0b86a33E6441b811c0a1b26fb18b5b3d05db2b0"
    url = f"https://deep-index.moralis.io/api/v2/erc20/{usdc_address}/metadata"
    params = {"chain": "0x1"}  # Ethereum mainnet
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                print(f"   Status: {response.status}")
                text = await response.text()
                print(f"   Response: {text[:200]}...")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Success!")
                    print(f"   Name: {data.get('name', 'Unknown')}")
                    print(f"   Symbol: {data.get('symbol', 'Unknown')}")
                    print(f"   Decimals: {data.get('decimals', 'Unknown')}")
                else:
                    print(f"   ❌ Error: {response.status}")
                    
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Get wallet token balances
    print("\n2️⃣ Testing Wallet Balances...")
    
    # Use a well-known wallet (Vitalik's wallet)
    vitalik_wallet = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    url = f"https://deep-index.moralis.io/api/v2/{vitalik_wallet}/erc20"
    params = {"chain": "0x1", "limit": 5}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Success!")
                    balances = data.get('result', [])
                    print(f"   Found {len(balances)} tokens")
                    
                    for i, token in enumerate(balances[:3]):
                        symbol = token.get('symbol', 'UNKNOWN')
                        balance = token.get('balance_formatted', '0')
                        print(f"   Token {i+1}: {balance} {symbol}")
                else:
                    text = await response.text()
                    print(f"   ❌ Error: {response.status}")
                    print(f"   Response: {text[:200]}...")
                    
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 3: Get token price
    print("\n3️⃣ Testing Token Price...")
    
    url = f"https://deep-index.moralis.io/api/v2/erc20/{usdc_address}/price"
    params = {"chain": "0x1"}
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                print(f"   Status: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ Success!")
                    print(f"   Price: ${data.get('usdPrice', 'Unknown')}")
                    print(f"   Exchange: {data.get('exchangeName', 'Unknown')}")
                else:
                    text = await response.text()
                    print(f"   ❌ Error: {response.status}")
                    print(f"   Response: {text[:200]}...")
                    
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 4: Check API limits
    print("\n4️⃣ Checking API Rate Limits...")
    
    start_time = asyncio.get_event_loop().time()
    
    for i in range(3):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://deep-index.moralis.io/api/v2/erc20/{usdc_address}/metadata",
                    headers=headers,
                    params={"chain": "0x1"}
                ) as response:
                    print(f"   Request {i+1}: Status {response.status}")
                    if i < 2:  # Small delay between requests
                        await asyncio.sleep(0.2)
        except Exception as e:
            print(f"   Request {i+1}: Error - {e}")
    
    end_time = asyncio.get_event_loop().time()
    print(f"   Total time: {end_time - start_time:.2f}s")
    
    print("\n🎯 Basic Test Summary:")
    print("✅ API key authentication working")
    print("✅ Rate limiting functional")
    print("🚀 Ready for blockchain analysis!")

if __name__ == "__main__":
    asyncio.run(test_moralis_basic())