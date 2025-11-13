"""
Debug LunarCrush API to understand the correct format
"""

import asyncio
import httpx

async def debug_lunarcrush_api():
    """Debug LunarCrush API to understand the correct endpoints and authentication"""
    
    api_key = "qlksp87v8fq338wbc13tmlqqskmhdojibdaexlpan"
    
    print("=== LUNARCRUSH API DEBUG ===")
    print(f"API Key: {api_key[:8]}...{api_key[-8:]}")
    print()
    
    # Test different endpoint formats
    endpoints_to_test = [
        "https://lunarcrush.com/api4/public/coins/list/v1",
        "https://lunarcrush.com/api4/public/topics/list/v1", 
        "https://lunarcrush.com/api4/public/coins/BTC/v1",
        "https://lunarcrush.com/api4/public/topic/BTC/v1",
        "https://api.lunarcrush.com/v2?data=market&key=" + api_key,
    ]
    
    # Test different authentication methods
    auth_methods = [
        {"Authorization": f"Bearer {api_key}"},
        {"Authorization": f"API-KEY {api_key}"},
        {"X-API-KEY": api_key},
        {"api_key": api_key},
        {}  # No auth header, key in URL
    ]
    
    async with httpx.AsyncClient() as client:
        for i, endpoint in enumerate(endpoints_to_test):
            print(f"Testing endpoint {i+1}: {endpoint}")
            
            for j, headers in enumerate(auth_methods):
                try:
                    if "key=" in endpoint:
                        # For URL-based key authentication
                        response = await client.get(endpoint, timeout=10)
                    else:
                        # For header-based authentication
                        response = await client.get(endpoint, headers=headers, timeout=10)
                    
                    print(f"  Auth method {j+1}: Status {response.status_code}")
                    
                    if response.status_code == 200:
                        print(f"  ✅ SUCCESS! Response: {response.text[:200]}...")
                        return endpoint, headers
                    elif response.status_code == 401:
                        print("  ❌ Unauthorized")
                    elif response.status_code == 402:
                        print("  💳 Payment Required")
                    elif response.status_code == 429:
                        print("  ⏳ Rate Limited")
                    else:
                        print(f"  ❓ Status: {response.status_code}")
                        
                except Exception as e:
                    print(f"  ❌ Error: {e}")
            
            print()
    
    print("=== API DEBUG COMPLETE ===")
    print("If no endpoint worked, the API key might require:")
    print("1. Account verification/activation")
    print("2. Different authentication method") 
    print("3. Paid subscription")
    print("4. Different API version/URL")

if __name__ == "__main__":
    asyncio.run(debug_lunarcrush_api())