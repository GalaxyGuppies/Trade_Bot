#!/usr/bin/env python3
"""
Debug script to check WorldNewsAPI response headers
"""

import asyncio
import httpx
from datetime import datetime, timedelta

async def debug_worldnews_headers():
    """Check what headers WorldNewsAPI actually returns"""
    api_key = "46af273710a543ee8e821382082bb08e"
    
    url = "https://api.worldnewsapi.com/search-news"
    params = {
        'text': 'Bitcoin',
        'language': 'en',
        'sort': 'publish-time',
        'sort-direction': 'DESC',
        'number': 5,
        'earliest-publish-date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'api-key': api_key
    }
    
    print("🔍 Debugging WorldNewsAPI Response Headers")
    print("=" * 50)
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=30)
            
            print(f"✅ Status Code: {response.status_code}")
            print(f"📊 All Response Headers:")
            for header, value in response.headers.items():
                print(f"   {header}: {value}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"\n📰 Response Summary:")
                print(f"   News Count: {len(result.get('news', []))}")
                print(f"   Available: {result.get('available', 'N/A')}")
                print(f"   Number: {result.get('number', 'N/A')}")
                print(f"   Offset: {result.get('offset', 'N/A')}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_worldnews_headers())