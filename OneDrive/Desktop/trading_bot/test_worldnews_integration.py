#!/usr/bin/env python3
"""
Test script for WorldNewsAPI integration
"""

import asyncio
import sys
import os
import logging
from datetime import datetime

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.social_sentiment import EnhancedSocialSentimentCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_worldnews_integration():
    """Test WorldNewsAPI integration"""
    print("🧪 Testing WorldNewsAPI Integration")
    print("=" * 50)
    
    # Initialize collector with WorldNewsAPI key
    worldnews_api_key = "46af273710a543ee8e821382082bb08e"
    
    collector = EnhancedSocialSentimentCollector(
        worldnewsapi_key=worldnews_api_key
    )
    
    # Test symbols
    test_symbols = ['BTC', 'ETH']
    
    print(f"📊 Testing WorldNewsAPI sentiment collection for {test_symbols}")
    print(f"🗞️  API Key: {worldnews_api_key[:10]}..." + "*" * 15)
    
    try:
        # Test WorldNewsAPI collection
        print("\n🔍 Testing WorldNewsAPI sentiment collection...")
        worldnews_results = await collector.collect_worldnews_sentiment(test_symbols)
        
        for symbol, data in worldnews_results.items():
            print(f"\n📈 {symbol} WorldNews Results:")
            print(f"   Sentiment Score: {data.get('sentiment_score', 'N/A')}")
            print(f"   Sentiment Label: {data.get('sentiment_label', 'N/A')}")
            print(f"   Confidence: {data.get('confidence', 'N/A'):.2%}" if data.get('confidence') else "   Confidence: N/A")
            print(f"   Articles Analyzed: {data.get('articles_analyzed', 'N/A')}")
            print(f"   Relevant Articles: {data.get('relevant_articles', 'N/A')}")
            print(f"   Data Quality: {data.get('data_quality', 'N/A')}")
            print(f"   API Quota Used: {data.get('api_quota_used', 'N/A')}")
            print(f"   API Quota Left: {data.get('api_quota_left', 'N/A')}")
        
        # Test quota tracking
        print(f"\n📊 API Quota Status:")
        print(f"   Quota Used: {collector.worldnews_quota_used}")
        print(f"   Quota Left: {collector.worldnews_quota_left}")
        
        # Test combined professional sentiment
        print(f"\n🎯 Testing Combined Professional Sentiment...")
        combined_results = await collector.get_professional_combined_sentiment(test_symbols)
        
        for symbol, data in combined_results.items():
            print(f"\n📊 {symbol} Combined Professional Results:")
            print(f"   Overall Sentiment: {data.get('sentiment_label', 'N/A')} ({data.get('sentiment_score', 'N/A'):.3f})")
            print(f"   Confidence: {data.get('confidence', 'N/A'):.2%}" if data.get('confidence') else "   Confidence: N/A")
            print(f"   Sources Used: {data.get('available_sources', [])}")
            print(f"   Source Count: {data.get('source_count', 'N/A')}")
            
            # Show WorldNews contribution
            worldnews_detail = data.get('detailed_data', {}).get('worldnews', {})
            if worldnews_detail:
                print(f"   WorldNews Contribution:")
                print(f"     Sentiment: {worldnews_detail.get('sentiment_label', 'N/A')} ({worldnews_detail.get('sentiment_score', 'N/A'):.3f})")
                print(f"     Articles: {worldnews_detail.get('articles_analyzed', 'N/A')} total, {worldnews_detail.get('relevant_articles', 'N/A')} relevant")
                print(f"     Quality: {worldnews_detail.get('data_quality', 'N/A')}")
        
        print(f"\n✅ WorldNewsAPI integration test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during WorldNewsAPI testing: {e}")
        import traceback
        traceback.print_exc()

async def test_quota_tracking():
    """Test quota tracking functionality"""
    print(f"\n🔄 Testing Quota Tracking...")
    
    worldnews_api_key = "46af273710a543ee8e821382082bb08e"
    collector = EnhancedSocialSentimentCollector(worldnewsapi_key=worldnews_api_key)
    
    print(f"Initial Quota Status:")
    print(f"   Used: {collector.worldnews_quota_used}")
    print(f"   Left: {collector.worldnews_quota_left}")
    
    # Make a small request to test quota tracking
    results = await collector.collect_worldnews_sentiment(['BTC'])
    
    print(f"After API Call:")
    print(f"   Used: {collector.worldnews_quota_used}")
    print(f"   Left: {collector.worldnews_quota_left}")

if __name__ == "__main__":
    print(f"🚀 Starting WorldNewsAPI Integration Tests")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        asyncio.run(test_worldnews_integration())
        asyncio.run(test_quota_tracking())
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()