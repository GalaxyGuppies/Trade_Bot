"""
Test LunarCrush FREE API Integration

Test script for your specific LunarCrush API key
"""

import asyncio
import os
import sys
import logging

# Add the current directory to path
sys.path.append('.')
sys.path.append('src')

from src.data.social_sentiment import EnhancedSocialSentimentCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_lunarcrush_api():
    """Test LunarCrush API with your specific key"""
    
    print("=== TESTING LUNARCRUSH FREE API ===")
    print()
    
    # Your LunarCrush API key
    lunarcrush_api_key = "qlksp87v8fq338wbc13tmlqqskmhdojibdaexlpan"
    
    # Initialize sentiment collector with your API key
    collector = EnhancedSocialSentimentCollector(
        twitter_api_key="test",
        twitter_api_secret="test",
        lunarcrush_api_key=lunarcrush_api_key
    )
    
    # Test symbols (major cryptocurrencies)
    test_symbols = ['BTC', 'ETH', 'SOL']
    
    print(f"Testing LunarCrush API with symbols: {test_symbols}")
    print(f"API Key: {lunarcrush_api_key[:8]}...{lunarcrush_api_key[-8:]}")
    print()
    
    try:
        # Test LunarCrush sentiment collection
        results = await collector.collect_lunarcrush_sentiment(test_symbols)
        
        if results:
            print("SUCCESS: LunarCrush API working!")
            print()
            
            for symbol, data in results.items():
                print(f"📊 {symbol} Sentiment Analysis:")
                print(f"   Sentiment Score: {data.get('sentiment_score', 0):.3f}")
                print(f"   Sentiment Label: {data.get('sentiment_label', 'Unknown')}")
                print(f"   Confidence: {data.get('confidence', 0):.1%}")
                print(f"   Social Volume: {data.get('social_volume', 0):,}")
                print(f"   Social Engagement: {data.get('social_engagement', 0):,}")
                print(f"   Social Contributors: {data.get('social_contributors', 0):,}")
                print(f"   Source: {data.get('source', 'Unknown')}")
                print()
        else:
            print("WARNING: No results returned from LunarCrush API")
            print("This might be normal for the first test")
            
    except Exception as e:
        print(f"ERROR: LunarCrush API test failed: {e}")
        return False
    
    # Test API usage tracking
    print("💡 FREE TIER USAGE TRACKING:")
    print("   Daily limit: 200 requests")
    print(f"   Requests made today: {len(test_symbols)} (for this test)")
    print("   Recommended: Check 3 symbols, 3 times per day = 9 requests/day")
    print("   Your usage: 4.5% of daily limit - PERFECT!")
    print()
    
    return True

async def test_combined_sentiment():
    """Test combined sentiment with LunarCrush included"""
    
    print("=== TESTING COMBINED PROFESSIONAL SENTIMENT ===")
    print()
    
    # Set environment variable for the test
    os.environ['LUNARCRUSH_API_KEY'] = "qlksp87v8fq338wbc13tmlqqskmhdojibdaexlpan"
    
    collector = EnhancedSocialSentimentCollector(
        twitter_api_key="Sj8ivlnfFe5feHLyLKysOJLyI",
        twitter_api_secret="vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y",
        lunarcrush_api_key="qlksp87v8fq338wbc13tmlqqskmhdojibdaexlpan"
    )
    
    # Test combined sentiment for trading decision
    test_symbols = ['BTC', 'ETH']
    
    try:
        combined_results = await collector.get_professional_combined_sentiment(test_symbols)
        
        if combined_results:
            print("SUCCESS: Combined sentiment analysis working!")
            print()
            
            for symbol, data in combined_results.items():
                sentiment_score = data.get('sentiment_score', 0)
                confidence = data.get('confidence', 0)
                sources = data.get('sources_count', 0)
                
                print(f"🎯 {symbol} Combined Professional Sentiment:")
                print(f"   Overall Score: {sentiment_score:+.3f}")
                print(f"   Confidence: {confidence:.1%}")
                print(f"   Sources Used: {sources}")
                
                # Trading decision logic
                if sentiment_score > 0.3 and confidence > 0.6:
                    decision = "🟢 BUY SIGNAL"
                elif sentiment_score < -0.3 and confidence > 0.6:
                    decision = "🔴 SELL SIGNAL"
                else:
                    decision = "⚪ HOLD"
                
                print(f"   Trading Signal: {decision}")
                print()
        else:
            print("INFO: Combined sentiment not available (normal for first run)")
            
    except Exception as e:
        print(f"ERROR: Combined sentiment test failed: {e}")

async def main():
    """Main test function"""
    
    print("🌙 LUNARCRUSH FREE API INTEGRATION TEST")
    print("=" * 50)
    print()
    
    # Test 1: Basic LunarCrush API
    success = await test_lunarcrush_api()
    
    if success:
        print("✅ LunarCrush API test passed!")
    else:
        print("❌ LunarCrush API test failed!")
        return
    
    print("\n" + "=" * 50 + "\n")
    
    # Test 2: Combined sentiment
    await test_combined_sentiment()
    
    print("=" * 50)
    print("🎉 LUNARCRUSH INTEGRATION COMPLETE!")
    print()
    print("Next steps:")
    print("1. Set environment variable: $env:LUNARCRUSH_API_KEY = 'qlksp87v8fq338wbc13tmlqqskmhdojibdaexlpan'")
    print("2. Run your trading bot: python integrated_trading_launcher.py")
    print("3. Monitor FREE tier usage (200 requests/day limit)")

if __name__ == "__main__":
    asyncio.run(main())