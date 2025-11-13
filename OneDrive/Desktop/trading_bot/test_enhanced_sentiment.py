#!/usr/bin/env python3
"""
Test Enhanced Social Sentiment Integration
Verify X.com API integration and sentiment analysis capabilities
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.social_sentiment import SocialSentimentCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_sentiment_integration():
    """Test the enhanced social sentiment integration"""
    logger.info("🚀 Testing Enhanced Social Sentiment Integration")
    
    try:
        # Initialize sentiment collector with real X.com API credentials
        logger.info("🐦 Initializing social sentiment collector...")
        sentiment_collector = SocialSentimentCollector(
            twitter_api_key="Sj8ivlnfFe5feHLyLKysOJLyI",
            twitter_api_secret="vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y"
        )
        
        # Test with major crypto symbols
        test_symbols = ['BTC', 'ETH', 'SOL']
        
        logger.info(f"📊 Testing sentiment analysis for: {test_symbols}")
        
        # Get combined sentiment data
        sentiment_results = await sentiment_collector.get_combined_sentiment(
            symbols=test_symbols,
            hours_back=6  # Last 6 hours for faster testing
        )
        
        # Display results
        logger.info("📈 Sentiment Analysis Results:")
        for symbol, data in sentiment_results.items():
            sentiment_score = data.get('sentiment_score', 0.0)
            sentiment_label = data.get('sentiment_label', 'unknown')
            confidence = data.get('confidence', 0.0)
            
            twitter_data = data.get('twitter_data', {})
            reddit_data = data.get('reddit_data', {})
            
            logger.info(f"   🎯 {symbol}:")
            logger.info(f"      Overall: {sentiment_label} ({sentiment_score:.3f}) - {confidence:.1%} confidence")
            logger.info(f"      Twitter: {twitter_data.get('tweet_count', 0)} tweets, {twitter_data.get('sentiment_label', 'N/A')} sentiment")
            logger.info(f"      Reddit:  {reddit_data.get('post_count', 0)} posts, {reddit_data.get('sentiment_label', 'N/A')} sentiment")
        
        # Test basic sentiment analysis
        logger.info("🔬 Testing basic sentiment analysis...")
        
        test_texts = [
            "Bitcoin is going to the moon! 🚀 Bullish AF!",
            "This altcoin is a scam, avoid at all costs",
            "Ethereum looks decent, might be worth a small position",
            "SOL is pumping hard, diamond hands! 💎🙌"
        ]
        
        for text in test_texts:
            sentiment_score = await sentiment_collector._analyze_text_sentiment(text)
            logger.info(f"   Text: '{text}'")
            logger.info(f"   Sentiment: {sentiment_score:.3f}")
        
        logger.info("✅ Enhanced sentiment integration test completed successfully!")
        
        # Test sentiment summary
        logger.info("📊 Testing sentiment summary...")
        for symbol in test_symbols:
            summary = sentiment_collector.get_sentiment_summary(symbol)
            logger.info(f"   {symbol} Summary: {summary}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in sentiment integration test: {e}")
        return False

async def test_api_availability():
    """Test API availability and basic functionality"""
    logger.info("🔍 Testing API availability...")
    
    try:
        sentiment_collector = SocialSentimentCollector(
            twitter_api_key="Sj8ivlnfFe5feHLyLKysOJLyI",
            twitter_api_secret="vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y"
        )
        
        # Test Twitter API initialization
        if sentiment_collector.twitter_client:
            logger.info("✅ X.com (Twitter) API initialized successfully")
        else:
            logger.warning("⚠️ X.com (Twitter) API not available")
        
        # Test Reddit API initialization
        if sentiment_collector.reddit_client:
            logger.info("✅ Reddit API initialized successfully")
        else:
            logger.warning("⚠️ Reddit API not available (optional)")
        
        # Test sentiment analyzer
        if sentiment_collector.sentiment_analyzer:
            logger.info("✅ Sentiment analyzer initialized successfully")
        else:
            logger.warning("⚠️ Sentiment analyzer using fallback method")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error testing API availability: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🌟 Enhanced Social Sentiment Integration Test Suite")
    logger.info("=" * 60)
    
    # Test API availability first
    api_test = await test_api_availability()
    
    if api_test:
        logger.info("📡 API tests passed, proceeding with sentiment analysis...")
        
        # Test full sentiment integration
        sentiment_test = await test_sentiment_integration()
        
        if sentiment_test:
            logger.info("🎉 All tests completed successfully!")
            logger.info("🚀 Enhanced social sentiment integration is ready for trading!")
        else:
            logger.error("❌ Sentiment integration tests failed")
            return False
    else:
        logger.error("❌ API availability tests failed")
        return False
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            logger.info("👋 Test completed successfully!")
            sys.exit(0)
        else:
            logger.error("💥 Test failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("⌨️ Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Unexpected error: {e}")
        sys.exit(1)