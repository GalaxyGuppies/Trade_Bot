#!/usr/bin/env python3
"""
Test Enhanced Professional Sentiment APIs
Test integration with Santiment, LunarCrush, OpenAI GPT-4, Messari, and others
"""
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.social_sentiment import EnhancedSocialSentimentCollector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_professional_sentiment_apis():
    """Test the enhanced professional sentiment APIs"""
    logger.info("🚀 Testing Enhanced Professional Sentiment APIs")
    
    try:
        # Initialize with API keys (you would set these in environment variables)
        sentiment_collector = EnhancedSocialSentimentCollector(
            # Twitter/X.com API
            twitter_api_key="Sj8ivlnfFe5feHLyLKysOJLyI",
            twitter_api_secret="vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y",
            
            # Third-party sentiment APIs (set these in environment or pass as parameters)
            santiment_api_key=os.getenv("SANTIMENT_API_KEY"),
            lunarcrush_api_key=os.getenv("LUNARCRUSH_API_KEY"), 
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            messari_api_key=os.getenv("MESSARI_API_KEY")
        )
        
        # Test symbols
        test_symbols = ['BTC', 'ETH', 'SOL']
        
        logger.info(f"📊 Testing professional sentiment analysis for: {test_symbols}")
        
        # Test individual third-party APIs
        logger.info("🔬 Testing individual third-party APIs...")
        
        # Test Santiment
        if sentiment_collector.santiment_api_key:
            logger.info("📊 Testing Santiment API...")
            santiment_results = await sentiment_collector.collect_santiment_sentiment(test_symbols)
            for symbol, data in santiment_results.items():
                logger.info(f"   Santiment {symbol}: {data.get('sentiment_label', 'N/A')} "
                          f"({data.get('sentiment_score', 0):.3f}) - {data.get('confidence', 0):.1%} confidence")
        else:
            logger.info("⚠️ Santiment API key not provided - skipping test")
        
        # Test LunarCrush
        if sentiment_collector.lunarcrush_api_key:
            logger.info("🌙 Testing LunarCrush API...")
            lunarcrush_results = await sentiment_collector.collect_lunarcrush_sentiment(test_symbols)
            for symbol, data in lunarcrush_results.items():
                logger.info(f"   LunarCrush {symbol}: {data.get('sentiment_label', 'N/A')} "
                          f"({data.get('sentiment_score', 0):.3f}) - Volume: {data.get('social_volume', 0)}")
        else:
            logger.info("⚠️ LunarCrush API key not provided - skipping test")
        
        # Test OpenAI GPT-4
        if sentiment_collector.openai_api_key:
            logger.info("🤖 Testing OpenAI GPT-4 sentiment analysis...")
            sample_news = [
                "Bitcoin reaches new all-time high as institutional adoption accelerates",
                "Ethereum's latest upgrade shows promising scalability improvements",
                "Solana network experiences high transaction volume amid DeFi growth"
            ]
            openai_results = await sentiment_collector.collect_openai_sentiment(test_symbols, sample_news)
            for symbol, data in openai_results.items():
                logger.info(f"   OpenAI {symbol}: {data.get('sentiment_label', 'N/A')} "
                          f"({data.get('sentiment_score', 0):.3f}) - Factors: {data.get('key_factors', [])}")
        else:
            logger.info("⚠️ OpenAI API key not provided - skipping test")
        
        # Test Messari
        if sentiment_collector.messari_api_key:
            logger.info("📈 Testing Messari API...")
            messari_results = await sentiment_collector.collect_messari_sentiment(test_symbols)
            for symbol, data in messari_results.items():
                logger.info(f"   Messari {symbol}: {data.get('sentiment_label', 'N/A')} "
                          f"({data.get('sentiment_score', 0):.3f}) - News: {data.get('news_count', 0)} articles")
        else:
            logger.info("⚠️ Messari API key not provided - skipping test")
        
        # Test professional combined sentiment
        logger.info("🌟 Testing professional combined sentiment analysis...")
        professional_results = await sentiment_collector.get_professional_combined_sentiment(test_symbols)
        
        logger.info("📈 Professional Sentiment Analysis Results:")
        for symbol, data in professional_results.items():
            sentiment_score = data.get('sentiment_score', 0.0)
            sentiment_label = data.get('sentiment_label', 'unknown')
            confidence = data.get('confidence', 0.0)
            sources = data.get('available_sources', [])
            source_count = data.get('source_count', 0)
            
            logger.info(f"   🎯 {symbol}:")
            logger.info(f"      Overall: {sentiment_label} ({sentiment_score:.3f}) - {confidence:.1%} confidence")
            logger.info(f"      Sources: {source_count} active - {', '.join(sources)}")
            
            # Show detailed breakdown
            detailed = data.get('detailed_data', {})
            for source_name, source_data in detailed.items():
                if source_data and source_data.get('sentiment_score') is not None:
                    source_sentiment = source_data.get('sentiment_score', 0)
                    source_label = source_data.get('sentiment_label', 'N/A')
                    logger.info(f"         {source_name}: {source_label} ({source_sentiment:.3f})")
        
        logger.info("✅ Professional sentiment API integration test completed!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in professional sentiment test: {e}")
        return False

async def test_api_availability():
    """Test which third-party APIs are available"""
    logger.info("🔍 Testing third-party API availability...")
    
    # Check environment variables for API keys
    apis_available = {
        'Santiment': bool(os.getenv("SANTIMENT_API_KEY")),
        'LunarCrush': bool(os.getenv("LUNARCRUSH_API_KEY")),
        'OpenAI': bool(os.getenv("OPENAI_API_KEY")),
        'Messari': bool(os.getenv("MESSARI_API_KEY")),
        'StockGeist': bool(os.getenv("STOCKGEIST_API_KEY"))
    }
    
    logger.info("📡 Third-party API availability:")
    for api_name, available in apis_available.items():
        status = "✅ Available" if available else "❌ Not configured"
        logger.info(f"   {api_name}: {status}")
    
    if not any(apis_available.values()):
        logger.warning("⚠️ No third-party sentiment APIs configured!")
        logger.info("💡 To test with real APIs, set environment variables:")
        logger.info("   export SANTIMENT_API_KEY='your_key'")
        logger.info("   export LUNARCRUSH_API_KEY='your_key'")
        logger.info("   export OPENAI_API_KEY='your_key'")
        logger.info("   export MESSARI_API_KEY='your_key'")
    
    return True

async def demonstrate_sentiment_features():
    """Demonstrate the enhanced sentiment analysis features"""
    logger.info("🎨 Demonstrating enhanced sentiment features...")
    
    # Create sentiment collector
    sentiment_collector = EnhancedSocialSentimentCollector()
    
    # Test basic sentiment analysis
    logger.info("🔬 Testing enhanced sentiment analysis...")
    
    test_texts = [
        "Bitcoin is absolutely crushing it! 🚀 To the moon! Diamond hands! 💎🙌",
        "This crypto is a complete scam, avoid at all costs. Total rugpull incoming.",
        "Ethereum's technology is solid but the market is uncertain right now.",
        "SOL pumping hard! Bullish on Solana ecosystem growth. Amazing fundamentals!",
        "Market looking bearish, might be time to take profits and wait for better entry."
    ]
    
    for text in test_texts:
        sentiment_score = sentiment_collector._basic_sentiment_analysis(text)
        sentiment_label = sentiment_collector._get_sentiment_label(sentiment_score)
        logger.info(f"   Text: '{text[:50]}...'")
        logger.info(f"   Sentiment: {sentiment_label} ({sentiment_score:.3f})")
        logger.info("")
    
    # Test empty data handling
    logger.info("🛡️ Testing error handling and fallbacks...")
    
    empty_sentiment = sentiment_collector._get_empty_third_party_sentiment('test_api')
    logger.info(f"   Empty sentiment structure: {empty_sentiment}")
    
    logger.info("✅ Enhanced sentiment features demonstration completed!")
    
    return True

async def main():
    """Main test function"""
    logger.info("🌟 Enhanced Professional Sentiment API Test Suite")
    logger.info("=" * 70)
    
    # Test API availability
    await test_api_availability()
    logger.info("")
    
    # Demonstrate sentiment features
    await demonstrate_sentiment_features()
    logger.info("")
    
    # Test professional APIs (if available)
    api_test = await test_professional_sentiment_apis()
    
    if api_test:
        logger.info("🎉 All tests completed successfully!")
        logger.info("")
        logger.info("🚀 Enhanced Professional Sentiment System Features:")
        logger.info("   ✅ Multi-source sentiment aggregation")
        logger.info("   ✅ Professional-grade third-party APIs")
        logger.info("   ✅ OpenAI GPT-4 contextual analysis")
        logger.info("   ✅ Weighted sentiment scoring")
        logger.info("   ✅ Confidence-based filtering")
        logger.info("   ✅ Real-time social media monitoring")
        logger.info("   ✅ News and market intelligence integration")
        logger.info("")
        logger.info("💡 To enable full functionality, configure API keys:")
        logger.info("   • Santiment: On-chain + social sentiment data")
        logger.info("   • LunarCrush: Aggregated social sentiment")
        logger.info("   • OpenAI: Advanced contextual analysis")
        logger.info("   • Messari: Professional market intelligence")
        logger.info("   • StockGeist: Multi-platform sentiment aggregation")
    else:
        logger.error("❌ Some tests failed")
        return False
    
    return True

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        if result:
            logger.info("👋 Professional sentiment API test completed successfully!")
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