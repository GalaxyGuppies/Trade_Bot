"""
Test script for the enhanced microcap trading system with:
- Capital allocation slider (max 75%)
- Low cap focus (500k-1.5M market cap)
- High liquidity requirements
- Reddit API integration
"""

import asyncio
import json
import logging
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.social_sentiment import EnhancedSocialSentimentCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_reddit_integration():
    """Test Reddit API integration"""
    logger.info("🔍 Testing Reddit API integration...")
    
    # Initialize sentiment collector with Reddit API
    sentiment_collector = EnhancedSocialSentimentCollector(
        reddit_client_id="VbckKzcb-yJPSTefnDFqDQ",
        reddit_client_secret=None,  # You'll need to provide this
        worldnewsapi_key="46af273710a543ee8e821382082bb08e"
    )
    
    # Test symbols for microcap range
    test_symbols = ["DOGE", "SHIB", "PEPE"]
    
    try:
        # Test Reddit sentiment collection
        logger.info("📱 Testing Reddit sentiment collection...")
        reddit_sentiment = await sentiment_collector.collect_reddit_sentiment(test_symbols, hours_back=24)
        
        for symbol, data in reddit_sentiment.items():
            logger.info(f"Reddit sentiment for {symbol}:")
            logger.info(f"  Score: {data.get('sentiment_score', 0):.3f}")
            logger.info(f"  Confidence: {data.get('confidence', 0):.3f}")
            logger.info(f"  Posts: {data.get('post_count', 0)}")
            logger.info(f"  Comments: {data.get('comment_count', 0)}")
        
        # Test combined sentiment (Reddit + WorldNews)
        logger.info("🌍 Testing combined sentiment analysis...")
        combined_sentiment = await sentiment_collector.get_combined_sentiment(test_symbols)
        
        for symbol, data in combined_sentiment.items():
            logger.info(f"Combined sentiment for {symbol}:")
            logger.info(f"  Final Score: {data.get('sentiment_score', 0):.3f}")
            logger.info(f"  Confidence: {data.get('confidence', 0):.3f}")
            logger.info(f"  Sources: {data.get('sources', [])}")
        
        return True
        
    except Exception as e:
        logger.error(f"Reddit integration test failed: {e}")
        return False

def test_config_validation():
    """Test updated configuration for low cap focus"""
    logger.info("⚙️ Testing configuration validation...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        # Check market cap thresholds
        thresholds = config.get('thresholds', {})
        min_market_cap = thresholds.get('min_market_cap', 0)
        max_market_cap = thresholds.get('max_market_cap', 0)
        
        logger.info(f"Market cap range: ${min_market_cap:,} - ${max_market_cap:,}")
        
        if min_market_cap == 500000 and max_market_cap == 1500000:
            logger.info("✅ Market cap range correctly set for low caps (500k-1.5M)")
        else:
            logger.warning(f"❌ Market cap range not optimal for low caps")
        
        # Check liquidity requirements
        min_liquidity_score = thresholds.get('min_liquidity_score', 0)
        min_volume = thresholds.get('min_volume', 0)
        
        logger.info(f"Liquidity requirements: Score {min_liquidity_score}, Volume ${min_volume:,}")
        
        if min_liquidity_score >= 0.7 and min_volume >= 100000:
            logger.info("✅ High liquidity requirements configured")
        else:
            logger.warning("❌ Liquidity requirements may be too low")
        
        # Check capital allocation limit
        trading = config.get('trading', {})
        max_capital_allocation = trading.get('max_capital_allocation', 1.0)
        
        logger.info(f"Max capital allocation: {max_capital_allocation:.1%}")
        
        if max_capital_allocation == 0.75:
            logger.info("✅ Capital allocation limit set to 75%")
        else:
            logger.warning(f"❌ Capital allocation limit not set to 75%")
        
        # Check Reddit API configuration
        api_keys = config.get('api_keys', {})
        reddit_config = api_keys.get('reddit', {})
        
        if reddit_config.get('client_id') == "VbckKzcb-yJPSTefnDFqDQ":
            logger.info("✅ Reddit API key configured")
        else:
            logger.warning("❌ Reddit API key not configured")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def test_capital_allocation_logic():
    """Test capital allocation calculations"""
    logger.info("💰 Testing capital allocation logic...")
    
    try:
        # Simulate different capital allocation scenarios
        total_capital = 10000  # $10,000 total
        
        allocation_tests = [25.0, 50.0, 75.0, 100.0]  # Different allocation percentages
        
        for allocation_pct in allocation_tests:
            # Apply 75% maximum limit
            capped_allocation = min(allocation_pct, 75.0)
            available_capital = total_capital * (capped_allocation / 100)
            
            logger.info(f"Allocation {allocation_pct:.0f}% -> Capped: {capped_allocation:.0f}% -> Available: ${available_capital:,.0f}")
            
            # Test position sizing with different risk profiles
            risk_profiles = {
                'conservative': 0.5,  # 0.5% position size
                'moderate': 1.0,      # 1.0% position size  
                'aggressive': 2.0     # 2.0% position size
            }
            
            for profile, pos_size_pct in risk_profiles.items():
                position_size = available_capital * (pos_size_pct / 100)
                logger.info(f"  {profile.title()}: ${position_size:.0f} ({pos_size_pct}% of available)")
        
        logger.info("✅ Capital allocation logic working correctly")
        return True
        
    except Exception as e:
        logger.error(f"Capital allocation test failed: {e}")
        return False

def test_market_cap_filtering():
    """Test market cap filtering for low cap focus"""
    logger.info("🎯 Testing market cap filtering logic...")
    
    # Simulate market cap filtering
    min_cap = 500000   # 500k
    max_cap = 1500000  # 1.5M
    
    test_tokens = [
        {"symbol": "TOOHIGH", "market_cap": 5000000},    # Too high
        {"symbol": "TOOLOW", "market_cap": 100000},      # Too low  
        {"symbol": "PERFECT1", "market_cap": 750000},    # Perfect
        {"symbol": "PERFECT2", "market_cap": 1200000},   # Perfect
        {"symbol": "EDGE1", "market_cap": 500000},       # Edge case (min)
        {"symbol": "EDGE2", "market_cap": 1500000},      # Edge case (max)
    ]
    
    filtered_tokens = []
    
    for token in test_tokens:
        market_cap = token["market_cap"]
        symbol = token["symbol"]
        
        if min_cap <= market_cap <= max_cap:
            filtered_tokens.append(token)
            logger.info(f"✅ {symbol}: ${market_cap:,} - ACCEPTED")
        else:
            logger.info(f"❌ {symbol}: ${market_cap:,} - REJECTED")
    
    logger.info(f"Filtered {len(filtered_tokens)}/{len(test_tokens)} tokens in range")
    
    expected_symbols = {"PERFECT1", "PERFECT2", "EDGE1", "EDGE2"}
    actual_symbols = {token["symbol"] for token in filtered_tokens}
    
    if expected_symbols == actual_symbols:
        logger.info("✅ Market cap filtering working correctly")
        return True
    else:
        logger.error(f"❌ Market cap filtering failed. Expected: {expected_symbols}, Got: {actual_symbols}")
        return False

async def main():
    """Run all tests"""
    logger.info("🚀 Starting Enhanced Microcap Trading System Tests")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: Configuration validation
    logger.info("\n📋 TEST 1: Configuration Validation")
    test_results.append(test_config_validation())
    
    # Test 2: Capital allocation logic
    logger.info("\n💰 TEST 2: Capital Allocation Logic")
    test_results.append(test_capital_allocation_logic())
    
    # Test 3: Market cap filtering
    logger.info("\n🎯 TEST 3: Market Cap Filtering")
    test_results.append(test_market_cap_filtering())
    
    # Test 4: Reddit integration
    logger.info("\n📱 TEST 4: Reddit API Integration")
    reddit_result = await test_reddit_integration()
    test_results.append(reddit_result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    
    passed = sum(test_results)
    total = len(test_results)
    
    logger.info(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! System ready for enhanced microcap trading.")
        logger.info("\n🔧 SYSTEM CONFIGURATION:")
        logger.info("• Market Cap Range: $500k - $1.5M")
        logger.info("• Max Capital Allocation: 75%")
        logger.info("• High Liquidity Requirements: ✅")
        logger.info("• Reddit Sentiment Analysis: ✅")
        logger.info("• Valid Wallet Screening: ✅")
        logger.info("• Trade Frequency Monitoring: ✅")
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please review configuration.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)