"""
Enhanced Trading Integration Example

Demonstrates how to use professional sentiment APIs in trading decisions
"""

import asyncio
import logging
from src.data.social_sentiment import EnhancedSocialSentimentCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def trading_sentiment_example():
    """Example of integrating professional sentiment into trading decisions"""
    
    logger.info("🎯 Professional Sentiment Trading Integration Example")
    logger.info("=" * 60)
    
    # Initialize enhanced sentiment collector
    sentiment_collector = EnhancedSocialSentimentCollector(
        twitter_api_key="your_twitter_key",
        twitter_api_secret="your_twitter_secret"
    )
    
    # Example symbols to analyze
    symbols = ['BTC', 'ETH', 'SOL', 'MATIC', 'AVAX']
    
    logger.info(f"📊 Analyzing sentiment for: {symbols}")
    
    # Collect professional sentiment data
    sentiment_results = await sentiment_collector.get_professional_combined_sentiment(symbols)
    
    # Trading decision logic
    trading_decisions = []
    
    for symbol, data in sentiment_results.items():
        sentiment_score = data.get('sentiment_score', 0.0)
        confidence = data.get('confidence', 0.0)
        sources_count = data.get('sources_count', 0)
        
        # Trading decision rules based on sentiment
        decision = "HOLD"  # Default
        reasoning = []
        
        # High confidence bullish sentiment
        if sentiment_score > 0.6 and confidence > 0.7:
            decision = "BUY"
            reasoning.append(f"Strong bullish sentiment ({sentiment_score:.2f})")
            reasoning.append(f"High confidence ({confidence:.1%})")
        
        # High confidence bearish sentiment
        elif sentiment_score < -0.6 and confidence > 0.7:
            decision = "SELL"
            reasoning.append(f"Strong bearish sentiment ({sentiment_score:.2f})")
            reasoning.append(f"High confidence ({confidence:.1%})")
        
        # Moderate bullish with multiple sources
        elif sentiment_score > 0.3 and sources_count >= 3:
            decision = "BUY_SMALL"
            reasoning.append(f"Moderate bullish sentiment from {sources_count} sources")
        
        # Moderate bearish with multiple sources
        elif sentiment_score < -0.3 and sources_count >= 3:
            decision = "SELL_SMALL"
            reasoning.append(f"Moderate bearish sentiment from {sources_count} sources")
        
        # Low confidence or neutral
        else:
            reasoning.append("Insufficient sentiment signal for action")
        
        trading_decisions.append({
            'symbol': symbol,
            'decision': decision,
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'sources_count': sources_count,
            'reasoning': '; '.join(reasoning)
        })
    
    # Display trading decisions
    logger.info("🎯 Professional Sentiment-Based Trading Decisions:")
    logger.info("-" * 60)
    
    for decision in trading_decisions:
        action_emoji = {
            'BUY': '🟢',
            'BUY_SMALL': '🟡', 
            'SELL': '🔴',
            'SELL_SMALL': '🟠',
            'HOLD': '⚪'
        }.get(decision['decision'], '⚪')
        
        logger.info(f"{action_emoji} {decision['symbol']}: {decision['decision']}")
        logger.info(f"   Sentiment: {decision['sentiment_score']:+.3f} | "
                   f"Confidence: {decision['confidence']:.1%} | "
                   f"Sources: {decision['sources_count']}")
        logger.info(f"   Reasoning: {decision['reasoning']}")
        logger.info("")
    
    # Generate summary statistics
    buy_signals = len([d for d in trading_decisions if 'BUY' in d['decision']])
    sell_signals = len([d for d in trading_decisions if 'SELL' in d['decision']])
    hold_signals = len([d for d in trading_decisions if d['decision'] == 'HOLD'])
    
    avg_confidence = sum(d['confidence'] for d in trading_decisions) / len(trading_decisions)
    
    logger.info("📈 Trading Summary:")
    logger.info(f"   Buy Signals: {buy_signals}")
    logger.info(f"   Sell Signals: {sell_signals}")
    logger.info(f"   Hold Signals: {hold_signals}")
    logger.info(f"   Average Confidence: {avg_confidence:.1%}")
    
    return trading_decisions

async def sentiment_monitoring_example():
    """Example of continuous sentiment monitoring"""
    
    logger.info("📡 Starting continuous sentiment monitoring...")
    
    sentiment_collector = EnhancedSocialSentimentCollector(
        twitter_api_key="your_twitter_key",
        twitter_api_secret="your_twitter_secret"
    )
    
    # Monitor these symbols continuously
    watch_symbols = ['BTC', 'ETH']
    
    try:
        for cycle in range(3):  # 3 monitoring cycles for demo
            logger.info(f"🔄 Monitoring Cycle {cycle + 1}")
            
            for symbol in watch_symbols:
                sentiment_data = await sentiment_collector.get_professional_combined_sentiment([symbol])
                
                if sentiment_data and symbol in sentiment_data:
                    data = sentiment_data[symbol]
                    score = data.get('sentiment_score', 0.0)
                    confidence = data.get('confidence', 0.0)
                    
                    # Alert on extreme sentiment
                    if abs(score) > 0.7 and confidence > 0.6:
                        alert_type = "🚨 EXTREME BULLISH" if score > 0 else "🚨 EXTREME BEARISH"
                        logger.info(f"{alert_type} SENTIMENT ALERT: {symbol}")
                        logger.info(f"   Score: {score:+.3f} | Confidence: {confidence:.1%}")
                        
                        # This is where you could trigger immediate trading actions
                        logger.info("   → Consider immediate position adjustment")
            
            # Wait before next monitoring cycle (in real trading, this might be 5-15 minutes)
            logger.info("⏱️  Waiting for next monitoring cycle...")
            await asyncio.sleep(2)  # Short delay for demo
            
    except KeyboardInterrupt:
        logger.info("🛑 Sentiment monitoring stopped by user")

async def main():
    """Main demo function"""
    logger.info("🌟 Enhanced Professional Sentiment Trading Demo")
    logger.info("🔧 Note: Configure API keys in environment variables for full functionality")
    logger.info("")
    
    # Demo 1: Trading decisions based on sentiment
    await trading_sentiment_example()
    
    logger.info("\n" + "="*60 + "\n")
    
    # Demo 2: Continuous sentiment monitoring
    await sentiment_monitoring_example()
    
    logger.info("✅ Professional sentiment integration demo completed!")

if __name__ == "__main__":
    asyncio.run(main())