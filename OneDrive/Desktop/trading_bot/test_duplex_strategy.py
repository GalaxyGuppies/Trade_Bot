"""
Test the Duplex Trading Strategy System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.duplex_strategy import DuplexTradingStrategy, TradeStrategy

def test_duplex_strategy():
    """Test the duplex strategy with sample data"""
    
    # Initialize strategy
    config = {}
    duplex = DuplexTradingStrategy(config)
    
    # Test candidates - various scenarios
    test_candidates = [
        {
            'symbol': 'BTC',
            'price_usd': 42000,
            'volume_24h': 15000000000,
            'market_cap': 800000000000,
            'liquidity_usd': 50000000,
            'volatility_score': 5.5,
            'confidence': 0.95,
            'rugpull_risk': 0.02
        },
        {
            'symbol': 'ALTCOIN1',
            'price_usd': 0.05,
            'volume_24h': 2000000,
            'market_cap': 15000000,
            'liquidity_usd': 300000,
            'volatility_score': 8.5,
            'confidence': 0.7,
            'rugpull_risk': 0.3
        },
        {
            'symbol': 'MICROCAP1',
            'price_usd': 0.001,
            'volume_24h': 500000,
            'market_cap': 2000000,
            'liquidity_usd': 80000,
            'volatility_score': 9.8,
            'confidence': 0.45,
            'rugpull_risk': 0.65
        },
        {
            'symbol': 'ETH',
            'price_usd': 2400,
            'volume_24h': 8000000000,
            'market_cap': 300000000000,
            'liquidity_usd': 40000000,
            'volatility_score': 6.0,
            'confidence': 0.92,
            'rugpull_risk': 0.02
        }
    ]
    
    # Test technical scenarios
    technical_scenarios = [
        {
            'name': 'RSI Oversold + High Volume',
            'rsi': 25,
            'bb_position': 0.15,
            'macd_signal': 'BULLISH'
        },
        {
            'name': 'Neutral Technical + Stable',
            'rsi': 55,
            'bb_position': 0.6,
            'macd_signal': 'NEUTRAL'
        },
        {
            'name': 'RSI Overbought + Trend Reversal',
            'rsi': 75,
            'bb_position': 0.9,
            'macd_signal': 'BEARISH'
        },
        {
            'name': 'Moderate Oversold + Bullish MACD',
            'rsi': 38,
            'bb_position': 0.3,
            'macd_signal': 'BULLISH'
        }
    ]
    
    market_conditions = {
        'volatility': 'medium',
        'trend': 'neutral',
        'time_of_day': 14
    }
    
    available_capital = 1000.0
    
    print("🔄 Testing Duplex Trading Strategy System")
    print("=" * 60)
    
    for candidate in test_candidates:
        print(f"\n📊 Testing: {candidate['symbol']}")
        print(f"   Market Cap: ${candidate['market_cap']:,}")
        print(f"   Volume 24h: ${candidate['volume_24h']:,}")
        print(f"   Volatility: {candidate['volatility_score']}")
        print(f"   Confidence: {candidate['confidence']}")
        
        for scenario in technical_scenarios:
            technical_data = {
                'rsi': scenario['rsi'],
                'bb_position': scenario['bb_position'],
                'macd_signal': scenario['macd_signal'],
                'volume_24h': candidate['volume_24h'],
                'volatility': candidate['volatility_score'],
                'market_cap': candidate['market_cap'],
                'liquidity': candidate['liquidity_usd']
            }
            
            try:
                signal = duplex.evaluate_opportunity(
                    candidate, technical_data, market_conditions, available_capital
                )
                
                print(f"\n   🔬 Scenario: {scenario['name']}")
                print(f"      Strategy: {signal.strategy.value.upper()}")
                print(f"      Confidence: {signal.confidence:.2f}")
                print(f"      Position Size: ${signal.position_size:.2f}")
                print(f"      Duration: {signal.hold_duration}")
                print(f"      Reasoning: {signal.reasoning}")
                
                if signal.strategy != TradeStrategy.SKIP:
                    print(f"      Stop Loss: ${signal.stop_loss:.4f}")
                    print(f"      Take Profit: ${signal.take_profit:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {e}")
    
    # Test strategy statistics
    print("\n📈 Strategy Configuration:")
    stats = duplex.get_strategy_stats()
    print(f"   Scalp Min Volatility: {stats['scalp_thresholds']['min_volatility']}")
    print(f"   Scalp Max Volatility: {stats['scalp_thresholds']['max_volatility']}")
    print(f"   Swing Min Market Cap: ${stats['swing_thresholds']['min_market_cap']:,}")
    print(f"   Scalp Base Position %: {stats['scalp_sizing']['base_percent']}%")
    print(f"   Swing Base Position %: {stats['swing_sizing']['base_percent']}%")

if __name__ == "__main__":
    test_duplex_strategy()