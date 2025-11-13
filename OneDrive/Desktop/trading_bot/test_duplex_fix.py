#!/usr/bin/env python3
"""
Test script for the enhanced duplex trading strategy
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.strategies.duplex_strategy import DuplexTradingStrategy, TradeStrategy

def test_duplex_strategy():
    """Test the duplex strategy to ensure confidence variable works"""
    
    # Mock config
    config = {
        'trading': {
            'min_volume_usd': 50000,
            'min_liquidity_usd': 50000
        }
    }
    
    # Initialize strategy
    strategy = DuplexTradingStrategy(config)
    
    # Mock candidate data
    candidate = {
        'symbol': 'TEST',
        'price_usd': 1.0,
        'market_cap': 1000000,
        'volume_24h': 100000,
        'liquidity_usd': 75000,
        'volatility_score': 7.5
    }
    
    # Mock technical data
    technical_data = {
        'rsi': 35,  # Oversold
        'bb_position': 0.1,  # Bollinger bands oversold
        'macd_signal': 'BULLISH'
    }
    
    # Mock market conditions
    market_conditions = {
        'volatility': 'medium',
        'trend': 'bullish'
    }
    
    try:
        # Test the strategy evaluation
        signal = strategy.evaluate_opportunity(
            candidate=candidate,
            technical_data=technical_data,
            market_conditions=market_conditions,
            available_capital=1000.0
        )
        
        print(f"✅ Strategy evaluation successful!")
        print(f"   Strategy: {signal.strategy}")
        print(f"   Confidence: {signal.confidence:.2f}")
        print(f"   Position Size: ${signal.position_size:.2f}")
        print(f"   Stop Loss: ${signal.stop_loss:.6f}")
        print(f"   Take Profit: ${signal.take_profit:.6f}")
        print(f"   Reasoning: {signal.reasoning}")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy evaluation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Enhanced Duplex Trading Strategy...")
    success = test_duplex_strategy()
    
    if success:
        print("\n🎉 All tests passed! The confidence variable fix is working.")
    else:
        print("\n💥 Tests failed! Need to investigate further.")