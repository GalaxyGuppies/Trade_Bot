"""
Comprehensive Demo: Adaptive Profit Scaling with Real Market Data
Shows how position sizes and profit targets scale incrementally with profits
"""

import asyncio
import logging
import json
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk.adaptive_scaling import AdaptiveProfitScaling
from src.data.coinmarketcap_provider import CoinMarketCapProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def comprehensive_scaling_demo():
    """Comprehensive demo of adaptive scaling with real market data"""
    
    print("🚀 ADAPTIVE PROFIT SCALING - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("This demo shows how your trading bot incrementally increases")
    print("position sizes and profit targets as it becomes more profitable!")
    print()
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        cmc_api_key = config['api_keys']['coinmarketcap']
        print("✅ Using real market data from CoinMarketCap")
    except (FileNotFoundError, KeyError):
        print("❌ CoinMarketCap API key not found")
        return
    
    # Initialize components
    scaler = AdaptiveProfitScaling()
    market_provider = CoinMarketCapProvider(cmc_api_key)
    
    # Get real market data
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    print(f"\n📊 CURRENT MARKET CONDITIONS")
    print("-" * 40)
    
    market_data = {}
    for symbol in symbols:
        metrics = await market_provider.get_market_metrics(symbol)
        if metrics:
            market_data[symbol] = metrics
            print(f"{symbol}: ${metrics['price']:,.2f} | "
                  f"Vol: ${metrics['volume_24h']:,.0f} | "
                  f"Volatility: {metrics['volatility_24h']:.1%}")
    
    # Show scaling thresholds
    print(f"\n🎢 PROFIT SCALING THRESHOLDS")
    print("-" * 40)
    for i, (threshold, multiplier) in enumerate(scaler.profit_thresholds, 1):
        print(f"{i}. ${threshold:,} profit → {multiplier:.1f}x position size")
    
    # Current status
    status = scaler.get_scaling_summary()
    print(f"\n📈 CURRENT SCALING STATUS")
    print("-" * 40)
    print(f"Position Multiplier: {status['current_position_multiplier']:.2f}x")
    print(f"Profit Target Multiplier: {status['current_profit_multiplier']:.2f}x")
    print(f"Total Profit: ${status['total_profit']:.2f}")
    print(f"Total Trades: {status['total_trades']}")
    print(f"Win Rate: {status['win_rate']:.1%}")
    
    # Demo position sizing at different profit levels
    print(f"\n💰 POSITION SIZING DEMONSTRATION")
    print("-" * 40)
    print("Showing how position sizes change as profits accumulate...")
    print()
    
    # Simulate different profit levels
    demo_profits = [0, 250, 750, 1500, 3000, 7500, 15000]
    initial_capital = 10000
    
    for profit in demo_profits:
        # Temporarily set profit for demo
        scaler.scaling_params.current_position_multiplier = 1.0
        for threshold, multiplier in scaler.profit_thresholds:
            if profit >= threshold:
                scaler.scaling_params.current_position_multiplier = multiplier
        
        # Calculate position size
        confidence = 0.8
        current_capital = initial_capital + profit
        position_size = scaler.get_position_size(confidence, current_capital)
        profit_target = scaler.get_profit_target(confidence)
        
        # Show scaling status
        multiplier = scaler.scaling_params.current_position_multiplier
        print(f"💵 ${profit:,} profit: {multiplier:.1f}x scaling → "
              f"${position_size:,.0f} position | {profit_target:.1%} target")
    
    # Live trading simulation
    print(f"\n🎯 LIVE TRADING SIMULATION")
    print("-" * 40)
    print("Simulating 10 trades with real market conditions...")
    print()
    
    simulated_capital = 10000
    
    for i in range(10):
        # Pick a random symbol
        symbol = symbols[i % len(symbols)]
        real_price = market_data[symbol]['price']
        volatility = market_data[symbol]['volatility_24h']
        
        # Generate confidence based on market strength
        market_strength = market_data[symbol]['market_strength']
        confidence = 0.6 + (market_strength * 0.3)  # 0.6 to 0.9
        
        # Calculate position size
        position_size = scaler.get_position_size(confidence, simulated_capital)
        profit_target = scaler.get_profit_target(confidence)
        stop_loss = scaler.get_stop_loss(confidence)
        
        # Simulate trade outcome based on confidence and volatility
        success_probability = confidence * (1 - volatility)  # Higher volatility reduces success
        is_winner = success_probability > 0.5
        
        # Record trade
        trade_id = scaler.record_trade(
            symbol=symbol,
            action="buy",
            entry_price=real_price,
            size=position_size,
            strategy="adaptive_demo",
            confidence=confidence,
            market_regime="live_market"
        )
        
        # Simulate profit/loss
        if is_winner:
            # Profit based on target and confidence
            profit = position_size * profit_target * confidence
        else:
            # Loss based on stop loss
            profit = -position_size * stop_loss
        
        # Update trade
        exit_price = real_price * (1 + profit / position_size)
        scaler.update_trade_exit(trade_id, exit_price, profit)
        
        simulated_capital += profit
        current_status = scaler.get_scaling_summary()
        
        # Display trade result
        result_icon = "✅" if is_winner else "❌"
        print(f"{result_icon} Trade {i+1}: {symbol} | "
              f"${position_size:,.0f} position | "
              f"${profit:+,.0f} P&L | "
              f"{current_status['current_position_multiplier']:.2f}x scaling")
    
    # Final results
    final_status = scaler.get_scaling_summary()
    print(f"\n🏆 SIMULATION RESULTS")
    print("-" * 40)
    print(f"Final Capital: ${simulated_capital:,.2f}")
    print(f"Total Profit: ${final_status['total_profit']:,.2f}")
    print(f"Position Scaling: {final_status['current_position_multiplier']:.2f}x")
    print(f"Profit Target Scaling: {final_status['current_profit_multiplier']:.2f}x")
    print(f"Win Rate: {final_status['win_rate']:.1%}")
    
    if final_status['next_profit_threshold']:
        next_threshold, next_multiplier = final_status['next_profit_threshold']
        remaining = next_threshold - final_status['total_profit']
        print(f"\n🚀 Next Goal: ${remaining:,.0f} more profit to reach {next_multiplier:.1f}x scaling!")
    else:
        print(f"\n🏆 MAXIMUM SCALING ACHIEVED! 🚀")
    
    # Performance analysis
    print(f"\n📊 PERFORMANCE ANALYSIS")
    print("-" * 40)
    
    metrics = scaler.calculate_performance_metrics()
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: ${metrics.max_drawdown:.2f}")
    print(f"Profit Factor: {metrics.profit_factor:.2f}")
    print(f"Average Profit per Trade: ${metrics.average_profit_per_trade:.2f}")
    
    # Real-world examples
    print(f"\n💡 REAL-WORLD EXAMPLES")
    print("-" * 40)
    print("Here's how the scaling would work in practice:")
    print()
    
    examples = [
        (0, "Starting out", "Conservative 1x scaling, $100 positions"),
        (500, "First milestone", "1.2x scaling unlocked, $120 positions"),
        (1000, "Building momentum", "1.5x scaling, $150 positions"),
        (2500, "Proven system", "2x scaling, $200 positions"),
        (5000, "Strong performance", "3x scaling, $300 positions"),
        (10000, "Exceptional results", "5x scaling, $500 positions"),
        (25000, "Trading mastery", "10x scaling, $1000 positions")
    ]
    
    for profit, milestone, description in examples:
        icon = "✅" if final_status['total_profit'] >= profit else "⏳"
        print(f"{icon} ${profit:,}: {milestone} - {description}")
    
    print(f"\n🎯 KEY BENEFITS OF ADAPTIVE SCALING:")
    print("-" * 40)
    print("• Automatically scales up as you prove profitability")
    print("• Protects capital during losing streaks")
    print("• Maximizes profits during winning streaks")
    print("• Higher confidence trades get larger positions")
    print("• Profit targets increase with better performance")
    print("• Reduces risk after consecutive losses")
    print("• Multiple performance metrics ensure robust scaling")
    
    print(f"\n🚀 Your bot is now ready to scale profits incrementally!")
    print("   Run 'python advanced_launcher.py' to start live trading!")

if __name__ == "__main__":
    asyncio.run(comprehensive_scaling_demo())