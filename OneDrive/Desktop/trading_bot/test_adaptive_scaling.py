"""
Test script for Adaptive Profit Scaling System
Demonstrates how position sizes and profit targets scale with performance
"""

import asyncio
import logging
import numpy as np
from datetime import datetime
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.risk.adaptive_scaling import AdaptiveProfitScaling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_adaptive_scaling():
    """Test the adaptive scaling system with simulated trading"""
    
    print("🎯 ADAPTIVE PROFIT SCALING TEST")
    print("=" * 60)
    
    # Initialize scaling system
    scaler = AdaptiveProfitScaling()
    
    # Test parameters
    initial_capital = 10000  # $10k starting capital
    trade_scenarios = [
        # Phase 1: Getting started (small wins)
        {"profit_range": (20, 80), "win_rate": 0.6, "trades": 10},
        # Phase 2: Building momentum (consistent profits)
        {"profit_range": (50, 150), "win_rate": 0.7, "trades": 15},
        # Phase 3: Strong performance (larger wins)
        {"profit_range": (100, 300), "win_rate": 0.75, "trades": 20},
        # Phase 4: Exceptional performance (big wins)
        {"profit_range": (200, 500), "win_rate": 0.8, "trades": 25},
    ]
    
    cumulative_profit = 0
    
    print(f"🚀 Starting with ${initial_capital:,} capital")
    print(f"📊 Base position size: ${scaler.scaling_params.base_position_size}")
    print(f"🎯 Base profit target: {scaler.scaling_params.base_profit_target:.1%}")
    print()
    
    for phase, scenario in enumerate(trade_scenarios, 1):
        print(f"🔄 PHASE {phase}: Simulating {scenario['trades']} trades")
        print(f"   Expected win rate: {scenario['win_rate']:.1%}")
        print(f"   Profit range: ${scenario['profit_range'][0]}-${scenario['profit_range'][1]}")
        
        phase_profit = 0
        
        for trade_num in range(scenario['trades']):
            # Generate trade parameters
            symbol = "BTC/USDT"
            action = "buy"
            entry_price = 45000 + np.random.uniform(-1000, 1000)
            confidence = np.random.uniform(0.7, 0.95)
            
            # Calculate position size using current scaling
            current_capital = initial_capital + cumulative_profit
            position_size = scaler.get_position_size(confidence, current_capital)
            
            # Record trade entry
            trade_id = scaler.record_trade(
                symbol=symbol,
                action=action,
                entry_price=entry_price,
                size=position_size,
                strategy="adaptive_test",
                confidence=confidence,
                market_regime="trending"
            )
            
            # Simulate trade outcome
            is_winner = np.random.random() < scenario['win_rate']
            
            if is_winner:
                # Generate profit within range
                profit = np.random.uniform(scenario['profit_range'][0], scenario['profit_range'][1])
                exit_price = entry_price * (1 + profit / position_size)
            else:
                # Generate loss (smaller than profits)
                loss = np.random.uniform(20, 100)
                profit = -loss
                exit_price = entry_price * (1 + profit / position_size)
            
            # Record trade exit
            scaler.update_trade_exit(trade_id, exit_price, profit)
            
            cumulative_profit += profit
            phase_profit += profit
            
            # Periodic status updates
            if (trade_num + 1) % 5 == 0:
                status = scaler.get_scaling_summary()
                print(f"   Trade {trade_num + 1}: ${profit:+.2f} | "
                      f"Position: {status['current_position_multiplier']:.2f}x | "
                      f"Total: ${status['total_profit']:.2f}")
        
        # Phase summary
        status = scaler.get_scaling_summary()
        print(f"   Phase {phase} Results:")
        print(f"     Phase Profit: ${phase_profit:+.2f}")
        print(f"     Cumulative Profit: ${cumulative_profit:+.2f}")
        print(f"     Position Scaling: {status['current_position_multiplier']:.2f}x")
        print(f"     Profit Target Scaling: {status['current_profit_multiplier']:.2f}x")
        print(f"     Win Rate: {status['win_rate']:.1%}")
        print()
    
    # Final results
    print("🏆 FINAL RESULTS")
    print("=" * 40)
    
    final_status = scaler.get_scaling_summary()
    metrics = scaler.calculate_performance_metrics()
    
    print(f"💰 Total Profit: ${final_status['total_profit']:,.2f}")
    print(f"📈 Position Scaling: {final_status['current_position_multiplier']:.2f}x")
    print(f"🎯 Profit Target Scaling: {final_status['current_profit_multiplier']:.2f}x")
    print(f"🎲 Win Rate: {final_status['win_rate']:.1%}")
    print(f"📊 Total Trades: {final_status['total_trades']}")
    print(f"🔥 Consecutive Wins: {final_status['consecutive_wins']}")
    print(f"📉 Max Drawdown: ${metrics.max_drawdown:.2f}")
    print(f"⚡ Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"💪 Profit Factor: {metrics.profit_factor:.2f}")
    
    # Test position sizing at different confidence levels
    print("\n🎯 POSITION SIZING EXAMPLES")
    print("=" * 40)
    
    current_capital = initial_capital + cumulative_profit
    confidence_levels = [0.6, 0.7, 0.8, 0.9, 0.95]
    
    for confidence in confidence_levels:
        position_size = scaler.get_position_size(confidence, current_capital)
        profit_target = scaler.get_profit_target(confidence)
        stop_loss = scaler.get_stop_loss(confidence)
        
        print(f"Confidence {confidence:.0%}: ${position_size:,.0f} position | "
              f"{profit_target:.1%} target | {stop_loss:.1%} stop")
    
    # Test scaling thresholds
    print("\n🎢 SCALING PROGRESSION")
    print("=" * 40)
    
    for threshold, multiplier in scaler.profit_thresholds:
        status = "✅ REACHED" if final_status['total_profit'] >= threshold else "⏳ Pending"
        print(f"${threshold:,} → {multiplier:.1f}x scaling | {status}")
    
    if final_status['next_profit_threshold']:
        next_threshold, next_multiplier = final_status['next_profit_threshold']
        remaining = next_threshold - final_status['total_profit']
        print(f"\n🚀 Next Goal: ${remaining:,.0f} more profit to reach {next_multiplier:.1f}x scaling!")
    else:
        print(f"\n🏆 MAXIMUM SCALING ACHIEVED! You're a trading legend! 🚀")
    
    print("\n💡 KEY INSIGHTS:")
    print("=" * 40)
    print("• Position sizes scale up as profits accumulate")
    print("• Higher confidence trades get larger position sizes")
    print("• Profit targets increase with better performance")
    print("• System protects against consecutive losses")
    print("• Multiple performance metrics contribute to scaling")
    
    return scaler

async def test_risk_management():
    """Test risk management features"""
    
    print("\n🛡️ RISK MANAGEMENT TEST")
    print("=" * 40)
    
    scaler = AdaptiveProfitScaling()
    
    # Simulate consecutive losses
    print("Testing consecutive loss protection...")
    
    for i in range(8):  # Simulate 8 consecutive losses
        trade_id = scaler.record_trade(
            symbol="BTC/USDT",
            action="buy",
            entry_price=45000,
            size=1000,
            strategy="risk_test",
            confidence=0.8,
            market_regime="volatile"
        )
        
        # Record loss
        scaler.update_trade_exit(trade_id, 44000, -100)
        
        # Check if trading should continue
        should_trade = scaler.should_trade(0.8, {"market_regime": "volatile"})
        status = scaler.get_scaling_summary()
        
        print(f"Loss {i+1}: {status['consecutive_losses']} consecutive losses | "
              f"Should trade: {should_trade} | "
              f"Scaling: {status['current_position_multiplier']:.2f}x")
        
        if not should_trade:
            print("🛑 Trading halted due to consecutive losses!")
            break
    
    print("\nTesting recovery...")
    
    # Simulate recovery with wins
    for i in range(3):
        trade_id = scaler.record_trade(
            symbol="BTC/USDT",
            action="buy",
            entry_price=45000,
            size=1000,
            strategy="recovery_test",
            confidence=0.9,
            market_regime="trending"
        )
        
        # Record win
        scaler.update_trade_exit(trade_id, 46000, 150)
        
        status = scaler.get_scaling_summary()
        print(f"Win {i+1}: {status['consecutive_wins']} consecutive wins | "
              f"Scaling: {status['current_position_multiplier']:.2f}x")

if __name__ == "__main__":
    asyncio.run(test_adaptive_scaling())
    asyncio.run(test_risk_management())