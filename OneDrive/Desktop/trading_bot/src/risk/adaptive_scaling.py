"""
Adaptive Profit Scaling System
Incrementally increases position sizes and profit targets based on performance
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    """Individual trade record for performance tracking"""
    timestamp: float
    symbol: str
    action: str  # 'buy', 'sell'
    entry_price: float
    exit_price: Optional[float]
    size: float
    profit_loss: Optional[float]
    success: Optional[bool]
    strategy: str
    confidence: float
    market_regime: str

@dataclass
class PerformanceMetrics:
    """Performance metrics for scaling decisions"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_profit: float
    average_profit_per_trade: float
    sharpe_ratio: float
    max_drawdown: float
    consecutive_wins: int
    consecutive_losses: int
    profit_factor: float
    
@dataclass
class ScalingParameters:
    """Current scaling parameters"""
    base_position_size: float
    current_position_multiplier: float
    base_profit_target: float
    current_profit_multiplier: float
    risk_per_trade: float
    max_position_multiplier: float
    confidence_threshold: float

class AdaptiveProfitScaling:
    """
    Adaptive scaling system that increases position sizes and profit targets
    based on bot performance and accumulated profits
    """
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.trade_history: List[TradeRecord] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize scaling parameters
        self.scaling_params = ScalingParameters(
            base_position_size=self.config.get('base_position_size', 100.0),
            current_position_multiplier=1.0,
            base_profit_target=self.config.get('base_profit_target', 0.02),  # 2%
            current_profit_multiplier=1.0,
            risk_per_trade=self.config.get('risk_per_trade', 0.01),  # 1%
            max_position_multiplier=self.config.get('max_position_multiplier', 10.0),
            confidence_threshold=self.config.get('confidence_threshold', 0.7)
        )
        
        # Scaling thresholds
        self.profit_thresholds = [
            (500, 1.2),    # $500 profit: 1.2x scaling
            (1000, 1.5),   # $1000 profit: 1.5x scaling
            (2500, 2.0),   # $2500 profit: 2x scaling
            (5000, 3.0),   # $5000 profit: 3x scaling
            (10000, 5.0),  # $10000 profit: 5x scaling
            (25000, 10.0)  # $25000 profit: 10x scaling
        ]
        
        # Performance thresholds for additional scaling
        self.performance_thresholds = {
            'win_rate': [(0.6, 1.1), (0.7, 1.3), (0.8, 1.5)],
            'sharpe_ratio': [(1.0, 1.1), (1.5, 1.2), (2.0, 1.4)],
            'profit_factor': [(1.5, 1.1), (2.0, 1.2), (3.0, 1.4)]
        }
        
        # Load existing trade history
        self._load_trade_history()
        
        logger.info("Adaptive Profit Scaling initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f).get('adaptive_scaling', {})
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return {}
    
    def _load_trade_history(self):
        """Load trade history from persistent storage"""
        try:
            with open('trade_history.json', 'r') as f:
                data = json.load(f)
                self.trade_history = [TradeRecord(**trade) for trade in data.get('trades', [])]
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
        except FileNotFoundError:
            logger.info("No existing trade history found, starting fresh")
    
    def _save_trade_history(self):
        """Save trade history to persistent storage"""
        data = {
            'trades': [asdict(trade) for trade in self.trade_history],
            'last_updated': time.time()
        }
        with open('trade_history.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    def record_trade(self, 
                    symbol: str,
                    action: str,
                    entry_price: float,
                    size: float,
                    strategy: str,
                    confidence: float,
                    market_regime: str,
                    exit_price: Optional[float] = None,
                    profit_loss: Optional[float] = None) -> str:
        """Record a new trade"""
        
        trade = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            profit_loss=profit_loss,
            success=profit_loss > 0 if profit_loss is not None else None,
            strategy=strategy,
            confidence=confidence,
            market_regime=market_regime
        )
        
        self.trade_history.append(trade)
        self._save_trade_history()
        
        # Update scaling if trade is complete
        if profit_loss is not None:
            self._update_scaling_parameters()
        
        trade_id = f"{symbol}_{int(trade.timestamp)}"
        logger.info(f"Trade recorded: {trade_id}")
        return trade_id
    
    def update_trade_exit(self, trade_id: str, exit_price: float, profit_loss: float):
        """Update trade with exit information"""
        # Find trade by reconstructing ID pattern
        symbol, timestamp_str = trade_id.split('_')
        timestamp = int(timestamp_str)
        
        for trade in reversed(self.trade_history):
            if (trade.symbol == symbol and 
                abs(trade.timestamp - timestamp) < 60):  # Within 1 minute
                trade.exit_price = exit_price
                trade.profit_loss = profit_loss
                trade.success = profit_loss > 0
                self._save_trade_history()
                self._update_scaling_parameters()
                logger.info(f"Trade updated: {trade_id}, P&L: ${profit_loss:.2f}")
                return
        
        logger.warning(f"Trade not found for update: {trade_id}")
    
    def calculate_performance_metrics(self, lookback_days: int = 30) -> PerformanceMetrics:
        """Calculate performance metrics for recent period"""
        cutoff_time = time.time() - (lookback_days * 24 * 3600)
        recent_trades = [t for t in self.trade_history 
                        if t.timestamp > cutoff_time and t.profit_loss is not None]
        
        if not recent_trades:
            return PerformanceMetrics(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0.0, total_profit=0.0, average_profit_per_trade=0.0,
                sharpe_ratio=0.0, max_drawdown=0.0, consecutive_wins=0,
                consecutive_losses=0, profit_factor=0.0
            )
        
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t.success)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [t.profit_loss for t in recent_trades]
        total_profit = sum(profits)
        average_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate Sharpe ratio (simplified)
        if len(profits) > 1:
            profit_std = np.std(profits)
            sharpe_ratio = (average_profit / profit_std) if profit_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative_profits = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative_profits)
        drawdowns = running_max - cumulative_profits
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Calculate consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        
        for trade in reversed(recent_trades):
            if trade.success:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    break
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    break
        
        if current_streak > 0:
            consecutive_wins = current_streak
        else:
            consecutive_losses = abs(current_streak)
        
        # Calculate profit factor
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            average_profit_per_trade=average_profit,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            profit_factor=profit_factor
        )
    
    def _update_scaling_parameters(self):
        """Update scaling parameters based on current performance"""
        metrics = self.calculate_performance_metrics()
        self.performance_history.append(metrics)
        
        # 1. Profit-based scaling
        profit_multiplier = 1.0
        for threshold, multiplier in self.profit_thresholds:
            if metrics.total_profit >= threshold:
                profit_multiplier = multiplier
        
        # 2. Performance-based scaling
        performance_multiplier = 1.0
        
        # Win rate bonus
        for threshold, multiplier in self.performance_thresholds['win_rate']:
            if metrics.win_rate >= threshold:
                performance_multiplier *= multiplier
        
        # Sharpe ratio bonus
        for threshold, multiplier in self.performance_thresholds['sharpe_ratio']:
            if metrics.sharpe_ratio >= threshold:
                performance_multiplier *= multiplier
        
        # Profit factor bonus
        for threshold, multiplier in self.performance_thresholds['profit_factor']:
            if metrics.profit_factor >= threshold:
                performance_multiplier *= multiplier
        
        # 3. Risk adjustment for consecutive losses
        risk_adjustment = 1.0
        if metrics.consecutive_losses > 3:
            risk_adjustment = 0.5  # Reduce position size after consecutive losses
        elif metrics.consecutive_losses > 5:
            risk_adjustment = 0.25  # Further reduce after more losses
        
        # 4. Calculate final multipliers
        total_multiplier = profit_multiplier * performance_multiplier * risk_adjustment
        total_multiplier = min(total_multiplier, self.scaling_params.max_position_multiplier)
        
        # Update scaling parameters
        self.scaling_params.current_position_multiplier = total_multiplier
        self.scaling_params.current_profit_multiplier = min(total_multiplier, 3.0)  # Cap profit target scaling
        
        logger.info(f"Scaling updated - Position: {total_multiplier:.2f}x, "
                   f"Profit Target: {self.scaling_params.current_profit_multiplier:.2f}x")
        logger.info(f"Performance - Win Rate: {metrics.win_rate:.1%}, "
                   f"Total Profit: ${metrics.total_profit:.2f}, "
                   f"Sharpe: {metrics.sharpe_ratio:.2f}")
    
    def get_position_size(self, confidence: float, available_capital: float) -> float:
        """Calculate position size based on current scaling and confidence"""
        
        # Base position size
        base_size = min(
            self.scaling_params.base_position_size * self.scaling_params.current_position_multiplier,
            available_capital * 0.1  # Never risk more than 10% of capital
        )
        
        # Confidence adjustment
        confidence_multiplier = 1.0
        if confidence >= 0.9:
            confidence_multiplier = 1.5  # High confidence boost
        elif confidence >= 0.8:
            confidence_multiplier = 1.2
        elif confidence < 0.6:
            confidence_multiplier = 0.7  # Reduce size for low confidence
        
        # Risk per trade adjustment
        risk_adjusted_size = min(
            base_size * confidence_multiplier,
            available_capital * self.scaling_params.risk_per_trade
        )
        
        return max(risk_adjusted_size, 10.0)  # Minimum $10 position
    
    def get_profit_target(self, confidence: float) -> float:
        """Calculate profit target based on current scaling and confidence"""
        
        base_target = self.scaling_params.base_profit_target * self.scaling_params.current_profit_multiplier
        
        # Confidence adjustment
        if confidence >= 0.9:
            return base_target * 1.5  # Higher targets for high confidence
        elif confidence >= 0.8:
            return base_target * 1.2
        elif confidence < 0.6:
            return base_target * 0.8  # Lower targets for low confidence
        
        return base_target
    
    def get_stop_loss(self, confidence: float) -> float:
        """Calculate stop loss based on confidence and current performance"""
        
        base_stop = 0.02  # 2% base stop loss
        
        # Tighter stops for high confidence trades
        if confidence >= 0.9:
            return base_stop * 0.7
        elif confidence >= 0.8:
            return base_stop * 0.85
        elif confidence < 0.6:
            return base_stop * 1.3  # Wider stops for low confidence
        
        return base_stop
    
    def should_trade(self, confidence: float, market_conditions: Dict) -> bool:
        """Determine if we should take a trade based on current scaling status"""
        
        metrics = self.calculate_performance_metrics(lookback_days=7)  # Recent performance
        
        # Don't trade if recent performance is poor
        if metrics.consecutive_losses > 5:
            logger.warning("Pausing trading due to consecutive losses")
            return False
        
        if metrics.win_rate < 0.3 and metrics.total_trades > 10:
            logger.warning("Pausing trading due to low win rate")
            return False
        
        # Require higher confidence during poor performance
        if metrics.consecutive_losses > 3:
            required_confidence = 0.8
        else:
            required_confidence = self.scaling_params.confidence_threshold
        
        return confidence >= required_confidence
    
    def get_scaling_summary(self) -> Dict:
        """Get current scaling status summary"""
        
        metrics = self.calculate_performance_metrics()
        
        return {
            'current_position_multiplier': self.scaling_params.current_position_multiplier,
            'current_profit_multiplier': self.scaling_params.current_profit_multiplier,
            'total_profit': metrics.total_profit,
            'win_rate': metrics.win_rate,
            'total_trades': metrics.total_trades,
            'consecutive_wins': metrics.consecutive_wins,
            'consecutive_losses': metrics.consecutive_losses,
            'sharpe_ratio': metrics.sharpe_ratio,
            'next_profit_threshold': self._get_next_threshold(metrics.total_profit),
            'scaling_enabled': metrics.consecutive_losses < 5,
            'confidence_threshold': self.scaling_params.confidence_threshold
        }
    
    def _get_next_threshold(self, current_profit: float) -> Optional[Tuple[float, float]]:
        """Get the next profit threshold for scaling"""
        for threshold, multiplier in self.profit_thresholds:
            if current_profit < threshold:
                return threshold, multiplier
        return None

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize scaling system
    scaler = AdaptiveProfitScaling()
    
    # Simulate some trades
    print("🔄 Testing Adaptive Profit Scaling...")
    
    # Simulate successful trades
    for i in range(10):
        trade_id = scaler.record_trade(
            symbol="BTC/USDT",
            action="buy",
            entry_price=50000 + i * 100,
            size=100,
            strategy="signal_fusion",
            confidence=0.8,
            market_regime="trending"
        )
        
        # Simulate profit
        profit = np.random.uniform(20, 100)  # $20-100 profit
        scaler.update_trade_exit(trade_id, 50000 + i * 100 + 50, profit)
    
    # Get scaling summary
    summary = scaler.get_scaling_summary()
    print(f"\n📊 Scaling Summary:")
    print(f"Position Multiplier: {summary['current_position_multiplier']:.2f}x")
    print(f"Profit Multiplier: {summary['current_profit_multiplier']:.2f}x")
    print(f"Total Profit: ${summary['total_profit']:.2f}")
    print(f"Win Rate: {summary['win_rate']:.1%}")
    print(f"Next Threshold: ${summary['next_profit_threshold'][0] if summary['next_profit_threshold'] else 'Max reached'}")
    
    # Test position sizing
    capital = 10000
    confidence = 0.85
    position_size = scaler.get_position_size(confidence, capital)
    profit_target = scaler.get_profit_target(confidence)
    
    print(f"\n🎯 Trade Sizing:")
    print(f"Available Capital: ${capital}")
    print(f"Confidence: {confidence:.1%}")
    print(f"Position Size: ${position_size:.2f}")
    print(f"Profit Target: {profit_target:.1%}")
    
    print("\n✅ Adaptive scaling test completed!")