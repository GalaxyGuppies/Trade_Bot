"""
Advanced Safety Manager for Real Trading
Implements strict risk controls for small capital trading
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate" 
    AGGRESSIVE = "aggressive"

@dataclass
class SafetyLimits:
    """Safety limits for $44 capital trading"""
    # Position sizing (based on your $44 balance)
    min_position_size: float = 1.0   # $1 minimum
    max_position_size: float = 5.0   # $5 maximum
    
    # Stop losses (very tight for scalping)
    scalp_stop_loss: float = 0.005   # 0.5% for scalping
    swing_stop_loss: float = 0.02    # 2.0% for swing trades
    
    # Risk per trade
    max_risk_per_trade: float = 0.44  # 1% of $44 balance
    max_daily_loss: float = 2.20      # 5% of balance max daily loss
    
    # Position limits
    max_concurrent_positions: int = 3  # Maximum 3 positions
    max_daily_trades: int = 10         # Prevent overtrading
    
    # Emergency stops
    portfolio_stop_loss: float = 0.15  # 15% total portfolio loss = emergency stop

@dataclass
class TradeRisk:
    """Risk assessment for individual trade"""
    symbol: str
    position_size: float
    risk_amount: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    confidence_level: float

class SafetyManager:
    """Advanced safety manager for real trading with small amounts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.limits = SafetyLimits()
        
        # Track daily statistics
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.active_positions = {}
        self.trade_history = []
        
        # Current portfolio value
        self.portfolio_value = config.get('wallet', {}).get('current_balance', 44.0)
        self.initial_balance = self.portfolio_value
        
        logger.info(f"🛡️ Safety Manager initialized with ${self.portfolio_value:.2f} capital")
        logger.info(f"📊 Safety Limits: ${self.limits.min_position_size}-${self.limits.max_position_size} per trade")
    
    def calculate_position_size(self, confidence: float, strategy: str) -> float:
        """Calculate safe position size based on confidence and strategy"""
        try:
            # Base position size on confidence (1-5 scale)
            if confidence >= 0.9:   # Very high confidence
                base_size = 5.0
            elif confidence >= 0.8: # High confidence  
                base_size = 4.0
            elif confidence >= 0.7: # Good confidence
                base_size = 3.0
            elif confidence >= 0.6: # Moderate confidence
                base_size = 2.0
            else:                   # Low confidence
                base_size = 1.0
            
            # Adjust for strategy type
            if strategy.upper() == "SCALP":
                size_multiplier = 0.8  # Smaller positions for scalping
            else:  # SWING
                size_multiplier = 1.0  # Full size for swing trades
            
            calculated_size = base_size * size_multiplier
            
            # Ensure within limits
            position_size = max(self.limits.min_position_size, 
                              min(calculated_size, self.limits.max_position_size))
            
            # Check available capital
            available_capital = self.portfolio_value * 0.88  # Keep 12% as buffer
            position_size = min(position_size, available_capital / 3)  # Max 1/3 of available per trade
            
            logger.info(f"💰 Position sizing: confidence={confidence:.2f}, strategy={strategy}, size=${position_size:.2f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.limits.min_position_size
    
    def calculate_stop_loss(self, entry_price: float, strategy: str, direction: str) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Get appropriate stop loss percentage
            if strategy.upper() == "SCALP":
                stop_pct = self.limits.scalp_stop_loss  # 0.5%
                profit_pct = stop_pct * 2.5  # 1.25% profit target (2.5:1 RR)
            else:  # SWING
                stop_pct = self.limits.swing_stop_loss  # 2.0%
                profit_pct = stop_pct * 2.0  # 4.0% profit target (2:1 RR)
            
            if direction.upper() == "BUY":
                stop_loss = entry_price * (1 - stop_pct)
                take_profit = entry_price * (1 + profit_pct)
            else:  # SELL
                stop_loss = entry_price * (1 + stop_pct)
                take_profit = entry_price * (1 - profit_pct)
            
            logger.info(f"🎯 Stop/Profit: Entry=${entry_price:.6f}, Stop=${stop_loss:.6f}, Target=${take_profit:.6f}")
            return stop_loss, take_profit
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return entry_price * 0.995, entry_price * 1.01  # Fallback values
    
    def assess_trade_risk(self, symbol: str, entry_price: float, position_size: float, 
                         strategy: str, direction: str, confidence: float) -> TradeRisk:
        """Comprehensive risk assessment for trade"""
        try:
            # Safety check for entry price
            if entry_price <= 0:
                logger.error(f"❌ Invalid entry price: ${entry_price:.6f}")
                return None
                
            stop_loss, take_profit = self.calculate_stop_loss(entry_price, strategy, direction)
            
            # Calculate risk amount
            if direction.upper() == "BUY":
                risk_amount = (entry_price - stop_loss) * (position_size / entry_price)
            else:
                risk_amount = (stop_loss - entry_price) * (position_size / entry_price)
            
            # Calculate risk-reward ratio
            profit_potential = abs(take_profit - entry_price) * (position_size / entry_price)
            risk_reward_ratio = profit_potential / risk_amount if risk_amount > 0 else 0
            
            trade_risk = TradeRisk(
                symbol=symbol,
                position_size=position_size,
                risk_amount=risk_amount,
                stop_loss_price=stop_loss,
                take_profit_price=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                confidence_level=confidence
            )
            
            logger.info(f"📊 Risk Assessment: ${risk_amount:.2f} risk, {risk_reward_ratio:.1f}:1 RR")
            return trade_risk
            
        except Exception as e:
            logger.error(f"Error assessing trade risk: {e}")
            return None
    
    def validate_trade(self, trade_risk: TradeRisk) -> Tuple[bool, str]:
        """Validate if trade meets all safety requirements"""
        try:
            # Check position size limits
            if trade_risk.position_size < self.limits.min_position_size:
                return False, f"Position too small: ${trade_risk.position_size:.2f} < ${self.limits.min_position_size}"
            
            if trade_risk.position_size > self.limits.max_position_size:
                return False, f"Position too large: ${trade_risk.position_size:.2f} > ${self.limits.max_position_size}"
            
            # Check risk per trade
            if trade_risk.risk_amount > self.limits.max_risk_per_trade:
                return False, f"Risk too high: ${trade_risk.risk_amount:.2f} > ${self.limits.max_risk_per_trade:.2f}"
            
            # Check daily trade limit
            if self.daily_trades >= self.limits.max_daily_trades:
                return False, f"Daily trade limit reached: {self.daily_trades}/{self.limits.max_daily_trades}"
            
            # Check concurrent positions
            if len(self.active_positions) >= self.limits.max_concurrent_positions:
                return False, f"Too many positions: {len(self.active_positions)}/{self.limits.max_concurrent_positions}"
            
            # Check daily loss limit
            if self.daily_pnl <= -self.limits.max_daily_loss:
                return False, f"Daily loss limit hit: ${self.daily_pnl:.2f} <= ${-self.limits.max_daily_loss:.2f}"
            
            # Check portfolio stop loss
            portfolio_loss_pct = (self.initial_balance - self.portfolio_value) / self.initial_balance
            if portfolio_loss_pct >= self.limits.portfolio_stop_loss:
                return False, f"Portfolio stop loss triggered: {portfolio_loss_pct:.1%} loss"
            
            # Check risk-reward ratio
            if trade_risk.risk_reward_ratio < 1.5:
                return False, f"Poor risk-reward: {trade_risk.risk_reward_ratio:.1f}:1 < 1.5:1"
            
            # Check confidence level
            if trade_risk.confidence_level < 0.6:
                return False, f"Low confidence: {trade_risk.confidence_level:.1%} < 60%"
            
            return True, "All safety checks passed"
            
        except Exception as e:
            logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def register_trade(self, trade_risk: TradeRisk, trade_id: str):
        """Register a new trade for tracking"""
        try:
            self.active_positions[trade_id] = {
                'symbol': trade_risk.symbol,
                'position_size': trade_risk.position_size,
                'risk_amount': trade_risk.risk_amount,
                'stop_loss': trade_risk.stop_loss_price,
                'take_profit': trade_risk.take_profit_price,
                'entry_time': datetime.now(),
                'confidence': trade_risk.confidence_level
            }
            
            self.daily_trades += 1
            logger.info(f"✅ Trade registered: {trade_id} for {trade_risk.symbol}")
            
        except Exception as e:
            logger.error(f"Error registering trade: {e}")
    
    def close_position(self, trade_id: str, exit_price: float, exit_reason: str):
        """Close position and update P&L"""
        try:
            if trade_id not in self.active_positions:
                logger.warning(f"Trade {trade_id} not found in active positions")
                return
            
            position = self.active_positions[trade_id]
            
            # Calculate P&L (simplified)
            entry_value = position['position_size']
            # This would need actual price calculation based on entry vs exit
            pnl = (exit_price - entry_value) / entry_value * position['position_size']
            
            self.daily_pnl += pnl
            self.portfolio_value += pnl
            
            # Remove from active positions
            del self.active_positions[trade_id]
            
            logger.info(f"🔒 Position closed: {trade_id}, P&L: ${pnl:.2f}, Reason: {exit_reason}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def get_safety_status(self) -> Dict:
        """Get current safety status"""
        portfolio_loss_pct = (self.initial_balance - self.portfolio_value) / self.initial_balance
        
        return {
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'active_positions': len(self.active_positions),
            'portfolio_loss_pct': portfolio_loss_pct,
            'daily_limit_remaining': self.limits.max_daily_trades - self.daily_trades,
            'risk_budget_remaining': self.limits.max_daily_loss + self.daily_pnl,
            'emergency_stop_triggered': portfolio_loss_pct >= self.limits.portfolio_stop_loss
        }