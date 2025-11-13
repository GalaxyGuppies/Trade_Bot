"""
Advanced Profit Optimization Module
Dynamic profit target adjustments based on momentum and market conditions
"""

import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProfitOptimizer:
    """
    Advanced profit optimization system that dynamically adjusts targets
    based on momentum, volume, volatility, and market conditions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Momentum thresholds
        self.momentum_thresholds = {
            'high_momentum': 0.05,      # 5% price change in monitoring period
            'extreme_momentum': 0.10,   # 10% price change
            'volume_surge': 2.0,        # 2x average volume
            'volatility_spike': 1.5     # 1.5x normal volatility
        }
        
        # Target adjustment multipliers
        self.target_multipliers = {
            'scalp': {
                'base_tp': 1.0,
                'momentum_bonus': 0.5,      # +50% target extension
                'volume_bonus': 0.3,        # +30% for volume surge
                'max_extension': 2.0        # Max 2x original target
            },
            'swing': {
                'base_tp': 1.0,
                'momentum_bonus': 0.7,      # +70% target extension
                'volume_bonus': 0.4,        # +40% for volume surge  
                'max_extension': 2.5        # Max 2.5x original target
            }
        }
        
        logger.info("🎯 Advanced profit optimizer initialized")
    
    def optimize_profit_targets(self, position: Dict, current_data: Dict, 
                              market_conditions: Dict) -> Tuple[float, Dict]:
        """
        Dynamically optimize profit targets based on current conditions
        Returns: (new_take_profit, optimization_info)
        """
        try:
            symbol = position.get('symbol', 'UNKNOWN')
            strategy = position.get('discovery_mode', 'scalping')
            entry_price = position['entry_price']
            current_price = position['current_price']
            original_tp = position['take_profit']
            
            # Calculate current momentum
            price_change_pct = ((current_price - entry_price) / entry_price)
            momentum_score = self._calculate_momentum_score(
                price_change_pct, current_data, position
            )
            
            # Get strategy-specific multipliers
            multipliers = self.target_multipliers.get(strategy, self.target_multipliers['scalp'])
            
            # Calculate target extension
            extension_factor = multipliers['base_tp']
            
            # Momentum bonus
            if momentum_score['momentum'] >= self.momentum_thresholds['extreme_momentum']:
                extension_factor += multipliers['momentum_bonus']
                logger.info(f"🚀 {symbol}: Extreme momentum detected ({momentum_score['momentum']*100:.1f}%)")
            elif momentum_score['momentum'] >= self.momentum_thresholds['high_momentum']:
                extension_factor += multipliers['momentum_bonus'] * 0.6
                logger.info(f"📈 {symbol}: High momentum detected ({momentum_score['momentum']*100:.1f}%)")
            
            # Volume surge bonus
            if momentum_score['volume_ratio'] >= self.momentum_thresholds['volume_surge']:
                extension_factor += multipliers['volume_bonus']
                logger.info(f"📊 {symbol}: Volume surge detected ({momentum_score['volume_ratio']:.1f}x)")
            
            # Cap the extension
            extension_factor = min(extension_factor, multipliers['max_extension'])
            
            # Calculate new take profit
            original_tp_pct = (original_tp - entry_price) / entry_price
            new_tp_pct = original_tp_pct * extension_factor
            new_take_profit = entry_price * (1 + new_tp_pct)
            
            # Only increase targets, never decrease
            if new_take_profit > original_tp:
                optimization_info = {
                    'extended': True,
                    'extension_factor': extension_factor,
                    'momentum_score': momentum_score,
                    'original_tp_pct': original_tp_pct * 100,
                    'new_tp_pct': new_tp_pct * 100,
                    'reasoning': f"Momentum: {momentum_score['momentum']*100:.1f}%, Volume: {momentum_score['volume_ratio']:.1f}x"
                }
                
                logger.info(f"🎯 {symbol}: Target extended from {original_tp_pct*100:.1f}% to {new_tp_pct*100:.1f}% (factor: {extension_factor:.2f})")
                return new_take_profit, optimization_info
            else:
                return original_tp, {'extended': False, 'reasoning': 'No extension criteria met'}
                
        except Exception as e:
            logger.error(f"❌ Error optimizing targets for {symbol}: {e}")
            return position['take_profit'], {'extended': False, 'error': str(e)}
    
    def _calculate_momentum_score(self, price_change_pct: float, 
                                current_data: Dict, position: Dict) -> Dict:
        """Calculate momentum score based on multiple factors"""
        
        # Price momentum (absolute change since entry)
        momentum = abs(price_change_pct)
        
        # Volume ratio (current vs average)
        current_volume = current_data.get('volume_24h', 0)
        avg_volume = position.get('avg_volume', current_volume)
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Volatility ratio
        current_volatility = current_data.get('volatility_score', 5.0)
        entry_volatility = position.get('entry_volatility', current_volatility)
        volatility_ratio = current_volatility / entry_volatility if entry_volatility > 0 else 1.0
        
        # Time factor (momentum should be recent)
        entry_time = position.get('entry_time', datetime.now())
        time_elapsed = (datetime.now() - entry_time).total_seconds() / 3600  # hours
        time_factor = max(0.5, 1.0 - (time_elapsed / 24))  # Decay over 24 hours
        
        return {
            'momentum': momentum * time_factor,
            'volume_ratio': volume_ratio,
            'volatility_ratio': volatility_ratio,
            'time_factor': time_factor,
            'time_elapsed_hours': time_elapsed
        }
    
    def should_extend_targets(self, position: Dict, price_change_pct: float) -> bool:
        """Quick check if position qualifies for target extension"""
        strategy = position.get('discovery_mode', 'scalping')
        
        if strategy == 'scalping':
            return price_change_pct > 0.03  # 3% for scalps
        else:
            return price_change_pct > 0.08  # 8% for swings
    
    def get_optimization_stats(self) -> Dict:
        """Get statistics about profit optimization"""
        return {
            'momentum_thresholds': self.momentum_thresholds,
            'target_multipliers': self.target_multipliers
        }