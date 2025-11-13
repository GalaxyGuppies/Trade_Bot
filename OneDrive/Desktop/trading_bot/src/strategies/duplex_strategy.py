"""
Duplex Trading Strategy System
Simultaneously evaluates scalping and swing opportunities, selecting optimal strategy
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class TradeStrategy(Enum):
    SCALP = "scalp"
    SWING = "swing"
    SKIP = "skip"

@dataclass
class StrategySignal:
    """Container for strategy analysis results"""
    strategy: TradeStrategy
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    hold_duration: str  # "minutes", "hours", "days"
    reasoning: str
    technical_score: float
    fundamental_score: float

class DuplexTradingStrategy:
    """
    Duplex system that evaluates both scalping and swing opportunities
    Automatically selects the best strategy based on market conditions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Strategy thresholds - aligned with config
        self.scalp_thresholds = {
            'min_volatility': 4.0,  # Lowered from 6.0
            'max_volatility': 15.0,  # Increased from 12.0
            'min_volume': 50000,     # Reduced from 1M to match config
            'min_liquidity': 50000,  # Reduced from 100K to match config
            'max_market_cap': 5000000,  # Reduced from 50M to match config
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'max_hold_minutes': 60
        }
        
        self.swing_thresholds = {
            'min_volatility': 3.0,     # Lowered from 4.0
            'max_volatility': 10.0,    # Increased from 8.0
            'min_volume': 50000,       # Reduced from 500K to match config
            'min_liquidity': 50000,    # Reduced from 200K to match config
            'min_market_cap': 100000,  # Matches config min
            'rsi_oversold': 35,
            'rsi_overbought': 65,
            'min_hold_hours': 4,
            'max_hold_days': 7
        }
        
        # Position sizing
        self.scalp_sizing = {
            'base_percent': 3.0,  # 3% of capital for scalping
            'max_position': 2000,  # $2000 max per scalp
            'volatility_adjustment': 0.5,
            'confidence_multiplier': 1.5,  # Scale up to 1.5x for high confidence
            'min_confidence_size': 0.7      # Min 70% of base size for low confidence
        }
        
        self.swing_sizing = {
            'base_percent': 5.0,  # 5% of capital for swing
            'max_position': 5000,  # $5000 max per swing
            'volatility_adjustment': 0.3,
            'confidence_multiplier': 2.0,   # Scale up to 2x for high confidence swings
            'min_confidence_size': 0.6      # Min 60% of base size for low confidence
        }
        
        logger.info("🔄 Duplex Trading Strategy initialized")
    
    def evaluate_opportunity(self, candidate: Dict, technical_data: Dict, 
                           market_conditions: Dict, available_capital: float) -> StrategySignal:
        """
        Evaluate a candidate for both scalping and swing opportunities
        Return the optimal strategy with reasoning
        """
        try:
            # Analyze scalping potential
            scalp_signal = self._analyze_scalping_opportunity(
                candidate, technical_data, available_capital
            )
            
            # Analyze swing potential  
            swing_signal = self._analyze_swing_opportunity(
                candidate, technical_data, market_conditions, available_capital
            )
            
            # Select optimal strategy
            optimal_signal = self._select_optimal_strategy(
                candidate, scalp_signal, swing_signal, market_conditions
            )
            
            return optimal_signal
            
        except Exception as e:
            logger.error(f"❌ Error evaluating opportunity for {candidate.get('symbol', 'UNKNOWN')}: {e}")
            return StrategySignal(
                strategy=TradeStrategy.SKIP,
                confidence=0.0,
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                hold_duration="none",
                reasoning=f"Evaluation error: {str(e)}",
                technical_score=0.0,
                fundamental_score=0.0
            )
    
    def _analyze_scalping_opportunity(self, candidate: Dict, technical_data: Dict, 
                                    available_capital: float) -> StrategySignal:
        """Analyze token for scalping potential"""
        symbol = candidate.get('symbol', 'UNKNOWN')
        
        # Technical scoring for scalping
        rsi = technical_data.get('rsi', 50)
        bb_position = technical_data.get('bb_position', 0.5)
        macd_signal = technical_data.get('macd_signal', 'NEUTRAL')
        volume_24h = candidate.get('volume_24h', 0)
        volatility = candidate.get('volatility_score', 5.0)
        market_cap = candidate.get('market_cap', 0)
        liquidity = candidate.get('liquidity_usd', 0)
        
        # Scalping criteria scoring
        scalp_score = 0.0
        reasoning_parts = []
        
        # Volume check (20 points)
        if volume_24h >= self.scalp_thresholds['min_volume']:
            volume_score = min(20, (volume_24h / self.scalp_thresholds['min_volume']) * 10)
            scalp_score += volume_score
            reasoning_parts.append(f"Volume: {volume_24h/1e6:.1f}M ✓")
        else:
            reasoning_parts.append(f"Volume: {volume_24h/1e6:.1f}M ✗")
        
        # Volatility check (25 points)
        if self.scalp_thresholds['min_volatility'] <= volatility <= self.scalp_thresholds['max_volatility']:
            vol_score = 25
            scalp_score += vol_score
            reasoning_parts.append(f"Volatility: {volatility:.1f} ✓")
        else:
            reasoning_parts.append(f"Volatility: {volatility:.1f} ✗")
        
        # Technical signals (30 points)
        tech_score = 0
        if rsi <= self.scalp_thresholds['rsi_oversold'] or rsi >= self.scalp_thresholds['rsi_overbought']:
            tech_score += 15
            reasoning_parts.append(f"RSI extreme: {rsi:.0f} ✓")
        
        if bb_position <= 0.2 or bb_position >= 0.8:
            tech_score += 10
            reasoning_parts.append(f"BB extreme: {bb_position:.2f} ✓")
        
        if macd_signal in ['BULLISH', 'BEARISH']:
            tech_score += 5
            reasoning_parts.append(f"MACD: {macd_signal} ✓")
        
        scalp_score += tech_score
        
        # Market cap check (15 points)
        if market_cap <= self.scalp_thresholds['max_market_cap']:
            mc_score = 15
            scalp_score += mc_score
            reasoning_parts.append(f"MC: ${market_cap/1e6:.1f}M ✓")
        else:
            reasoning_parts.append(f"MC: ${market_cap/1e6:.1f}M ✗")
        
        # Liquidity check (10 points)
        if liquidity >= self.scalp_thresholds['min_liquidity']:
            liq_score = 10
            scalp_score += liq_score
            reasoning_parts.append(f"Liquidity: ${liquidity/1e3:.0f}K ✓")
        else:
            reasoning_parts.append(f"Liquidity: ${liquidity/1e3:.0f}K ✗")
        
        # Calculate confidence early (needed for position sizing)
        confidence = scalp_score / 100.0  # Convert to 0-1 scale
        
        # Calculate position size with confidence scaling
        base_size = available_capital * (self.scalp_sizing['base_percent'] / 100)
        vol_adjustment = 1.0 - (volatility / 10.0) * self.scalp_sizing['volatility_adjustment']
        
        # Scale position size based on confidence (0.7x to 1.5x multiplier)
        confidence_multiplier = max(
            self.scalp_sizing['min_confidence_size'],
            min(self.scalp_sizing['confidence_multiplier'], confidence * 1.5)
        )
        
        position_size = min(
            base_size * vol_adjustment * confidence_multiplier, 
            self.scalp_sizing['max_position']
        )
        
        logger.info(f"   💰 Position sizing: base=${base_size:.2f}, vol_adj={vol_adjustment:.2f}, conf_mult={confidence_multiplier:.2f}, final=${position_size:.2f}")
        
        # Calculate dynamic stops and targets for scalping based on volatility
        current_price = candidate.get('price_usd', 1.0)
        
        # Dynamic stop loss: 2-4% based on volatility (higher vol = tighter stops)
        dynamic_stop_pct = 0.02 + (volatility / 100.0)  # 2-4% range
        stop_loss = current_price * (1.0 - dynamic_stop_pct)
        
        # Dynamic take profit: 4-8% based on volatility and confidence
        dynamic_tp_pct = 0.04 + (volatility / 100.0) + (confidence * 0.02)  # 4-8% range
        take_profit = current_price * (1.0 + dynamic_tp_pct)
        
        # Debug logging for scoring
        logger.info(f"🔍 SCALP analysis for {symbol}: Score={scalp_score:.0f}/100, Confidence={confidence:.2f}")
        logger.info(f"   💰 Volume: {volume_24h/1e6:.1f}M (need {self.scalp_thresholds['min_volume']/1e6:.1f}M)")
        logger.info(f"   📊 Market cap: ${market_cap/1e6:.1f}M (max ${self.scalp_thresholds['max_market_cap']/1e6:.1f}M)")
        logger.info(f"   🌊 Volatility: {volatility:.1f} (range {self.scalp_thresholds['min_volatility']}-{self.scalp_thresholds['max_volatility']})")
        logger.info(f"   💧 Liquidity: ${liquidity/1e3:.0f}K (need ${self.scalp_thresholds['min_liquidity']/1e3:.0f}K)")
        
        return StrategySignal(
            strategy=TradeStrategy.SCALP,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            hold_duration="minutes",
            reasoning=f"SCALP: {', '.join(reasoning_parts)} (Score: {scalp_score:.0f}/100)",
            technical_score=tech_score / 30.0,
            fundamental_score=(scalp_score - tech_score) / 70.0
        )
    
    def _analyze_swing_opportunity(self, candidate: Dict, technical_data: Dict,
                                 market_conditions: Dict, available_capital: float) -> StrategySignal:
        """Analyze token for swing trading potential"""
        symbol = candidate.get('symbol', 'UNKNOWN')
        
        # Technical data
        rsi = technical_data.get('rsi', 50)
        bb_position = technical_data.get('bb_position', 0.5)
        macd_signal = technical_data.get('macd_signal', 'NEUTRAL')
        volume_24h = candidate.get('volume_24h', 0)
        volatility = candidate.get('volatility_score', 5.0)
        market_cap = candidate.get('market_cap', 0)
        liquidity = candidate.get('liquidity_usd', 0)
        
        # Swing criteria scoring
        swing_score = 0.0
        reasoning_parts = []
        
        # Market cap check (25 points) - prefer larger caps for swing
        if market_cap >= self.swing_thresholds['min_market_cap']:
            mc_score = min(25, (market_cap / self.swing_thresholds['min_market_cap']) * 15)
            swing_score += mc_score
            reasoning_parts.append(f"MC: ${market_cap/1e6:.1f}M ✓")
        else:
            reasoning_parts.append(f"MC: ${market_cap/1e6:.1f}M ✗")
        
        # Volatility check (20 points) - prefer moderate volatility
        if self.swing_thresholds['min_volatility'] <= volatility <= self.swing_thresholds['max_volatility']:
            vol_score = 20
            swing_score += vol_score
            reasoning_parts.append(f"Volatility: {volatility:.1f} ✓")
        elif volatility > self.swing_thresholds['max_volatility']:
            # Penalize high volatility for swing
            vol_score = max(0, 20 - (volatility - self.swing_thresholds['max_volatility']) * 2)
            swing_score += vol_score
            reasoning_parts.append(f"Volatility: {volatility:.1f} ⚠️")
        else:
            reasoning_parts.append(f"Volatility: {volatility:.1f} ✗")
        
        # Technical trend analysis (30 points)
        trend_score = 0
        
        # RSI trend (not extreme levels)
        if 40 <= rsi <= 60:
            trend_score += 10
            reasoning_parts.append(f"RSI neutral: {rsi:.0f} ✓")
        elif self.swing_thresholds['rsi_oversold'] <= rsi <= self.swing_thresholds['rsi_overbought']:
            trend_score += 15
            reasoning_parts.append(f"RSI swing range: {rsi:.0f} ✓")
        
        # MACD for trend direction
        if macd_signal == 'BULLISH':
            trend_score += 15
            reasoning_parts.append(f"MACD bullish trend ✓")
        elif macd_signal == 'BEARISH':
            trend_score += 5  # Can swing short
            reasoning_parts.append(f"MACD bearish trend ⚠️")
        
        swing_score += trend_score
        
        # Volume consistency (15 points)
        if volume_24h >= self.swing_thresholds['min_volume']:
            vol_consistency = 15
            swing_score += vol_consistency
            reasoning_parts.append(f"Volume: {volume_24h/1e6:.1f}M ✓")
        else:
            reasoning_parts.append(f"Volume: {volume_24h/1e6:.1f}M ✗")
        
        # Liquidity for swing trades (10 points)
        if liquidity >= self.swing_thresholds['min_liquidity']:
            liq_score = 10
            swing_score += liq_score
            reasoning_parts.append(f"Liquidity: ${liquidity/1e3:.0f}K ✓")
        else:
            reasoning_parts.append(f"Liquidity: ${liquidity/1e3:.0f}K ✗")
        
        # Calculate confidence early (needed for position sizing)
        confidence = swing_score / 100.0
        
        # Calculate position size for swing with confidence scaling
        base_size = available_capital * (self.swing_sizing['base_percent'] / 100)
        vol_adjustment = 1.0 - (volatility / 10.0) * self.swing_sizing['volatility_adjustment']
        
        # Scale position size based on confidence (0.6x to 2x multiplier)
        confidence_multiplier = max(
            self.swing_sizing['min_confidence_size'],
            min(self.swing_sizing['confidence_multiplier'], confidence * 2.0)
        )
        
        position_size = min(
            base_size * vol_adjustment * confidence_multiplier, 
            self.swing_sizing['max_position']
        )
        
        logger.info(f"   💰 SWING sizing: base=${base_size:.2f}, vol_adj={vol_adjustment:.2f}, conf_mult={confidence_multiplier:.2f}, final=${position_size:.2f}")
        
        # Calculate dynamic stops and targets for swing trading
        current_price = candidate.get('price_usd', 1.0)
        
        # Dynamic stop loss: 8-15% based on volatility and market cap
        mc_factor = min(1.0, market_cap / 10000000)  # Smaller caps need wider stops
        dynamic_stop_pct = 0.08 + (volatility / 100.0) + ((1.0 - mc_factor) * 0.05)  # 8-18% range
        stop_loss = current_price * (1.0 - dynamic_stop_pct)
        
        # Dynamic take profit: 15-35% based on volatility, confidence, and market cap
        base_tp = 0.15  # 15% base
        vol_bonus = (volatility / 100.0) * 0.5  # Up to +5% for high volatility
        conf_bonus = confidence * 0.1  # Up to +10% for high confidence  
        mc_bonus = (1.0 - mc_factor) * 0.1  # Up to +10% for smaller caps
        
        dynamic_tp_pct = base_tp + vol_bonus + conf_bonus + mc_bonus
        take_profit = current_price * (1.0 + dynamic_tp_pct)
        
        logger.info(f"   🎯 SWING targets: SL={dynamic_stop_pct*100:.1f}% (${stop_loss:.6f}), TP={dynamic_tp_pct*100:.1f}% (${take_profit:.6f})")
        
        return StrategySignal(
            strategy=TradeStrategy.SWING,
            confidence=confidence,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            hold_duration="days",
            reasoning=f"SWING: {', '.join(reasoning_parts)} (Score: {swing_score:.0f}/100)",
            technical_score=trend_score / 30.0,
            fundamental_score=(swing_score - trend_score) / 70.0
        )
    
    def _select_optimal_strategy(self, candidate: Dict, scalp_signal: StrategySignal,
                               swing_signal: StrategySignal, market_conditions: Dict) -> StrategySignal:
        """Select the optimal strategy based on analysis"""
        symbol = candidate.get('symbol', 'UNKNOWN')
        
        # Minimum confidence thresholds - adjusted for realistic market conditions
        min_scalp_confidence = 0.35  # 35% threshold for scalping
        min_swing_confidence = 0.40  # 40% threshold for swing
        
        # Market condition adjustments
        market_volatility = market_conditions.get('volatility', 'medium')
        
        # Decision logic
        scalp_viable = scalp_signal.confidence >= min_scalp_confidence
        swing_viable = swing_signal.confidence >= min_swing_confidence
        
        reasoning_parts = []
        
        if not scalp_viable and not swing_viable:
            reasoning_parts.append(f"Both strategies below threshold (S:{scalp_signal.confidence:.2f}, W:{swing_signal.confidence:.2f})")
            return StrategySignal(
                strategy=TradeStrategy.SKIP,
                confidence=0.0,
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                hold_duration="none",
                reasoning=f"SKIP: {', '.join(reasoning_parts)}",
                technical_score=0.0,
                fundamental_score=0.0
            )
        
        # If only one strategy is viable
        if scalp_viable and not swing_viable:
            reasoning_parts.append(f"Only scalping viable (S:{scalp_signal.confidence:.2f} > W:{swing_signal.confidence:.2f})")
            scalp_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
            return scalp_signal
        
        if swing_viable and not scalp_viable:
            reasoning_parts.append(f"Only swing viable (W:{swing_signal.confidence:.2f} > S:{scalp_signal.confidence:.2f})")
            swing_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
            return swing_signal
        
        # Both strategies viable - select based on conditions
        confidence_diff = abs(scalp_signal.confidence - swing_signal.confidence)
        
        # If confidence is very close, use market conditions
        if confidence_diff < 0.1:
            if market_volatility == 'high':
                reasoning_parts.append(f"High volatility favors scalping (S:{scalp_signal.confidence:.2f} ≈ W:{swing_signal.confidence:.2f})")
                scalp_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
                return scalp_signal
            else:
                reasoning_parts.append(f"Stable market favors swing (W:{swing_signal.confidence:.2f} ≈ S:{scalp_signal.confidence:.2f})")
                swing_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
                return swing_signal
        
        # Select higher confidence strategy
        if scalp_signal.confidence > swing_signal.confidence:
            reasoning_parts.append(f"Scalping has higher confidence (S:{scalp_signal.confidence:.2f} > W:{swing_signal.confidence:.2f})")
            scalp_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
            return scalp_signal
        else:
            reasoning_parts.append(f"Swing has higher confidence (W:{swing_signal.confidence:.2f} > S:{scalp_signal.confidence:.2f})")
            swing_signal.reasoning += f" | SELECTED: {', '.join(reasoning_parts)}"
            return swing_signal
    
    def get_strategy_stats(self) -> Dict:
        """Get statistics about strategy selection"""
        return {
            'scalp_thresholds': self.scalp_thresholds,
            'swing_thresholds': self.swing_thresholds,
            'scalp_sizing': self.scalp_sizing,
            'swing_sizing': self.swing_sizing
        }