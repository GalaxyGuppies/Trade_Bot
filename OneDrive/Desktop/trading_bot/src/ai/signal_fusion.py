"""
Multi-Modal AI Signal Fusion Engine
Advanced signal processing that combines multiple data sources with AI weighting
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import json

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Individual signal data structure"""
    source: str
    strength: float  # -1 to 1 (bearish to bullish)
    confidence: float  # 0 to 1
    timestamp: datetime
    metadata: Dict

@dataclass
class MarketRegime:
    """Market regime classification"""
    regime_type: str  # 'bull', 'bear', 'sideways', 'volatile'
    volatility_level: str  # 'low', 'medium', 'high', 'extreme'
    trend_strength: float  # 0 to 1
    mean_reversion: float  # 0 to 1 (trending vs mean reverting)

class MarketMicrostructureAnalyzer:
    """Analyzes market microstructure signals"""
    
    def __init__(self):
        self.lookback_periods = [5, 15, 30, 60]  # minutes
        
    async def get_signal(self, symbol: str, market_data: Dict) -> Signal:
        """Analyze market microstructure for trading signals"""
        try:
            orderbook = market_data.get('orderbook', {})
            trades = market_data.get('recent_trades', [])
            
            # Calculate microstructure indicators
            bid_ask_spread = self._calculate_spread(orderbook)
            order_flow_imbalance = self._calculate_order_flow_imbalance(orderbook)
            trade_intensity = self._calculate_trade_intensity(trades)
            volume_profile = self._analyze_volume_profile(trades)
            
            # Combine into signal strength
            signal_strength = self._combine_microstructure_signals(
                spread=bid_ask_spread,
                flow_imbalance=order_flow_imbalance,
                intensity=trade_intensity,
                volume_profile=volume_profile
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(orderbook, trades)
            
            return Signal(
                source='market_microstructure',
                strength=signal_strength,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'spread': bid_ask_spread,
                    'flow_imbalance': order_flow_imbalance,
                    'trade_intensity': trade_intensity,
                    'volume_profile': volume_profile
                }
            )
            
        except Exception as e:
            logger.error(f"Error in microstructure analysis: {e}")
            return Signal('market_microstructure', 0.0, 0.0, datetime.now(), {})
    
    def _calculate_spread(self, orderbook: Dict) -> float:
        """Calculate normalized bid-ask spread"""
        try:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.0
                
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            mid_price = (best_bid + best_ask) / 2
            
            spread = (best_ask - best_bid) / mid_price
            return min(spread * 1000, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.0
    
    def _calculate_order_flow_imbalance(self, orderbook: Dict) -> float:
        """Calculate order flow imbalance (buying vs selling pressure)"""
        try:
            bids = orderbook.get('bids', [])[:10]  # Top 10 levels
            asks = orderbook.get('asks', [])[:10]
            
            if not bids or not asks:
                return 0.0
                
            bid_volume = sum(float(bid[1]) for bid in bids)
            ask_volume = sum(float(ask[1]) for ask in asks)
            total_volume = bid_volume + ask_volume
            
            if total_volume == 0:
                return 0.0
                
            # Return imbalance (-1 = heavy selling, +1 = heavy buying)
            imbalance = (bid_volume - ask_volume) / total_volume
            return np.clip(imbalance, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_trade_intensity(self, trades: List) -> float:
        """Calculate recent trade intensity"""
        try:
            if not trades:
                return 0.0
                
            now = datetime.now()
            recent_trades = [
                trade for trade in trades
                if (now - datetime.fromtimestamp(trade.get('timestamp', 0))).seconds < 300  # 5 minutes
            ]
            
            if not recent_trades:
                return 0.0
                
            # Calculate trade intensity (trades per minute)
            intensity = len(recent_trades) / 5.0  # 5 minute window
            return min(intensity / 100, 1.0)  # Normalize
            
        except Exception:
            return 0.0
    
    def _analyze_volume_profile(self, trades: List) -> float:
        """Analyze volume distribution for market bias"""
        try:
            if not trades:
                return 0.0
                
            buy_volume = sum(
                float(trade.get('amount', 0)) 
                for trade in trades 
                if trade.get('side') == 'buy'
            )
            sell_volume = sum(
                float(trade.get('amount', 0)) 
                for trade in trades 
                if trade.get('side') == 'sell'
            )
            
            total_volume = buy_volume + sell_volume
            if total_volume == 0:
                return 0.0
                
            # Return volume bias (-1 = heavy selling, +1 = heavy buying)
            bias = (buy_volume - sell_volume) / total_volume
            return np.clip(bias, -1.0, 1.0)
            
        except Exception:
            return 0.0
    
    def _combine_microstructure_signals(self, **kwargs) -> float:
        """Combine microstructure signals into single strength value"""
        signals = []
        
        # Spread signal (tight spread = bullish)
        spread = kwargs.get('spread', 0)
        spread_signal = max(0, 1 - spread * 10)  # Invert spread
        signals.append(spread_signal * 0.2)
        
        # Flow imbalance (positive = bullish)
        flow = kwargs.get('flow_imbalance', 0)
        signals.append(flow * 0.3)
        
        # Trade intensity (high intensity amplifies other signals)
        intensity = kwargs.get('intensity', 0)
        intensity_multiplier = 1 + (intensity * 0.5)
        
        # Volume profile (positive = bullish)
        volume_bias = kwargs.get('volume_profile', 0)
        signals.append(volume_bias * 0.5)
        
        # Combine signals
        combined = sum(signals) * intensity_multiplier
        return np.clip(combined, -1.0, 1.0)
    
    def _calculate_confidence(self, orderbook: Dict, trades: List) -> float:
        """Calculate confidence based on data quality"""
        try:
            confidence_factors = []
            
            # Orderbook depth
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            depth_factor = min(len(bids) + len(asks), 20) / 20
            confidence_factors.append(depth_factor)
            
            # Trade frequency
            trade_factor = min(len(trades), 100) / 100
            confidence_factors.append(trade_factor)
            
            # Data freshness (assume fresh for now)
            confidence_factors.append(1.0)
            
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5

class WhaleTracker:
    """Tracks whale wallet movements and patterns"""
    
    def __init__(self):
        self.known_whales = self._load_whale_addresses()
        self.whale_patterns = {}
        
    async def get_signal(self, symbol: str) -> Signal:
        """Get whale movement signal for symbol"""
        try:
            # Get recent whale transactions
            whale_activity = await self._get_whale_activity(symbol)
            
            # Analyze whale behavior patterns
            behavior_signal = self._analyze_whale_behavior(whale_activity)
            
            # Calculate signal strength
            signal_strength = self._calculate_whale_signal(behavior_signal)
            
            # Calculate confidence
            confidence = self._calculate_whale_confidence(whale_activity)
            
            return Signal(
                source='whale_tracker',
                strength=signal_strength,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'whale_activity': whale_activity,
                    'behavior_pattern': behavior_signal
                }
            )
            
        except Exception as e:
            logger.error(f"Error in whale tracking: {e}")
            return Signal('whale_tracker', 0.0, 0.0, datetime.now(), {})
    
    def _load_whale_addresses(self) -> List[str]:
        """Load known whale wallet addresses"""
        # In production, this would load from a database
        return [
            '0x8b83de7649d23b28b3ee4c7b1e7e2d07d57b6c8e',  # Example whale addresses
            '0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208',
            # Add more whale addresses
        ]
    
    async def _get_whale_activity(self, symbol: str) -> Dict:
        """Get recent whale activity for symbol"""
        # Placeholder - in production would query blockchain APIs
        return {
            'large_transfers': [],
            'accumulation_pattern': 'neutral',
            'distribution_pattern': 'neutral',
            'net_flow': 0.0
        }
    
    def _analyze_whale_behavior(self, activity: Dict) -> str:
        """Analyze whale behavior patterns"""
        # Simplified pattern recognition
        large_transfers = activity.get('large_transfers', [])
        
        if len(large_transfers) > 5:
            return 'high_activity'
        elif len(large_transfers) > 2:
            return 'moderate_activity'
        else:
            return 'low_activity'
    
    def _calculate_whale_signal(self, behavior: str) -> float:
        """Convert whale behavior to signal strength"""
        behavior_signals = {
            'high_activity': 0.7,
            'moderate_activity': 0.3,
            'low_activity': 0.0
        }
        return behavior_signals.get(behavior, 0.0)
    
    def _calculate_whale_confidence(self, activity: Dict) -> float:
        """Calculate confidence in whale signal"""
        # Based on data quality and recency
        return 0.8  # Placeholder

class MarketRegimeDetector:
    """Detects current market regime for signal weighting"""
    
    def __init__(self):
        self.regime_history = []
        
    async def detect_market_regime(self, market_data: Dict) -> MarketRegime:
        """Detect current market regime"""
        try:
            # Get price history
            prices = market_data.get('price_history', [])
            volumes = market_data.get('volume_history', [])
            
            if len(prices) < 50:  # Need enough data
                return MarketRegime('sideways', 'medium', 0.5, 0.5)
            
            # Calculate regime indicators
            trend_strength = self._calculate_trend_strength(prices)
            volatility_level = self._calculate_volatility_level(prices)
            mean_reversion = self._calculate_mean_reversion_tendency(prices)
            
            # Classify regime
            regime_type = self._classify_regime(trend_strength, volatility_level)
            volatility_classification = self._classify_volatility(volatility_level)
            
            return MarketRegime(
                regime_type=regime_type,
                volatility_level=volatility_classification,
                trend_strength=trend_strength,
                mean_reversion=mean_reversion
            )
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return MarketRegime('sideways', 'medium', 0.5, 0.5)
    
    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength (0 = no trend, 1 = strong trend)"""
        if len(prices) < 20:
            return 0.5
            
        # Use linear regression slope
        x = np.arange(len(prices))
        slope, _ = np.polyfit(x, prices, 1)
        
        # Normalize slope relative to price level
        normalized_slope = abs(slope) / np.mean(prices)
        return min(normalized_slope * 1000, 1.0)
    
    def _calculate_volatility_level(self, prices: List[float]) -> float:
        """Calculate volatility level"""
        if len(prices) < 20:
            return 0.5
            
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(24 * 365)  # Annualized
        return min(volatility, 2.0) / 2.0  # Normalize to 0-1
    
    def _calculate_mean_reversion_tendency(self, prices: List[float]) -> float:
        """Calculate mean reversion tendency"""
        if len(prices) < 20:
            return 0.5
            
        # Calculate Hurst exponent (simplified)
        price_series = pd.Series(prices)
        mean_price = price_series.rolling(10).mean()
        deviations = price_series - mean_price
        
        # Count mean reversion events
        reversion_count = 0
        for i in range(1, len(deviations) - 1):
            if deviations.iloc[i] != 0:
                if (deviations.iloc[i] > 0 and deviations.iloc[i+1] < deviations.iloc[i]) or \
                   (deviations.iloc[i] < 0 and deviations.iloc[i+1] > deviations.iloc[i]):
                    reversion_count += 1
        
        return reversion_count / max(len(deviations) - 2, 1)
    
    def _classify_regime(self, trend_strength: float, volatility: float) -> str:
        """Classify market regime based on trend and volatility"""
        if trend_strength > 0.6:
            return 'bull' if trend_strength > 0 else 'bear'
        elif volatility > 0.7:
            return 'volatile'
        else:
            return 'sideways'
    
    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility level"""
        if volatility < 0.3:
            return 'low'
        elif volatility < 0.6:
            return 'medium'
        elif volatility < 0.9:
            return 'high'
        else:
            return 'extreme'

class QuantumSignalFusion:
    """Main signal fusion engine using AI"""
    
    def __init__(self):
        self.signals = {
            'market_microstructure': MarketMicrostructureAnalyzer(),
            'whale_tracker': WhaleTracker(),
        }
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector()
        
        # Initialize AI models
        self.fusion_model = None
        self.scaler = StandardScaler()
        self.signal_weights = {}
        
        # Load or initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models for signal fusion"""
        try:
            # Try to load existing model
            self.fusion_model = joblib.load('models/signal_fusion_model.pkl')
            self.scaler = joblib.load('models/signal_scaler.pkl')
            
            with open('models/signal_weights.json', 'r') as f:
                self.signal_weights = json.load(f)
                
        except FileNotFoundError:
            # Initialize new model
            self.fusion_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Default signal weights
            self.signal_weights = {
                'bull': {
                    'market_microstructure': 0.3,
                    'whale_tracker': 0.4,
                    'social_sentiment': 0.2,
                    'technical_patterns': 0.1
                },
                'bear': {
                    'market_microstructure': 0.4,
                    'whale_tracker': 0.3,
                    'social_sentiment': 0.2,
                    'technical_patterns': 0.1
                },
                'sideways': {
                    'market_microstructure': 0.2,
                    'whale_tracker': 0.2,
                    'social_sentiment': 0.3,
                    'technical_patterns': 0.3
                },
                'volatile': {
                    'market_microstructure': 0.5,
                    'whale_tracker': 0.2,
                    'social_sentiment': 0.2,
                    'technical_patterns': 0.1
                }
            }
    
    async def fuse_signals(self, symbol: str, market_data: Dict) -> Dict:
        """Main signal fusion method"""
        try:
            # Detect market regime
            regime = await self.regime_detector.detect_market_regime(market_data)
            
            # Collect signals from all sources
            signals = {}
            signal_metadata = {}
            
            for signal_name, analyzer in self.signals.items():
                if signal_name == 'market_microstructure':
                    signal = await analyzer.get_signal(symbol, market_data)
                else:
                    signal = await analyzer.get_signal(symbol)
                
                signals[signal_name] = {
                    'strength': signal.strength,
                    'confidence': signal.confidence
                }
                signal_metadata[signal_name] = signal.metadata
            
            # Get regime-specific weights
            weights = self.signal_weights.get(regime.regime_type, self.signal_weights['sideways'])
            
            # Calculate weighted signal
            fused_signal = self._calculate_weighted_signal(signals, weights, regime)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(signals, weights)
            
            return {
                'fused_signal': fused_signal,
                'confidence': overall_confidence,
                'regime': regime,
                'individual_signals': signals,
                'signal_weights': weights,
                'metadata': signal_metadata
            }
            
        except Exception as e:
            logger.error(f"Error in signal fusion: {e}")
            return {
                'fused_signal': 0.0,
                'confidence': 0.0,
                'regime': MarketRegime('sideways', 'medium', 0.5, 0.5),
                'individual_signals': {},
                'signal_weights': {},
                'metadata': {}
            }
    
    def _calculate_weighted_signal(self, signals: Dict, weights: Dict, regime: MarketRegime) -> float:
        """Calculate weighted signal strength"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for signal_name, signal_data in signals.items():
            if signal_name in weights:
                weight = weights[signal_name]
                confidence = signal_data['confidence']
                strength = signal_data['strength']
                
                # Weight by confidence and regime-specific weight
                effective_weight = weight * confidence
                weighted_sum += strength * effective_weight
                total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0
            
        # Apply regime-specific adjustments
        fused_signal = weighted_sum / total_weight
        
        # Adjust based on market regime
        if regime.regime_type == 'volatile':
            # Reduce signal strength in volatile markets
            fused_signal *= 0.7
        elif regime.volatility_level == 'extreme':
            # Further reduce in extreme volatility
            fused_signal *= 0.5
        
        return np.clip(fused_signal, -1.0, 1.0)
    
    def _calculate_overall_confidence(self, signals: Dict, weights: Dict) -> float:
        """Calculate overall confidence in the fused signal"""
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for signal_name, signal_data in signals.items():
            if signal_name in weights:
                weight = weights[signal_name]
                confidence = signal_data['confidence']
                
                weighted_confidence += confidence * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
            
        return weighted_confidence / total_weight
    
    def get_signal_interpretation(self, fused_result: Dict) -> str:
        """Get human-readable interpretation of the fused signal"""
        signal = fused_result['fused_signal']
        confidence = fused_result['confidence']
        regime = fused_result['regime']
        
        # Determine signal strength
        if abs(signal) < 0.2:
            strength = "weak"
        elif abs(signal) < 0.5:
            strength = "moderate"
        elif abs(signal) < 0.8:
            strength = "strong"
        else:
            strength = "very strong"
        
        # Determine direction
        direction = "bullish" if signal > 0 else "bearish"
        
        # Determine confidence level
        if confidence < 0.3:
            conf_level = "low"
        elif confidence < 0.6:
            conf_level = "medium"
        else:
            conf_level = "high"
        
        return f"{strength.title()} {direction} signal with {conf_level} confidence in {regime.regime_type} market"

# Example usage and testing
async def test_signal_fusion():
    """Test the signal fusion engine"""
    fusion_engine = QuantumSignalFusion()
    
    # Mock market data
    mock_market_data = {
        'orderbook': {
            'bids': [['100.5', '1.2'], ['100.4', '2.1'], ['100.3', '1.8']],
            'asks': [['100.6', '1.1'], ['100.7', '2.0'], ['100.8', '1.5']]
        },
        'recent_trades': [
            {'side': 'buy', 'amount': '0.5', 'timestamp': datetime.now().timestamp()},
            {'side': 'sell', 'amount': '0.3', 'timestamp': datetime.now().timestamp()}
        ],
        'price_history': [100 + i * 0.1 for i in range(100)],
        'volume_history': [1000 + i * 10 for i in range(100)]
    }
    
    result = await fusion_engine.fuse_signals('BTC/USDT', mock_market_data)
    interpretation = fusion_engine.get_signal_interpretation(result)
    
    print(f"Fused Signal: {result['fused_signal']:.3f}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Market Regime: {result['regime'].regime_type}")
    print(f"Interpretation: {interpretation}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_signal_fusion())