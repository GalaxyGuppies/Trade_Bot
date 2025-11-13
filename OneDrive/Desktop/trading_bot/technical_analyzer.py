"""
Technical Analysis Module for Trading Bot
Implements RSI, Bollinger Bands, MACD, and other key indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Advanced technical analysis for trading decisions"""
    
    def __init__(self):
        self.price_cache = {}  # Cache historical prices
        self.indicator_cache = {}  # Cache calculated indicators
        
    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0  # Neutral if insufficient data
            
        prices = np.array(prices)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return float(rsi)
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1] if prices else 0
            return {
                'upper': current_price * 1.02,
                'middle': current_price,
                'lower': current_price * 0.98,
                'position': 0.5  # Neutral position
            }
        
        prices = np.array(prices)
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        current_price = prices[-1]
        
        # Calculate position within bands (0 = lower band, 1 = upper band)
        if upper == lower:
            position = 0.5
        else:
            position = (current_price - lower) / (upper - lower)
            position = max(0, min(1, position))  # Clamp to 0-1
        
        return {
            'upper': float(upper),
            'middle': float(sma),
            'lower': float(lower),
            'position': float(position)
        }
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        
        prices = np.array(prices)
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line (EMA of MACD)
        if len(prices) >= slow + signal:
            macd_values = []
            for i in range(slow - 1, len(prices)):
                ema_f = self._calculate_ema(prices[:i+1], fast)
                ema_s = self._calculate_ema(prices[:i+1], slow)
                macd_values.append(ema_f - ema_s)
            
            signal_line = self._calculate_ema(np.array(macd_values), signal)
        else:
            signal_line = 0.0
        
        histogram = macd_line - signal_line
        
        return {
            'macd': float(macd_line),
            'signal': float(signal_line),
            'histogram': float(histogram)
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        if len(prices) == 1:
            return float(prices[0])
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return float(ema)
    
    def calculate_moving_averages(self, prices: List[float]) -> Dict[str, float]:
        """Calculate various moving averages"""
        if not prices:
            return {'sma_20': 0.0, 'sma_50': 0.0, 'ema_20': 0.0}
        
        prices = np.array(prices)
        
        result = {}
        
        # Simple Moving Averages
        if len(prices) >= 20:
            result['sma_20'] = float(np.mean(prices[-20:]))
        else:
            result['sma_20'] = float(np.mean(prices))
            
        if len(prices) >= 50:
            result['sma_50'] = float(np.mean(prices[-50:]))
        else:
            result['sma_50'] = float(np.mean(prices))
        
        # Exponential Moving Average
        result['ema_20'] = self._calculate_ema(prices, 20)
        
        return result
    
    def analyze_symbol(self, symbol: str, prices: List[float]) -> Dict[str, any]:
        """Comprehensive technical analysis for a symbol"""
        if not prices or len(prices) < 2:
            return self._get_neutral_analysis(symbol)
        
        current_price = prices[-1]
        
        # Calculate all indicators
        rsi = self.calculate_rsi(prices)
        bollinger = self.calculate_bollinger_bands(prices)
        macd = self.calculate_macd(prices)
        mas = self.calculate_moving_averages(prices)
        
        # Trend analysis
        trend = self._analyze_trend(prices, mas)
        
        # Generate trading signals
        signals = self._generate_signals(rsi, bollinger, macd, trend, current_price, mas)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'indicators': {
                'rsi': rsi,
                'bollinger_bands': bollinger,
                'macd': macd,
                'moving_averages': mas
            },
            'trend': trend,
            'signals': signals,
            'overall_score': self._calculate_overall_score(signals),
            'confidence': self._calculate_confidence(signals, len(prices))
        }
    
    def _analyze_trend(self, prices: List[float], mas: Dict[str, float]) -> Dict[str, any]:
        """Analyze price trend"""
        if len(prices) < 10:
            return {'direction': 'neutral', 'strength': 0.5}
        
        current_price = prices[-1]
        sma_20 = mas.get('sma_20', current_price)
        ema_20 = mas.get('ema_20', current_price)
        
        # Price vs moving averages
        price_vs_sma = (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
        price_vs_ema = (current_price - ema_20) / ema_20 if ema_20 > 0 else 0
        
        # Recent price action (last 5 periods)
        recent_prices = prices[-5:]
        price_momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] if recent_prices[0] > 0 else 0
        
        # Determine trend direction
        trend_signals = []
        if price_vs_sma > 0.02:  # 2% above SMA
            trend_signals.append('bullish')
        elif price_vs_sma < -0.02:  # 2% below SMA
            trend_signals.append('bearish')
        
        if price_momentum > 0.01:  # 1% momentum up
            trend_signals.append('bullish')
        elif price_momentum < -0.01:  # 1% momentum down
            trend_signals.append('bearish')
        
        # Calculate overall trend
        bullish_count = trend_signals.count('bullish')
        bearish_count = trend_signals.count('bearish')
        
        if bullish_count > bearish_count:
            direction = 'bullish'
            strength = bullish_count / len(trend_signals) if trend_signals else 0.5
        elif bearish_count > bullish_count:
            direction = 'bearish'
            strength = bearish_count / len(trend_signals) if trend_signals else 0.5
        else:
            direction = 'neutral'
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': strength,
            'price_vs_sma': price_vs_sma,
            'momentum': price_momentum
        }
    
    def _generate_signals(self, rsi: float, bollinger: Dict, macd: Dict, trend: Dict, 
                         current_price: float, mas: Dict) -> Dict[str, any]:
        """Generate buy/sell signals based on technical indicators"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': [],
            'strength': 0.0
        }
        
        # RSI Signals
        if rsi < 30:
            signals['buy_signals'].append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            signals['sell_signals'].append(f"RSI overbought ({rsi:.1f})")
        elif 45 <= rsi <= 55:
            signals['neutral_signals'].append(f"RSI neutral ({rsi:.1f})")
        
        # Bollinger Band Signals
        bb_pos = bollinger['position']
        if bb_pos < 0.2:  # Near lower band
            signals['buy_signals'].append(f"Price near BB lower band ({bb_pos:.2f})")
        elif bb_pos > 0.8:  # Near upper band
            signals['sell_signals'].append(f"Price near BB upper band ({bb_pos:.2f})")
        
        # MACD Signals
        macd_line = macd['macd']
        signal_line = macd['signal']
        histogram = macd['histogram']
        
        if macd_line > signal_line and histogram > 0:
            signals['buy_signals'].append("MACD bullish crossover")
        elif macd_line < signal_line and histogram < 0:
            signals['sell_signals'].append("MACD bearish crossover")
        
        # Trend Signals
        if trend['direction'] == 'bullish' and trend['strength'] > 0.6:
            signals['buy_signals'].append(f"Strong bullish trend ({trend['strength']:.2f})")
        elif trend['direction'] == 'bearish' and trend['strength'] > 0.6:
            signals['sell_signals'].append(f"Strong bearish trend ({trend['strength']:.2f})")
        
        # Moving Average Signals
        if current_price > mas.get('sma_20', 0) > mas.get('sma_50', 0):
            signals['buy_signals'].append("Price above rising MAs")
        elif current_price < mas.get('sma_20', 0) < mas.get('sma_50', 0):
            signals['sell_signals'].append("Price below falling MAs")
        
        # Calculate overall signal strength
        buy_strength = len(signals['buy_signals'])
        sell_strength = len(signals['sell_signals'])
        total_signals = buy_strength + sell_strength + len(signals['neutral_signals'])
        
        if total_signals > 0:
            signals['strength'] = (buy_strength - sell_strength) / total_signals
        
        return signals
    
    def _calculate_overall_score(self, signals: Dict) -> float:
        """Calculate overall technical score (-1 to +1, where +1 is very bullish)"""
        buy_count = len(signals['buy_signals'])
        sell_count = len(signals['sell_signals'])
        total_count = buy_count + sell_count
        
        if total_count == 0:
            return 0.0
        
        return (buy_count - sell_count) / total_count
    
    def _calculate_confidence(self, signals: Dict, data_points: int) -> float:
        """Calculate confidence in the analysis"""
        total_signals = len(signals['buy_signals']) + len(signals['sell_signals'])
        
        # Base confidence on number of signals and data quality
        signal_confidence = min(total_signals / 5.0, 1.0)  # Max confidence at 5+ signals
        data_confidence = min(data_points / 50.0, 1.0)  # Max confidence at 50+ data points
        
        return (signal_confidence + data_confidence) / 2
    
    def _get_neutral_analysis(self, symbol: str) -> Dict[str, any]:
        """Return neutral analysis when insufficient data"""
        return {
            'symbol': symbol,
            'current_price': 0.0,
            'indicators': {
                'rsi': 50.0,
                'bollinger_bands': {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5},
                'macd': {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
                'moving_averages': {'sma_20': 0.0, 'sma_50': 0.0, 'ema_20': 0.0}
            },
            'trend': {'direction': 'neutral', 'strength': 0.5},
            'signals': {
                'buy_signals': [],
                'sell_signals': [],
                'neutral_signals': ['Insufficient data'],
                'strength': 0.0
            },
            'overall_score': 0.0,
            'confidence': 0.1
        }
    
    def get_trading_recommendation(self, analysis: Dict) -> Dict[str, any]:
        """Get specific trading recommendation based on analysis"""
        score = analysis['overall_score']
        confidence = analysis['confidence']
        signals = analysis['signals']
        
        # Determine recommendation
        if score > 0.3 and confidence > 0.6:
            recommendation = 'BUY'
            strength = min(score * confidence, 1.0)
        elif score < -0.3 and confidence > 0.6:
            recommendation = 'SELL'
            strength = min(abs(score) * confidence, 1.0)
        else:
            recommendation = 'HOLD'
            strength = 0.5
        
        # Generate reasoning
        reasoning = []
        if signals['buy_signals']:
            reasoning.extend(signals['buy_signals'])
        if signals['sell_signals']:
            reasoning.extend(signals['sell_signals'])
        if not reasoning:
            reasoning = ['Insufficient clear signals']
        
        return {
            'recommendation': recommendation,
            'strength': strength,
            'confidence': confidence,
            'reasoning': reasoning[:3],  # Top 3 reasons
            'risk_level': 'HIGH' if confidence < 0.4 else 'MEDIUM' if confidence < 0.7 else 'LOW'
        }

# Integration helper for the main trading bot
def enhance_candidate_with_technical_analysis(candidate: Dict, analyzer: TechnicalAnalyzer) -> Dict:
    """Enhance a trading candidate with technical analysis"""
    symbol = candidate.get('symbol', '')
    
    # Generate mock price data for demonstration
    # In real implementation, this would fetch actual price history
    import random
    current_price = candidate.get('price_usd', 1.0)
    
    # Generate 50 periods of realistic price data
    prices = []
    price = current_price * 0.95  # Start slightly lower
    for i in range(50):
        change = random.uniform(-0.03, 0.03)  # ±3% change per period
        price *= (1 + change)
        prices.append(price)
    
    # Add final price as current
    prices[-1] = current_price
    
    # Perform technical analysis
    analysis = analyzer.analyze_symbol(symbol, prices)
    recommendation = analyzer.get_trading_recommendation(analysis)
    
    # Enhance candidate with technical data
    candidate['technical_analysis'] = analysis
    candidate['technical_recommendation'] = recommendation
    candidate['technical_score'] = analysis['overall_score']
    candidate['technical_confidence'] = analysis['confidence']
    
    return candidate