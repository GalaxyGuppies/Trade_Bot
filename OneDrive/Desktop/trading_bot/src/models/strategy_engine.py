"""
Strategy engine that combines market data, sentiment, and on-chain analysis
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class StrategyEngine:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.signals = {}
        self.running = False
        
        # Strategy parameters
        self.config = {
            'sentiment_weight': 0.3,
            'technical_weight': 0.4,
            'onchain_weight': 0.3,
            'min_confidence': 0.6,
            'rugpull_threshold': 0.3,
            'position_size_base': 0.02,  # 2% of portfolio per trade
            'max_positions': 5,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15
        }
        
        # Risk management
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.daily_loss_limit = -0.1  # -10% daily limit
        
        # Initialize models
        self.init_models()
    
    def init_models(self):
        """Initialize ML models for different components"""
        try:
            # Simple Random Forest for demonstration
            # In production, you'd load pre-trained models
            self.models['price_direction'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.scalers['features'] = StandardScaler()
            
            logger.info("Strategy models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def start(self):
        """Start the strategy engine"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting strategy engine...")
        
        # Start signal generation loop
        asyncio.create_task(self.generate_signals_loop())
        
        # Start risk monitoring
        asyncio.create_task(self.monitor_risk())
    
    async def stop(self):
        """Stop the strategy engine"""
        self.running = False
        logger.info("Stopping strategy engine...")
    
    async def generate_signals_loop(self):
        """Main loop for generating trading signals"""
        while self.running:
            try:
                # Get available symbols
                symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                
                for symbol in symbols:
                    signal = await self.generate_signal(symbol)
                    if signal:
                        self.signals[symbol] = signal
                        logger.info(f"Generated signal for {symbol}: {signal['action']} (confidence: {signal['confidence']:.2f})")
                
                await asyncio.sleep(30)  # Generate signals every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(30)
    
    async def generate_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Generate trading signal for a symbol"""
        try:
            # Collect all necessary data
            market_data = await self.get_market_features(symbol)
            sentiment_data = await self.get_sentiment_features(symbol)
            onchain_data = await self.get_onchain_features(symbol)
            
            if not all([market_data, sentiment_data]):
                return None
            
            # Check rugpull risk first
            rugpull_score = await self.assess_rugpull_risk(symbol)
            if rugpull_score > self.config['rugpull_threshold']:
                logger.warning(f"High rugpull risk for {symbol}: {rugpull_score:.2f}")
                return None
            
            # Combine features
            features = self.combine_features(market_data, sentiment_data, onchain_data)
            
            # Generate prediction
            prediction = self.predict_direction(features)
            
            # Calculate confidence
            confidence = self.calculate_confidence(features, prediction)
            
            if confidence < self.config['min_confidence']:
                return None
            
            # Determine action and position size
            action, size = self.determine_action(symbol, prediction, confidence, features)
            
            if action == 'hold':
                return None
            
            # Calculate stop loss and take profit
            stop_loss, take_profit = self.calculate_stop_take_profit(
                symbol, action, market_data['current_price']
            )
            
            signal = {
                'symbol': symbol,
                'action': action,
                'size': size,
                'confidence': confidence,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reasoning': self.generate_reasoning(features, prediction),
                'timestamp': datetime.now(),
                'rugpull_score': rugpull_score
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def get_market_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get market-based features"""
        # This would connect to your market data collector
        # For now, returning mock data
        return {
            'current_price': 45000.0,
            'rsi': 65.0,
            'sma_20': 44500.0,
            'bb_upper': 46000.0,
            'bb_lower': 43000.0,
            'volume_ratio': 1.2,
            'price_change_24h': 0.02,
            'volatility': 0.05
        }
    
    async def get_sentiment_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment-based features"""
        # This would connect to your sentiment collector
        # For now, returning mock data
        return {
            'sentiment_score': 0.65,
            'sentiment_trend': 0.1,
            'volume_sentiment': 0.7,
            'news_sentiment': 0.6,
            'social_volume': 1000
        }
    
    async def get_onchain_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get on-chain features"""
        # This would connect to your on-chain analyzer
        # For now, returning mock data
        return {
            'whale_activity': 0.3,
            'exchange_flows': -0.1,  # Negative means outflow
            'active_addresses': 1.1,
            'transaction_volume': 0.8
        }
    
    def combine_features(self, market_data: Dict, sentiment_data: Dict, onchain_data: Dict) -> np.ndarray:
        """Combine all features into a single feature vector"""
        features = []
        
        # Market features
        features.extend([
            market_data['rsi'] / 100.0,
            (market_data['current_price'] - market_data['sma_20']) / market_data['sma_20'],
            market_data['volume_ratio'],
            market_data['price_change_24h'],
            market_data['volatility']
        ])
        
        # Sentiment features
        features.extend([
            sentiment_data['sentiment_score'],
            sentiment_data['sentiment_trend'],
            sentiment_data['volume_sentiment'],
            sentiment_data['news_sentiment']
        ])
        
        # On-chain features (if available)
        if onchain_data:
            features.extend([
                onchain_data['whale_activity'],
                onchain_data['exchange_flows'],
                onchain_data['active_addresses'],
                onchain_data['transaction_volume']
            ])
        
        return np.array(features).reshape(1, -1)
    
    def predict_direction(self, features: np.ndarray) -> Dict[str, float]:
        """Predict price direction using ML model"""
        try:
            # For now, using simple heuristics
            # In production, you'd use trained ML models
            
            rsi = features[0][0] * 100
            price_vs_sma = features[0][1]
            sentiment = features[0][5]
            
            # Simple rule-based prediction
            buy_probability = 0.5
            
            # RSI signals
            if rsi < 30:
                buy_probability += 0.2
            elif rsi > 70:
                buy_probability -= 0.2
            
            # Price vs SMA
            if price_vs_sma > 0.02:
                buy_probability -= 0.1
            elif price_vs_sma < -0.02:
                buy_probability += 0.1
            
            # Sentiment
            buy_probability += (sentiment - 0.5) * 0.3
            
            buy_probability = max(0, min(1, buy_probability))
            
            return {
                'buy_probability': buy_probability,
                'sell_probability': 1 - buy_probability
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {'buy_probability': 0.5, 'sell_probability': 0.5}
    
    def calculate_confidence(self, features: np.ndarray, prediction: Dict[str, float]) -> float:
        """Calculate confidence in the prediction"""
        # Distance from neutral (0.5)
        max_prob = max(prediction['buy_probability'], prediction['sell_probability'])
        confidence = abs(max_prob - 0.5) * 2
        
        # Adjust based on feature quality
        sentiment_strength = abs(features[0][5] - 0.5) * 2
        volume_strength = min(features[0][2], 2.0) / 2.0
        
        # Combine confidences
        final_confidence = (confidence * 0.6 + sentiment_strength * 0.2 + volume_strength * 0.2)
        
        return min(1.0, final_confidence)
    
    def determine_action(self, symbol: str, prediction: Dict[str, float], confidence: float, features: np.ndarray) -> Tuple[str, float]:
        """Determine trading action and position size"""
        # Check if we already have a position
        if symbol in self.active_positions:
            return 'hold', 0.0
        
        # Check daily loss limit
        if self.daily_pnl <= self.daily_loss_limit:
            logger.warning("Daily loss limit reached, no new positions")
            return 'hold', 0.0
        
        # Check maximum positions
        if len(self.active_positions) >= self.config['max_positions']:
            return 'hold', 0.0
        
        # Determine action
        if prediction['buy_probability'] > 0.6:
            action = 'buy'
        elif prediction['sell_probability'] > 0.6:
            action = 'sell'
        else:
            action = 'hold'
        
        if action == 'hold':
            return action, 0.0
        
        # Calculate position size based on confidence and volatility
        base_size = self.config['position_size_base']
        volatility_adj = min(2.0, 1.0 / features[0][4])  # Reduce size for high volatility
        confidence_adj = confidence
        
        size = base_size * confidence_adj * volatility_adj
        size = max(0.001, min(0.05, size))  # Clamp between 0.1% and 5%
        
        return action, size
    
    def calculate_stop_take_profit(self, symbol: str, action: str, current_price: float) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        if action == 'buy':
            stop_loss = current_price * (1 - self.config['stop_loss_pct'])
            take_profit = current_price * (1 + self.config['take_profit_pct'])
        else:  # sell
            stop_loss = current_price * (1 + self.config['stop_loss_pct'])
            take_profit = current_price * (1 - self.config['take_profit_pct'])
        
        return stop_loss, take_profit
    
    def generate_reasoning(self, features: np.ndarray, prediction: Dict[str, float]) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning = []
        
        rsi = features[0][0] * 100
        price_vs_sma = features[0][1]
        sentiment = features[0][5]
        
        if rsi < 30:
            reasoning.append("RSI indicates oversold condition")
        elif rsi > 70:
            reasoning.append("RSI indicates overbought condition")
        
        if price_vs_sma > 0.02:
            reasoning.append("Price significantly above SMA20")
        elif price_vs_sma < -0.02:
            reasoning.append("Price significantly below SMA20")
        
        if sentiment > 0.7:
            reasoning.append("Strong positive sentiment")
        elif sentiment < 0.3:
            reasoning.append("Strong negative sentiment")
        
        if prediction['buy_probability'] > 0.6:
            reasoning.append("Model predicts upward movement")
        elif prediction['sell_probability'] > 0.6:
            reasoning.append("Model predicts downward movement")
        
        return "; ".join(reasoning)
    
    async def assess_rugpull_risk(self, symbol: str) -> float:
        """Assess rugpull risk for a token"""
        # This would connect to your on-chain analyzer
        # For now, returning a low risk score
        return 0.1
    
    async def monitor_risk(self):
        """Monitor risk across all positions"""
        while self.running:
            try:
                # Update daily P&L
                await self.update_daily_pnl()
                
                # Check stop losses
                await self.check_stop_losses()
                
                # Check take profits
                await self.check_take_profits()
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(5)
    
    async def update_daily_pnl(self):
        """Update daily P&L calculation"""
        # This would calculate actual P&L from positions
        pass
    
    async def check_stop_losses(self):
        """Check if any positions need to be stopped out"""
        # This would check current prices against stop losses
        pass
    
    async def check_take_profits(self):
        """Check if any positions should take profit"""
        # This would check current prices against take profit levels
        pass
    
    async def get_current_signals(self) -> Dict[str, Any]:
        """Get current trading signals"""
        return self.signals.copy()
    
    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get strategy engine status"""
        return {
            'running': self.running,
            'active_positions': len(self.active_positions),
            'daily_pnl': self.daily_pnl,
            'signals_count': len(self.signals),
            'config': self.config,
            'last_update': datetime.now().isoformat()
        }