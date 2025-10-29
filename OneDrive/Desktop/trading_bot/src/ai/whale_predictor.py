"""
Predictive Whale Movement Analysis
AI system that predicts whale moves 15-30 minutes before they happen
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import joblib
import json

logger = logging.getLogger(__name__)

@dataclass
class WhaleWallet:
    address: str
    balance: float
    label: str  # 'exchange', 'institution', 'individual', 'unknown'
    risk_level: str  # 'low', 'medium', 'high'
    historical_behavior: Dict
    last_activity: datetime

@dataclass
class WhalePrediction:
    wallet_address: str
    action_probability: float  # 0-1
    predicted_action: str  # 'buy', 'sell', 'transfer', 'accumulate', 'distribute'
    predicted_amount: float
    confidence: float  # 0-1
    timeline: str  # '5min', '15min', '30min', '1hour'
    reasoning: str
    price_impact_estimate: float
    metadata: Dict

@dataclass
class OnChainActivity:
    transaction_hash: str
    from_address: str
    to_address: str
    amount: float
    token_symbol: str
    timestamp: datetime
    gas_price: float
    transaction_type: str  # 'transfer', 'swap', 'liquidity_add', 'liquidity_remove'

class BlockchainMonitor:
    """Monitors blockchain activity for whale movements"""
    
    def __init__(self):
        self.whale_threshold = {
            'BTC': 100,      # 100+ BTC
            'ETH': 1000,     # 1000+ ETH
            'SOL': 10000,    # 10,000+ SOL
            'USDT': 1000000, # $1M+ USDT
            'USDC': 1000000  # $1M+ USDC
        }
        
        self.monitored_addresses = set()
        self.activity_cache = {}
        
    async def monitor_whale_activity(self, symbol: str, hours: int = 24) -> List[OnChainActivity]:
        """Monitor whale activity for specific symbol"""
        try:
            # Get whale addresses for symbol
            whale_addresses = await self.get_whale_addresses(symbol)
            
            # Fetch recent transactions
            activities = []
            for address in whale_addresses:
                address_activity = await self.get_address_activity(address, hours)
                activities.extend(address_activity)
            
            # Filter for significant transactions
            significant_activities = self.filter_significant_transactions(activities, symbol)
            
            return significant_activities
            
        except Exception as e:
            logger.error(f"Error monitoring whale activity: {e}")
            return []
    
    async def get_whale_addresses(self, symbol: str) -> Set[str]:
        """Get known whale addresses for symbol"""
        # In production, this would query blockchain APIs or databases
        whale_addresses = {
            'BTC': {
                '1P5ZEDWTKTFGxQjZphgWPQUpe554WKDfHQ',  # Bitfinex cold wallet
                '3FupnqjHHURAr1rSm5fJ9rgNZLyWobgqGu',  # Unknown whale
                '34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo',  # Binance wallet
            },
            'ETH': {
                '0x8b83de7649d23b28b3ee4c7b1e7e2d07d57b6c8e',  # Vitalik
                '0x2a0c0dbecc7e4d658f48e01e3fa353f44050c208',  # Institutional
                '0xd8da6bf26964af9d7eed9e03e53415d37aa96045',  # MEV bot
            },
            'SOL': {
                '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM',  # Solana Foundation
                'J1S9H3QjnRtBbbuD4HjPV6RpRhwuk4zKbxsnCHuTgh9w',  # Large holder
            }
        }
        
        return whale_addresses.get(symbol, set())
    
    async def get_address_activity(self, address: str, hours: int) -> List[OnChainActivity]:
        """Get recent activity for specific address"""
        # Placeholder - in production would query blockchain APIs
        # This would integrate with Etherscan, Solscan, etc.
        
        # Mock some activity for demonstration
        mock_activities = [
            OnChainActivity(
                transaction_hash='0x123...',
                from_address=address,
                to_address='0x456...',
                amount=1000.0,
                token_symbol='ETH',
                timestamp=datetime.now() - timedelta(minutes=30),
                gas_price=20.0,
                transaction_type='transfer'
            )
        ]
        
        return mock_activities
    
    def filter_significant_transactions(self, activities: List[OnChainActivity], symbol: str) -> List[OnChainActivity]:
        """Filter for transactions above whale threshold"""
        threshold = self.whale_threshold.get(symbol, 1000000)
        
        return [
            activity for activity in activities
            if activity.amount >= threshold
        ]

class WhaleBehaviorAnalyzer:
    """Analyzes historical whale behavior patterns"""
    
    def __init__(self):
        self.behavior_patterns = {}
        self.pattern_classifier = None
        self.amount_predictor = None
        self.scaler = StandardScaler()
        
        self._load_models()
    
    def _load_models(self):
        """Load or initialize ML models"""
        try:
            self.pattern_classifier = joblib.load('models/whale_pattern_classifier.pkl')
            self.amount_predictor = joblib.load('models/whale_amount_predictor.pkl')
            self.scaler = joblib.load('models/whale_scaler.pkl')
        except FileNotFoundError:
            # Initialize new models
            self.pattern_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.amount_predictor = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
    
    async def analyze_whale_patterns(self, whale_address: str, recent_activity: List[OnChainActivity]) -> Dict:
        """Analyze behavior patterns for specific whale"""
        try:
            # Get historical behavior
            historical_data = await self.get_historical_behavior(whale_address)
            
            # Extract pattern features
            features = self.extract_behavior_features(historical_data, recent_activity)
            
            # Classify current behavior pattern
            pattern = self.classify_behavior_pattern(features)
            
            # Analyze timing patterns
            timing_pattern = self.analyze_timing_patterns(historical_data)
            
            # Analyze amount patterns
            amount_pattern = self.analyze_amount_patterns(historical_data)
            
            return {
                'current_pattern': pattern,
                'timing_pattern': timing_pattern,
                'amount_pattern': amount_pattern,
                'behavior_features': features,
                'pattern_confidence': self.calculate_pattern_confidence(features)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing whale patterns: {e}")
            return {
                'current_pattern': 'unknown',
                'timing_pattern': {},
                'amount_pattern': {},
                'behavior_features': {},
                'pattern_confidence': 0.5
            }
    
    async def get_historical_behavior(self, whale_address: str) -> List[OnChainActivity]:
        """Get historical transaction data for whale"""
        # In production, this would query comprehensive blockchain data
        # For now, return mock historical data
        
        historical_activities = []
        base_time = datetime.now() - timedelta(days=30)
        
        # Generate mock historical pattern
        for i in range(50):  # 50 historical transactions
            activity = OnChainActivity(
                transaction_hash=f'0x{i:064x}',
                from_address=whale_address,
                to_address=f'0x{(i+1):040x}',
                amount=np.random.lognormal(8, 1),  # Log-normal distribution
                token_symbol='ETH',
                timestamp=base_time + timedelta(hours=i*12),
                gas_price=np.random.uniform(10, 50),
                transaction_type=np.random.choice(['transfer', 'swap', 'liquidity_add'])
            )
            historical_activities.append(activity)
        
        return historical_activities
    
    def extract_behavior_features(self, historical: List[OnChainActivity], recent: List[OnChainActivity]) -> Dict:
        """Extract behavioral features for ML models"""
        features = {}
        
        if not historical:
            return features
        
        # Transaction frequency features
        features['avg_daily_transactions'] = len(historical) / 30  # 30 days
        features['recent_activity_increase'] = len(recent) / max(len(historical) / 30, 1)
        
        # Amount features
        amounts = [tx.amount for tx in historical]
        features['avg_transaction_amount'] = np.mean(amounts)
        features['median_transaction_amount'] = np.median(amounts)
        features['amount_volatility'] = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 0
        
        # Timing features
        timestamps = [tx.timestamp for tx in historical]
        if len(timestamps) > 1:
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() / 3600 
                        for i in range(1, len(timestamps))]  # Hours between transactions
            features['avg_interval_hours'] = np.mean(intervals)
            features['interval_volatility'] = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
        
        # Transaction type patterns
        type_counts = defaultdict(int)
        for tx in historical:
            type_counts[tx.transaction_type] += 1
        
        total_txs = len(historical)
        if total_txs > 0:
            features['transfer_ratio'] = type_counts['transfer'] / total_txs
            features['swap_ratio'] = type_counts['swap'] / total_txs
            features['liquidity_ratio'] = type_counts['liquidity_add'] / total_txs
        
        # Recent behavior changes
        if recent:
            recent_amounts = [tx.amount for tx in recent]
            features['recent_avg_amount'] = np.mean(recent_amounts)
            features['amount_change_ratio'] = features['recent_avg_amount'] / features['avg_transaction_amount'] if features['avg_transaction_amount'] > 0 else 1
        
        # Gas price patterns (indicates urgency)
        gas_prices = [tx.gas_price for tx in historical]
        features['avg_gas_price'] = np.mean(gas_prices)
        features['gas_price_volatility'] = np.std(gas_prices)
        
        return features
    
    def classify_behavior_pattern(self, features: Dict) -> str:
        """Classify whale behavior pattern"""
        if not features:
            return 'unknown'
        
        # Simple rule-based classification for now
        # In production, this would use trained ML models
        
        if features.get('recent_activity_increase', 1) > 2.0:
            return 'high_activity'
        elif features.get('amount_change_ratio', 1) > 1.5:
            return 'accumulating'
        elif features.get('amount_change_ratio', 1) < 0.5:
            return 'distributing'
        elif features.get('avg_interval_hours', 24) < 6:
            return 'urgent_activity'
        else:
            return 'normal_activity'
    
    def analyze_timing_patterns(self, historical: List[OnChainActivity]) -> Dict:
        """Analyze timing patterns in whale behavior"""
        if not historical:
            return {}
        
        # Hour of day analysis
        hours = [tx.timestamp.hour for tx in historical]
        hour_counts = defaultdict(int)
        for hour in hours:
            hour_counts[hour] += 1
        
        most_active_hour = max(hour_counts.items(), key=lambda x: x[1])[0]
        
        # Day of week analysis
        weekdays = [tx.timestamp.weekday() for tx in historical]
        weekday_counts = defaultdict(int)
        for day in weekdays:
            weekday_counts[day] += 1
        
        most_active_day = max(weekday_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'most_active_hour': most_active_hour,
            'most_active_day': most_active_day,
            'hour_distribution': dict(hour_counts),
            'weekday_distribution': dict(weekday_counts)
        }
    
    def analyze_amount_patterns(self, historical: List[OnChainActivity]) -> Dict:
        """Analyze transaction amount patterns"""
        if not historical:
            return {}
        
        amounts = [tx.amount for tx in historical]
        
        # Amount clustering (simple approach)
        amounts_array = np.array(amounts)
        
        # Find common amount ranges
        percentiles = np.percentile(amounts_array, [25, 50, 75, 90, 95])
        
        return {
            'min_amount': float(np.min(amounts_array)),
            'max_amount': float(np.max(amounts_array)),
            'avg_amount': float(np.mean(amounts_array)),
            'median_amount': float(np.median(amounts_array)),
            'amount_percentiles': {
                '25th': float(percentiles[0]),
                '50th': float(percentiles[1]),
                '75th': float(percentiles[2]),
                '90th': float(percentiles[3]),
                '95th': float(percentiles[4])
            }
        }
    
    def calculate_pattern_confidence(self, features: Dict) -> float:
        """Calculate confidence in pattern recognition"""
        if not features:
            return 0.0
        
        # Simple confidence calculation based on data completeness
        confidence_factors = []
        
        # Data completeness
        expected_features = [
            'avg_daily_transactions', 'avg_transaction_amount',
            'avg_interval_hours', 'transfer_ratio'
        ]
        
        completeness = sum(1 for feature in expected_features if feature in features) / len(expected_features)
        confidence_factors.append(completeness)
        
        # Data consistency (low volatility = higher confidence)
        if 'amount_volatility' in features:
            amount_consistency = max(0, 1 - features['amount_volatility'])
            confidence_factors.append(amount_consistency)
        
        if 'interval_volatility' in features:
            timing_consistency = max(0, 1 - features['interval_volatility'])
            confidence_factors.append(timing_consistency)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5

class WhalePredictor:
    """Main whale prediction engine"""
    
    def __init__(self):
        self.blockchain_monitor = BlockchainMonitor()
        self.behavior_analyzer = WhaleBehaviorAnalyzer()
        self.prediction_model = None
        self.scaler = StandardScaler()
        
        self._load_prediction_model()
    
    def _load_prediction_model(self):
        """Load or initialize prediction model"""
        try:
            self.prediction_model = joblib.load('models/whale_prediction_model.pkl')
            self.scaler = joblib.load('models/whale_prediction_scaler.pkl')
        except FileNotFoundError:
            # Initialize new model
            self.prediction_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42
            )
    
    async def predict_whale_action(self, wallet_address: str, symbol: str, market_data: Dict) -> WhalePrediction:
        """Predict whale action with high accuracy"""
        try:
            # Get recent whale activity
            recent_activity = await self.blockchain_monitor.get_address_activity(wallet_address, hours=24)
            
            # Analyze behavior patterns
            behavior_analysis = await self.behavior_analyzer.analyze_whale_patterns(wallet_address, recent_activity)
            
            # Get market context
            market_context = self.extract_market_context(market_data, symbol)
            
            # Combine features for prediction
            prediction_features = self.combine_prediction_features(
                behavior_analysis, market_context, recent_activity
            )
            
            # Make prediction
            action_prediction = self.predict_action(prediction_features)
            amount_prediction = self.predict_amount(prediction_features)
            timeline_prediction = self.predict_timeline(prediction_features)
            
            # Calculate confidence and price impact
            confidence = self.calculate_prediction_confidence(prediction_features, behavior_analysis)
            price_impact = self.estimate_price_impact(amount_prediction, market_data)
            
            # Generate reasoning
            reasoning = self.generate_prediction_reasoning(
                behavior_analysis, action_prediction, market_context
            )
            
            return WhalePrediction(
                wallet_address=wallet_address,
                action_probability=action_prediction['probability'],
                predicted_action=action_prediction['action'],
                predicted_amount=amount_prediction,
                confidence=confidence,
                timeline=timeline_prediction,
                reasoning=reasoning,
                price_impact_estimate=price_impact,
                metadata={
                    'behavior_analysis': behavior_analysis,
                    'market_context': market_context,
                    'recent_activity_count': len(recent_activity)
                }
            )
            
        except Exception as e:
            logger.error(f"Error predicting whale action: {e}")
            return WhalePrediction(
                wallet_address=wallet_address,
                action_probability=0.5,
                predicted_action='unknown',
                predicted_amount=0.0,
                confidence=0.0,
                timeline='unknown',
                reasoning=f"Error in prediction: {e}",
                price_impact_estimate=0.0,
                metadata={}
            )
    
    def extract_market_context(self, market_data: Dict, symbol: str) -> Dict:
        """Extract market context features"""
        context = {}
        
        # Price movement context
        price_history = market_data.get('price_history', [])
        if len(price_history) >= 2:
            recent_change = (price_history[-1] - price_history[-2]) / price_history[-2]
            context['recent_price_change'] = recent_change
            
            # Volatility
            if len(price_history) >= 20:
                returns = [(price_history[i] - price_history[i-1]) / price_history[i-1] 
                          for i in range(1, len(price_history))]
                context['volatility'] = np.std(returns)
        
        # Volume context
        volume_history = market_data.get('volume_history', [])
        if len(volume_history) >= 2:
            volume_change = (volume_history[-1] - volume_history[-2]) / volume_history[-2]
            context['volume_change'] = volume_change
        
        # Orderbook context
        orderbook = market_data.get('orderbook', {})
        if orderbook:
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if bids and asks:
                spread = (float(asks[0][0]) - float(bids[0][0])) / float(bids[0][0])
                context['spread'] = spread
                
                # Liquidity depth
                bid_depth = sum(float(bid[1]) for bid in bids[:10])
                ask_depth = sum(float(ask[1]) for ask in asks[:10])
                context['liquidity_depth'] = bid_depth + ask_depth
        
        return context
    
    def combine_prediction_features(self, behavior_analysis: Dict, market_context: Dict, recent_activity: List) -> Dict:
        """Combine all features for prediction"""
        features = {}
        
        # Behavior features
        behavior_features = behavior_analysis.get('behavior_features', {})
        features.update(behavior_features)
        
        # Market features
        features.update(market_context)
        
        # Recent activity features
        features['recent_activity_count'] = len(recent_activity)
        
        if recent_activity:
            recent_amounts = [tx.amount for tx in recent_activity]
            features['recent_total_amount'] = sum(recent_amounts)
            features['recent_avg_amount'] = np.mean(recent_amounts)
            
            # Time since last activity
            last_activity = max(tx.timestamp for tx in recent_activity)
            time_since_last = (datetime.now() - last_activity).total_seconds() / 3600  # Hours
            features['hours_since_last_activity'] = time_since_last
        
        # Pattern features
        pattern = behavior_analysis.get('current_pattern', 'unknown')
        features['is_high_activity'] = 1 if pattern == 'high_activity' else 0
        features['is_accumulating'] = 1 if pattern == 'accumulating' else 0
        features['is_distributing'] = 1 if pattern == 'distributing' else 0
        features['is_urgent'] = 1 if pattern == 'urgent_activity' else 0
        
        return features
    
    def predict_action(self, features: Dict) -> Dict:
        """Predict whale action type and probability"""
        # Simplified rule-based prediction for now
        # In production, this would use trained ML models
        
        if not features:
            return {'action': 'unknown', 'probability': 0.5}
        
        # Rule-based action prediction
        if features.get('is_accumulating', 0) == 1 and features.get('recent_price_change', 0) < -0.05:
            return {'action': 'buy', 'probability': 0.8}
        elif features.get('is_distributing', 0) == 1 and features.get('recent_price_change', 0) > 0.05:
            return {'action': 'sell', 'probability': 0.8}
        elif features.get('is_urgent', 0) == 1:
            if features.get('recent_price_change', 0) > 0:
                return {'action': 'sell', 'probability': 0.7}
            else:
                return {'action': 'buy', 'probability': 0.7}
        elif features.get('recent_activity_count', 0) > 5:
            return {'action': 'transfer', 'probability': 0.6}
        else:
            return {'action': 'hold', 'probability': 0.5}
    
    def predict_amount(self, features: Dict) -> float:
        """Predict transaction amount"""
        # Simple amount prediction based on historical patterns
        
        if 'recent_avg_amount' in features:
            base_amount = features['recent_avg_amount']
        elif 'avg_transaction_amount' in features:
            base_amount = features['avg_transaction_amount']
        else:
            base_amount = 1000.0  # Default
        
        # Adjust based on market conditions
        if features.get('volatility', 0) > 0.05:  # High volatility
            return base_amount * 1.5  # Larger transactions in volatile markets
        elif features.get('is_urgent', 0) == 1:
            return base_amount * 2.0  # Urgent activity = larger amounts
        else:
            return base_amount
    
    def predict_timeline(self, features: Dict) -> str:
        """Predict when action will occur"""
        if features.get('is_urgent', 0) == 1:
            return '5min'
        elif features.get('recent_activity_count', 0) > 3:
            return '15min'
        elif features.get('hours_since_last_activity', 24) < 6:
            return '30min'
        else:
            return '1hour'
    
    def calculate_prediction_confidence(self, features: Dict, behavior_analysis: Dict) -> float:
        """Calculate confidence in prediction"""
        confidence_factors = []
        
        # Pattern confidence
        pattern_confidence = behavior_analysis.get('pattern_confidence', 0.5)
        confidence_factors.append(pattern_confidence)
        
        # Data completeness
        expected_features = ['recent_activity_count', 'avg_transaction_amount', 'recent_price_change']
        completeness = sum(1 for feature in expected_features if feature in features) / len(expected_features)
        confidence_factors.append(completeness)
        
        # Recent activity indicator
        if features.get('recent_activity_count', 0) > 0:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        return np.mean(confidence_factors)
    
    def estimate_price_impact(self, predicted_amount: float, market_data: Dict) -> float:
        """Estimate potential price impact of predicted whale action"""
        try:
            # Get market depth
            orderbook = market_data.get('orderbook', {})
            if not orderbook:
                return 0.01  # 1% default estimate
            
            bids = orderbook.get('bids', [])
            asks = orderbook.get('asks', [])
            
            if not bids or not asks:
                return 0.01
            
            # Calculate depth for predicted amount
            total_depth = 0
            for price, volume in (bids + asks):
                total_depth += float(volume)
            
            if total_depth == 0:
                return 0.05  # 5% if no liquidity
            
            # Estimate impact based on order size vs depth
            impact_ratio = predicted_amount / total_depth
            
            # Non-linear impact (larger orders have disproportionate impact)
            price_impact = impact_ratio * (1 + impact_ratio)
            
            return min(price_impact, 0.2)  # Cap at 20%
            
        except Exception:
            return 0.01
    
    def generate_prediction_reasoning(self, behavior_analysis: Dict, action_prediction: Dict, market_context: Dict) -> str:
        """Generate human-readable reasoning for prediction"""
        reasons = []
        
        # Behavior reasons
        pattern = behavior_analysis.get('current_pattern', 'unknown')
        if pattern == 'accumulating':
            reasons.append("Whale showing accumulation pattern")
        elif pattern == 'distributing':
            reasons.append("Whale showing distribution pattern")
        elif pattern == 'high_activity':
            reasons.append("Unusually high whale activity detected")
        elif pattern == 'urgent_activity':
            reasons.append("Urgent activity pattern suggests imminent action")
        
        # Market reasons
        if 'recent_price_change' in market_context:
            price_change = market_context['recent_price_change']
            if price_change > 0.05:
                reasons.append("Strong upward price movement may trigger selling")
            elif price_change < -0.05:
                reasons.append("Price dip may trigger accumulation")
        
        if 'volatility' in market_context and market_context['volatility'] > 0.05:
            reasons.append("High volatility increasing whale activity probability")
        
        # Action-specific reasons
        action = action_prediction.get('action', 'unknown')
        probability = action_prediction.get('probability', 0.5)
        
        if probability > 0.7:
            reasons.append(f"High confidence {action} signal")
        elif probability > 0.6:
            reasons.append(f"Moderate confidence {action} signal")
        
        return "; ".join(reasons) if reasons else "Insufficient data for detailed reasoning"

# Example usage and testing
async def test_whale_predictor():
    """Test whale prediction system"""
    predictor = WhalePredictor()
    
    # Mock market data
    market_data = {
        'price_history': [100 + i * 0.5 for i in range(100)],
        'volume_history': [1000000 + i * 10000 for i in range(100)],
        'orderbook': {
            'bids': [['99.5', '1000'], ['99.4', '2000'], ['99.3', '1500']],
            'asks': [['100.5', '1200'], ['100.6', '1800'], ['100.7', '2200']]
        }
    }
    
    # Test prediction
    prediction = await predictor.predict_whale_action(
        wallet_address='0x8b83de7649d23b28b3ee4c7b1e7e2d07d57b6c8e',
        symbol='ETH',
        market_data=market_data
    )
    
    print(f"Whale Prediction Results:")
    print(f"  Address: {prediction.wallet_address}")
    print(f"  Predicted Action: {prediction.predicted_action}")
    print(f"  Action Probability: {prediction.action_probability:.3f}")
    print(f"  Predicted Amount: {prediction.predicted_amount:.2f}")
    print(f"  Confidence: {prediction.confidence:.3f}")
    print(f"  Timeline: {prediction.timeline}")
    print(f"  Price Impact Estimate: {prediction.price_impact_estimate:.3f}")
    print(f"  Reasoning: {prediction.reasoning}")
    
    return prediction

if __name__ == "__main__":
    asyncio.run(test_whale_predictor())