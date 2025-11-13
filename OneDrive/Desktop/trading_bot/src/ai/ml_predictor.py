"""
ML Performance Prediction Model
Advanced machine learning system for predicting token performance based on 
multiple data sources including sentiment, whale activity, contract analysis, and market patterns.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import sqlite3
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.feature_selection import SelectKBest, f_regression
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available. Install scikit-learn for ML predictions.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionTimeframe(Enum):
    HOURS_1 = "1h"
    HOURS_4 = "4h"
    HOURS_24 = "24h"
    DAYS_7 = "7d"

class ConfidenceLevel(Enum):
    VERY_LOW = "very_low"    # < 30%
    LOW = "low"              # 30-50%
    MEDIUM = "medium"        # 50-70%
    HIGH = "high"            # 70-85%
    VERY_HIGH = "very_high"  # > 85%

@dataclass
class FeatureSet:
    """Complete feature set for ML prediction"""
    # Price and volume features
    current_price: float
    volume_24h: float
    volume_7d_avg: float
    price_change_1h: float
    price_change_24h: float
    price_change_7d: float
    volatility_24h: float
    
    # Market cap and liquidity
    market_cap: float
    liquidity_usd: float
    holders_count: int
    
    # Sentiment features
    sentiment_score: float
    social_mentions: int
    twitter_sentiment: float
    reddit_sentiment: float
    news_sentiment: float
    
    # Contract security features
    security_score: float
    honeypot_risk: float
    ownership_renounced: bool
    liquidity_locked: bool
    verified_contract: bool
    
    # Whale activity features
    whale_activity_score: float
    large_transactions_24h: int
    whale_accumulation: float
    insider_activity: float
    smart_money_flow: float
    
    # Technical indicators
    rsi_14: float
    macd_signal: float
    bollinger_position: float
    support_distance: float
    resistance_distance: float
    
    # Network and ecosystem
    transaction_count_24h: int
    unique_traders_24h: int
    dex_presence_count: int
    
    # Time-based features
    hour_of_day: int
    day_of_week: int
    is_weekend: bool
    
    # Market context
    btc_price_change: float
    eth_price_change: float
    market_fear_greed: float

@dataclass
class PredictionResult:
    """ML prediction result with confidence metrics"""
    token_address: str
    symbol: str
    timeframe: PredictionTimeframe
    predicted_return: float
    confidence_level: ConfidenceLevel
    confidence_score: float
    feature_importance: Dict[str, float]
    model_used: str
    prediction_timestamp: datetime
    
    # Risk assessment
    upside_potential: float
    downside_risk: float
    volatility_prediction: float
    
    # Supporting signals
    bullish_signals: List[str]
    bearish_signals: List[str]
    key_factors: List[str]

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    timeframe: PredictionTimeframe
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    r2: float   # R-squared
    accuracy_bands: Dict[str, float]  # Accuracy within different % bands
    last_updated: datetime

class MLPredictor:
    """
    Advanced ML-based performance prediction system
    Uses multiple algorithms and feature engineering for accurate predictions
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_columns = []
        self.model_performance = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42
                }
            },
            'gradient_boost': {
                'model': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 6,
                    'random_state': 42
                }
            },
            'ridge': {
                'model': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        }
        
        # Training parameters
        self.min_training_samples = 1000
        self.feature_selection_k = 30  # Top K features to select
        self.retrain_interval_hours = 6
        self.last_training_time = {}
        
        self._init_database()
        logger.info("🧠 ML Predictor initialized")
    
    def _init_database(self):
        """Initialize ML prediction database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Features table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    symbol TEXT,
                    features TEXT NOT NULL,
                    price_after_1h REAL,
                    price_after_4h REAL,
                    price_after_24h REAL,
                    price_after_7d REAL,
                    timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_address TEXT NOT NULL,
                    symbol TEXT,
                    timeframe TEXT,
                    predicted_return REAL,
                    confidence_level TEXT,
                    confidence_score REAL,
                    model_used TEXT,
                    actual_return REAL,
                    prediction_error REAL,
                    prediction_data TEXT,
                    prediction_timestamp DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Model performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    timeframe TEXT,
                    mae REAL,
                    mse REAL,
                    rmse REAL,
                    r2_score REAL,
                    accuracy_bands TEXT,
                    training_samples INTEGER,
                    last_updated DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("ML database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing ML database: {e}")
    
    async def collect_features(self, token_address: str, symbol: str = None) -> FeatureSet:
        """
        Collect comprehensive feature set for ML prediction
        Integrates data from multiple sources
        """
        try:
            logger.info(f"🔬 Collecting features for {symbol or token_address}")
            
            # Get current timestamp for time-based features
            now = datetime.now()
            
            # Collect price and market data (placeholder - would integrate with real APIs)
            price_data = await self._get_price_data(token_address)
            
            # Collect sentiment data
            sentiment_data = await self._get_sentiment_data(token_address)
            
            # Collect contract security data
            security_data = await self._get_security_data(token_address)
            
            # Collect whale activity data
            whale_data = await self._get_whale_data(token_address)
            
            # Collect technical indicators
            technical_data = await self._get_technical_data(token_address)
            
            # Collect market context
            market_context = await self._get_market_context()
            
            # Construct feature set
            features = FeatureSet(
                # Price and volume features
                current_price=price_data.get('current_price', 0.0),
                volume_24h=price_data.get('volume_24h', 0.0),
                volume_7d_avg=price_data.get('volume_7d_avg', 0.0),
                price_change_1h=price_data.get('price_change_1h', 0.0),
                price_change_24h=price_data.get('price_change_24h', 0.0),
                price_change_7d=price_data.get('price_change_7d', 0.0),
                volatility_24h=price_data.get('volatility_24h', 0.0),
                
                # Market cap and liquidity
                market_cap=price_data.get('market_cap', 0.0),
                liquidity_usd=price_data.get('liquidity_usd', 0.0),
                holders_count=price_data.get('holders_count', 0),
                
                # Sentiment features
                sentiment_score=sentiment_data.get('overall_score', 0.0),
                social_mentions=sentiment_data.get('mentions_24h', 0),
                twitter_sentiment=sentiment_data.get('twitter_sentiment', 0.0),
                reddit_sentiment=sentiment_data.get('reddit_sentiment', 0.0),
                news_sentiment=sentiment_data.get('news_sentiment', 0.0),
                
                # Contract security features
                security_score=security_data.get('security_score', 50.0),
                honeypot_risk=security_data.get('honeypot_risk', 0.0),
                ownership_renounced=security_data.get('ownership_renounced', False),
                liquidity_locked=security_data.get('liquidity_locked', False),
                verified_contract=security_data.get('verified_contract', False),
                
                # Whale activity features
                whale_activity_score=whale_data.get('activity_score', 0.0),
                large_transactions_24h=whale_data.get('large_transactions_24h', 0),
                whale_accumulation=whale_data.get('accumulation_score', 0.0),
                insider_activity=whale_data.get('insider_activity', 0.0),
                smart_money_flow=whale_data.get('smart_money_flow', 0.0),
                
                # Technical indicators
                rsi_14=technical_data.get('rsi_14', 50.0),
                macd_signal=technical_data.get('macd_signal', 0.0),
                bollinger_position=technical_data.get('bollinger_position', 0.5),
                support_distance=technical_data.get('support_distance', 0.0),
                resistance_distance=technical_data.get('resistance_distance', 0.0),
                
                # Network and ecosystem
                transaction_count_24h=price_data.get('tx_count_24h', 0),
                unique_traders_24h=price_data.get('unique_traders_24h', 0),
                dex_presence_count=price_data.get('dex_count', 1),
                
                # Time-based features
                hour_of_day=now.hour,
                day_of_week=now.weekday(),
                is_weekend=now.weekday() >= 5,
                
                # Market context
                btc_price_change=market_context.get('btc_change_24h', 0.0),
                eth_price_change=market_context.get('eth_change_24h', 0.0),
                market_fear_greed=market_context.get('fear_greed_index', 50.0)
            )
            
            logger.info(f"✅ Feature collection complete: {len(asdict(features))} features")
            return features
            
        except Exception as e:
            logger.error(f"Error collecting features: {e}")
            # Return default features on error
            return self._get_default_features()
    
    async def predict_performance(self, token_address: str, symbol: str = None,
                                timeframe: PredictionTimeframe = PredictionTimeframe.HOURS_24) -> PredictionResult:
        """
        Predict token performance using ML models
        """
        try:
            if not ML_AVAILABLE:
                logger.error("ML libraries not available")
                return self._get_default_prediction(token_address, symbol, timeframe)
            
            logger.info(f"🔮 Predicting {timeframe.value} performance for {symbol or token_address}")
            
            # Collect features
            features = await self.collect_features(token_address, symbol)
            
            # Check if model exists and is trained
            model_key = f"{timeframe.value}"
            if model_key not in self.models or self._needs_retraining(timeframe):
                logger.info(f"🔄 Training model for {timeframe.value}")
                await self._train_models(timeframe)
            
            # Prepare features for prediction
            feature_array = self._features_to_array(features)
            
            # Scale features
            if model_key in self.scalers:
                feature_array = self.scalers[model_key].transform([feature_array])
            else:
                feature_array = [feature_array]
            
            # Select features
            if model_key in self.feature_selectors:
                feature_array = self.feature_selectors[model_key].transform(feature_array)
            
            # Get ensemble predictions from multiple models
            predictions = {}
            for model_name, model in self.models.get(model_key, {}).items():
                try:
                    pred = model.predict(feature_array)[0]
                    predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not predictions:
                logger.warning("No model predictions available")
                return self._get_default_prediction(token_address, symbol, timeframe)
            
            # Ensemble prediction (weighted average)
            ensemble_prediction = self._ensemble_predict(predictions)
            
            # Calculate confidence
            confidence_score, confidence_level = self._calculate_confidence(predictions, ensemble_prediction)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(model_key, features)
            
            # Analyze signals
            bullish_signals, bearish_signals = self._analyze_signals(features)
            key_factors = self._get_key_factors(feature_importance, features)
            
            # Calculate risk metrics
            upside_potential, downside_risk, volatility_pred = self._calculate_risk_metrics(
                ensemble_prediction, confidence_score, features
            )
            
            prediction_result = PredictionResult(
                token_address=token_address,
                symbol=symbol or "UNKNOWN",
                timeframe=timeframe,
                predicted_return=ensemble_prediction,
                confidence_level=confidence_level,
                confidence_score=confidence_score,
                feature_importance=feature_importance,
                model_used="ensemble",
                prediction_timestamp=datetime.now(),
                upside_potential=upside_potential,
                downside_risk=downside_risk,
                volatility_prediction=volatility_pred,
                bullish_signals=bullish_signals,
                bearish_signals=bearish_signals,
                key_factors=key_factors
            )
            
            # Store prediction
            await self._store_prediction(prediction_result)
            
            logger.info(f"🎯 Prediction complete: {ensemble_prediction:+.1%} return "
                       f"({confidence_level.value} confidence)")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return self._get_default_prediction(token_address, symbol, timeframe)
    
    async def _train_models(self, timeframe: PredictionTimeframe):
        """Train ML models for specific timeframe"""
        try:
            if not ML_AVAILABLE:
                return
            
            logger.info(f"🎯 Training models for {timeframe.value}")
            
            # Get training data
            training_data = await self._get_training_data(timeframe)
            
            if len(training_data) < self.min_training_samples:
                logger.warning(f"Insufficient training data: {len(training_data)} samples "
                              f"(minimum: {self.min_training_samples})")
                return
            
            # Prepare features and targets
            X, y = self._prepare_training_data(training_data, timeframe)
            
            if len(X) == 0:
                logger.error("No valid training samples")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Feature scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Feature selection
            selector = SelectKBest(score_func=f_regression, k=min(self.feature_selection_k, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)
            
            # Train multiple models
            model_key = timeframe.value
            self.models[model_key] = {}
            self.scalers[model_key] = scaler
            self.feature_selectors[model_key] = selector
            
            for model_name, config in self.model_configs.items():
                try:
                    # Create and train model
                    model = config['model'](**config['params'])
                    model.fit(X_train_selected, y_train)
                    
                    # Evaluate model
                    y_pred = model.predict(X_test_selected)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Calculate accuracy bands
                    accuracy_bands = self._calculate_accuracy_bands(y_test, y_pred)
                    
                    # Store model and performance
                    self.models[model_key][model_name] = model
                    
                    performance = ModelPerformance(
                        model_name=model_name,
                        timeframe=timeframe,
                        mae=mae,
                        mse=mse,
                        rmse=rmse,
                        r2=r2,
                        accuracy_bands=accuracy_bands,
                        last_updated=datetime.now()
                    )
                    
                    self.model_performance[f"{model_name}_{model_key}"] = performance
                    
                    logger.info(f"📊 {model_name}: MAE={mae:.4f}, R²={r2:.4f}, "
                               f"±5% accuracy: {accuracy_bands.get('5%', 0):.1%}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
            
            # Update training time
            self.last_training_time[model_key] = datetime.now()
            
            logger.info(f"✅ Model training complete for {timeframe.value}")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    def _ensemble_predict(self, predictions: Dict[str, float]) -> float:
        """Combine predictions from multiple models"""
        if not predictions:
            return 0.0
        
        # Weight models based on their historical performance
        weights = {
            'random_forest': 0.4,
            'gradient_boost': 0.4,
            'ridge': 0.2
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.2)
            weighted_sum += prediction * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_confidence(self, predictions: Dict[str, float], ensemble_pred: float) -> Tuple[float, ConfidenceLevel]:
        """Calculate prediction confidence based on model agreement"""
        if len(predictions) < 2:
            return 0.5, ConfidenceLevel.MEDIUM
        
        # Calculate standard deviation of predictions
        pred_values = list(predictions.values())
        pred_std = np.std(pred_values)
        
        # Calculate agreement score (lower std = higher confidence)
        max_std = 0.2  # Maximum expected standard deviation
        agreement_score = max(0, 1 - (pred_std / max_std))
        
        # Factor in ensemble prediction magnitude
        magnitude_factor = min(1.0, abs(ensemble_pred) / 0.1)  # Stronger predictions get higher confidence
        
        # Combine factors
        confidence_score = (agreement_score * 0.7 + magnitude_factor * 0.3)
        confidence_score = max(0.1, min(0.95, confidence_score))
        
        # Determine confidence level
        if confidence_score < 0.3:
            confidence_level = ConfidenceLevel.VERY_LOW
        elif confidence_score < 0.5:
            confidence_level = ConfidenceLevel.LOW
        elif confidence_score < 0.7:
            confidence_level = ConfidenceLevel.MEDIUM
        elif confidence_score < 0.85:
            confidence_level = ConfidenceLevel.HIGH
        else:
            confidence_level = ConfidenceLevel.VERY_HIGH
        
        return confidence_score, confidence_level
    
    def _analyze_signals(self, features: FeatureSet) -> Tuple[List[str], List[str]]:
        """Analyze features to identify bullish and bearish signals"""
        bullish_signals = []
        bearish_signals = []
        
        # Sentiment signals
        if features.sentiment_score > 0.7:
            bullish_signals.append("Strong positive sentiment")
        elif features.sentiment_score < 0.3:
            bearish_signals.append("Negative sentiment")
        
        # Whale activity signals
        if features.whale_accumulation > 0.5:
            bullish_signals.append("Whale accumulation detected")
        elif features.whale_accumulation < -0.3:
            bearish_signals.append("Whale distribution pattern")
        
        if features.smart_money_flow > 0.3:
            bullish_signals.append("Smart money inflow")
        elif features.smart_money_flow < -0.3:
            bearish_signals.append("Smart money outflow")
        
        # Technical signals
        if features.rsi_14 < 30:
            bullish_signals.append("RSI oversold (potential bounce)")
        elif features.rsi_14 > 70:
            bearish_signals.append("RSI overbought")
        
        # Volume signals
        if features.volume_24h > features.volume_7d_avg * 2:
            bullish_signals.append("High volume surge")
        elif features.volume_24h < features.volume_7d_avg * 0.5:
            bearish_signals.append("Low volume concern")
        
        # Security signals
        if features.security_score > 80:
            bullish_signals.append("Strong security profile")
        elif features.security_score < 40:
            bearish_signals.append("Security concerns")
        
        # Liquidity signals
        if features.liquidity_usd > 100000:  # $100k+
            bullish_signals.append("Good liquidity")
        elif features.liquidity_usd < 10000:  # <$10k
            bearish_signals.append("Low liquidity risk")
        
        return bullish_signals, bearish_signals
    
    def _get_feature_importance(self, model_key: str, features: FeatureSet) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            if model_key not in self.models:
                return {}
            
            # Get feature importance from random forest (most interpretable)
            rf_model = self.models[model_key].get('random_forest')
            if not rf_model or not hasattr(rf_model, 'feature_importances_'):
                return {}
            
            # Map feature importance back to feature names
            feature_names = list(asdict(features).keys())
            
            # Get selected features indices
            if model_key in self.feature_selectors:
                selected_indices = self.feature_selectors[model_key].get_support()
                selected_features = [feature_names[i] for i, selected in enumerate(selected_indices) if selected]
            else:
                selected_features = feature_names
            
            importance_dict = {}
            for i, feature_name in enumerate(selected_features):
                if i < len(rf_model.feature_importances_):
                    importance_dict[feature_name] = float(rf_model.feature_importances_[i])
            
            # Sort by importance
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _get_key_factors(self, feature_importance: Dict[str, float], features: FeatureSet) -> List[str]:
        """Identify key factors driving the prediction"""
        key_factors = []
        
        # Get top 5 most important features
        top_features = list(feature_importance.keys())[:5]
        
        feature_values = asdict(features)
        
        for feature in top_features:
            if feature in feature_values:
                value = feature_values[feature]
                
                # Create human-readable descriptions
                if feature == 'sentiment_score':
                    if value > 0.7:
                        key_factors.append("Very positive sentiment")
                    elif value < 0.3:
                        key_factors.append("Negative sentiment")
                elif feature == 'whale_activity_score':
                    if value > 0.6:
                        key_factors.append("High whale activity")
                elif feature == 'security_score':
                    if value > 80:
                        key_factors.append("Strong security")
                    elif value < 40:
                        key_factors.append("Security risks")
                elif feature == 'volume_24h':
                    key_factors.append("Volume patterns")
                elif feature == 'price_change_24h':
                    if value > 0.1:
                        key_factors.append("Strong recent momentum")
                    elif value < -0.1:
                        key_factors.append("Recent downtrend")
                else:
                    # Generic factor
                    key_factors.append(feature.replace('_', ' ').title())
        
        return key_factors[:3]  # Return top 3 factors
    
    def _calculate_risk_metrics(self, prediction: float, confidence: float, 
                               features: FeatureSet) -> Tuple[float, float, float]:
        """Calculate upside potential, downside risk, and volatility prediction"""
        
        # Base upside/downside from prediction
        if prediction > 0:
            upside_potential = prediction * (1 + confidence * 0.5)  # Higher confidence = higher upside
            downside_risk = abs(prediction) * 0.3 * (1 - confidence)  # Lower confidence = higher downside
        else:
            upside_potential = abs(prediction) * 0.2 * confidence  # Limited upside for negative predictions
            downside_risk = abs(prediction) * (1 + (1 - confidence) * 0.5)
        
        # Adjust based on features
        volatility_factors = [
            features.volatility_24h,
            features.volume_24h / max(features.volume_7d_avg, 1),  # Volume spike factor
            1 - features.security_score / 100,  # Security risk
            1 - min(features.liquidity_usd / 50000, 1)  # Liquidity risk (capped at $50k)
        ]
        
        avg_volatility_factor = np.mean(volatility_factors)
        volatility_prediction = min(0.5, avg_volatility_factor)  # Cap at 50%
        
        # Adjust risk metrics for volatility
        upside_potential *= (1 + volatility_prediction)
        downside_risk *= (1 + volatility_prediction)
        
        return upside_potential, downside_risk, volatility_prediction
    
    async def _get_training_data(self, timeframe: PredictionTimeframe) -> List[Dict]:
        """Get historical training data from database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get historical features with corresponding future returns
            cursor.execute('''
                SELECT features, price_after_1h, price_after_4h, price_after_24h, price_after_7d
                FROM ml_features
                WHERE timestamp > datetime('now', '-30 days')
                ORDER BY timestamp DESC
                LIMIT 10000
            ''')
            
            rows = cursor.fetchall()
            conn.close()
            
            training_data = []
            for row in rows:
                try:
                    features_dict = json.loads(row[0])
                    sample = {
                        'features': features_dict,
                        'price_after_1h': row[1],
                        'price_after_4h': row[2],
                        'price_after_24h': row[3],
                        'price_after_7d': row[4]
                    }
                    training_data.append(sample)
                except:
                    continue
            
            # If no real data, generate synthetic training data for demo
            if len(training_data) < 100:
                logger.info("Generating synthetic training data for demo")
                training_data = self._generate_synthetic_training_data(1000)
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return []
    
    def _generate_synthetic_training_data(self, count: int) -> List[Dict]:
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        training_data = []
        
        for _ in range(count):
            # Generate random features
            features = {
                'current_price': np.random.uniform(0.0001, 100),
                'volume_24h': np.random.uniform(1000, 1000000),
                'volume_7d_avg': np.random.uniform(1000, 1000000),
                'price_change_1h': np.random.normal(0, 0.05),
                'price_change_24h': np.random.normal(0, 0.15),
                'price_change_7d': np.random.normal(0, 0.3),
                'volatility_24h': np.random.uniform(0.1, 0.8),
                'market_cap': np.random.uniform(100000, 10000000),
                'liquidity_usd': np.random.uniform(5000, 500000),
                'holders_count': np.random.randint(50, 10000),
                'sentiment_score': np.random.uniform(0, 1),
                'social_mentions': np.random.randint(0, 1000),
                'twitter_sentiment': np.random.uniform(0, 1),
                'reddit_sentiment': np.random.uniform(0, 1),
                'news_sentiment': np.random.uniform(0, 1),
                'security_score': np.random.uniform(30, 95),
                'honeypot_risk': np.random.uniform(0, 0.3),
                'ownership_renounced': np.random.choice([True, False]),
                'liquidity_locked': np.random.choice([True, False]),
                'verified_contract': np.random.choice([True, False]),
                'whale_activity_score': np.random.uniform(-1, 1),
                'large_transactions_24h': np.random.randint(0, 50),
                'whale_accumulation': np.random.uniform(-1, 1),
                'insider_activity': np.random.uniform(-1, 1),
                'smart_money_flow': np.random.uniform(-1, 1),
                'rsi_14': np.random.uniform(20, 80),
                'macd_signal': np.random.uniform(-0.1, 0.1),
                'bollinger_position': np.random.uniform(0, 1),
                'support_distance': np.random.uniform(0, 0.2),
                'resistance_distance': np.random.uniform(0, 0.2),
                'transaction_count_24h': np.random.randint(10, 5000),
                'unique_traders_24h': np.random.randint(5, 2000),
                'dex_presence_count': np.random.randint(1, 10),
                'hour_of_day': np.random.randint(0, 23),
                'day_of_week': np.random.randint(0, 6),
                'is_weekend': np.random.choice([True, False]),
                'btc_price_change': np.random.normal(0, 0.05),
                'eth_price_change': np.random.normal(0, 0.05),
                'market_fear_greed': np.random.uniform(10, 90)
            }
            
            # Generate synthetic returns based on features (simplified relationships)
            base_return = (
                features['sentiment_score'] * 0.1 +
                features['whale_accumulation'] * 0.08 +
                features['smart_money_flow'] * 0.06 +
                (features['security_score'] - 50) / 1000 +
                features['price_change_24h'] * 0.3 +
                np.random.normal(0, 0.05)  # Random noise
            )
            
            sample = {
                'features': features,
                'price_after_1h': base_return * 0.2,
                'price_after_4h': base_return * 0.5,
                'price_after_24h': base_return,
                'price_after_7d': base_return * 1.5 + np.random.normal(0, 0.02)
            }
            
            training_data.append(sample)
        
        return training_data
    
    def _prepare_training_data(self, training_data: List[Dict], 
                              timeframe: PredictionTimeframe) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for model training"""
        X = []
        y = []
        
        target_map = {
            PredictionTimeframe.HOURS_1: 'price_after_1h',
            PredictionTimeframe.HOURS_4: 'price_after_4h',
            PredictionTimeframe.HOURS_24: 'price_after_24h',
            PredictionTimeframe.DAYS_7: 'price_after_7d'
        }
        
        target_key = target_map[timeframe]
        
        for sample in training_data:
            try:
                if target_key not in sample or sample[target_key] is None:
                    continue
                
                features_dict = sample['features']
                
                # Convert to feature array
                feature_array = self._dict_to_feature_array(features_dict)
                
                if len(feature_array) == 0:
                    continue
                
                X.append(feature_array)
                y.append(sample[target_key])
                
            except Exception as e:
                continue
        
        return np.array(X), np.array(y)
    
    def _dict_to_feature_array(self, features_dict: Dict) -> List[float]:
        """Convert feature dictionary to array"""
        feature_array = []
        
        # Expected feature order (should match FeatureSet)
        expected_features = [
            'current_price', 'volume_24h', 'volume_7d_avg', 'price_change_1h',
            'price_change_24h', 'price_change_7d', 'volatility_24h', 'market_cap',
            'liquidity_usd', 'holders_count', 'sentiment_score', 'social_mentions',
            'twitter_sentiment', 'reddit_sentiment', 'news_sentiment', 'security_score',
            'honeypot_risk', 'ownership_renounced', 'liquidity_locked', 'verified_contract',
            'whale_activity_score', 'large_transactions_24h', 'whale_accumulation',
            'insider_activity', 'smart_money_flow', 'rsi_14', 'macd_signal',
            'bollinger_position', 'support_distance', 'resistance_distance',
            'transaction_count_24h', 'unique_traders_24h', 'dex_presence_count',
            'hour_of_day', 'day_of_week', 'is_weekend', 'btc_price_change',
            'eth_price_change', 'market_fear_greed'
        ]
        
        for feature in expected_features:
            if feature in features_dict:
                value = features_dict[feature]
                
                # Handle boolean values
                if isinstance(value, bool):
                    value = 1.0 if value else 0.0
                elif value is None:
                    value = 0.0
                
                feature_array.append(float(value))
            else:
                feature_array.append(0.0)  # Default value
        
        return feature_array
    
    def _features_to_array(self, features: FeatureSet) -> List[float]:
        """Convert FeatureSet to array"""
        return self._dict_to_feature_array(asdict(features))
    
    def _calculate_accuracy_bands(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy within different percentage bands"""
        accuracy_bands = {}
        
        bands = [0.05, 0.10, 0.15, 0.20]  # 5%, 10%, 15%, 20%
        
        for band in bands:
            # Calculate percentage errors
            errors = np.abs((y_pred - y_true) / (np.abs(y_true) + 1e-8))  # Add small epsilon
            
            # Count predictions within band
            within_band = np.sum(errors <= band)
            accuracy = within_band / len(errors)
            
            accuracy_bands[f"{band*100:.0f}%"] = accuracy
        
        return accuracy_bands
    
    def _needs_retraining(self, timeframe: PredictionTimeframe) -> bool:
        """Check if model needs retraining"""
        model_key = timeframe.value
        
        if model_key not in self.last_training_time:
            return True
        
        last_training = self.last_training_time[model_key]
        hours_since_training = (datetime.now() - last_training).total_seconds() / 3600
        
        return hours_since_training >= self.retrain_interval_hours
    
    # Placeholder methods for data collection (would integrate with real systems)
    async def _get_price_data(self, token_address: str) -> Dict:
        """Get price and market data"""
        # Placeholder - would integrate with price APIs
        return {
            'current_price': 0.00123,
            'volume_24h': 250000,
            'volume_7d_avg': 180000,
            'price_change_1h': 0.02,
            'price_change_24h': 0.05,
            'price_change_7d': -0.08,
            'volatility_24h': 0.15,
            'market_cap': 1250000,
            'liquidity_usd': 85000,
            'holders_count': 850,
            'tx_count_24h': 420,
            'unique_traders_24h': 180,
            'dex_count': 3
        }
    
    async def _get_sentiment_data(self, token_address: str) -> Dict:
        """Get sentiment data"""
        # Placeholder - would integrate with sentiment APIs
        return {
            'overall_score': 0.72,
            'mentions_24h': 45,
            'twitter_sentiment': 0.68,
            'reddit_sentiment': 0.75,
            'news_sentiment': 0.65
        }
    
    async def _get_security_data(self, token_address: str) -> Dict:
        """Get contract security data"""
        # Placeholder - would integrate with security analysis
        return {
            'security_score': 78.5,
            'honeypot_risk': 0.12,
            'ownership_renounced': True,
            'liquidity_locked': True,
            'verified_contract': True
        }
    
    async def _get_whale_data(self, token_address: str) -> Dict:
        """Get whale activity data"""
        # Placeholder - would integrate with whale monitoring
        return {
            'activity_score': 0.35,
            'large_transactions_24h': 8,
            'accumulation_score': 0.15,
            'insider_activity': -0.05,
            'smart_money_flow': 0.22
        }
    
    async def _get_technical_data(self, token_address: str) -> Dict:
        """Get technical indicators"""
        # Placeholder - would integrate with technical analysis
        return {
            'rsi_14': 65.3,
            'macd_signal': 0.008,
            'bollinger_position': 0.72,
            'support_distance': 0.08,
            'resistance_distance': 0.15
        }
    
    async def _get_market_context(self) -> Dict:
        """Get market context data"""
        # Placeholder - would integrate with market data APIs
        return {
            'btc_change_24h': 0.025,
            'eth_change_24h': 0.018,
            'fear_greed_index': 68.0
        }
    
    def _get_default_features(self) -> FeatureSet:
        """Get default feature set when collection fails"""
        now = datetime.now()
        return FeatureSet(
            current_price=0.001, volume_24h=10000, volume_7d_avg=8000,
            price_change_1h=0.0, price_change_24h=0.0, price_change_7d=0.0,
            volatility_24h=0.2, market_cap=500000, liquidity_usd=25000,
            holders_count=100, sentiment_score=0.5, social_mentions=10,
            twitter_sentiment=0.5, reddit_sentiment=0.5, news_sentiment=0.5,
            security_score=50.0, honeypot_risk=0.2, ownership_renounced=False,
            liquidity_locked=False, verified_contract=False, whale_activity_score=0.0,
            large_transactions_24h=0, whale_accumulation=0.0, insider_activity=0.0,
            smart_money_flow=0.0, rsi_14=50.0, macd_signal=0.0, bollinger_position=0.5,
            support_distance=0.0, resistance_distance=0.0, transaction_count_24h=50,
            unique_traders_24h=25, dex_presence_count=1, hour_of_day=now.hour,
            day_of_week=now.weekday(), is_weekend=now.weekday() >= 5,
            btc_price_change=0.0, eth_price_change=0.0, market_fear_greed=50.0
        )
    
    def _get_default_prediction(self, token_address: str, symbol: str,
                               timeframe: PredictionTimeframe) -> PredictionResult:
        """Get default prediction when ML fails"""
        return PredictionResult(
            token_address=token_address, symbol=symbol or "UNKNOWN",
            timeframe=timeframe, predicted_return=0.0,
            confidence_level=ConfidenceLevel.LOW, confidence_score=0.3,
            feature_importance={}, model_used="fallback",
            prediction_timestamp=datetime.now(), upside_potential=0.05,
            downside_risk=0.1, volatility_prediction=0.3,
            bullish_signals=[], bearish_signals=["Insufficient data"],
            key_factors=["Limited data available"]
        )
    
    async def _store_prediction(self, prediction: PredictionResult):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO ml_predictions (
                    token_address, symbol, timeframe, predicted_return, confidence_level,
                    confidence_score, model_used, prediction_data, prediction_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.token_address, prediction.symbol, prediction.timeframe.value,
                prediction.predicted_return, prediction.confidence_level.value,
                prediction.confidence_score, prediction.model_used,
                json.dumps(asdict(prediction), default=str),
                prediction.prediction_timestamp.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
    
    def get_model_summary(self) -> Dict:
        """Get summary of all trained models and their performance"""
        summary = {
            'models_trained': len(self.models),
            'timeframes': list(self.models.keys()),
            'performance': {}
        }
        
        for key, performance in self.model_performance.items():
            summary['performance'][key] = {
                'mae': performance.mae,
                'r2': performance.r2,
                'accuracy_5pct': performance.accuracy_bands.get('5%', 0),
                'last_updated': performance.last_updated.isoformat()
            }
        
        return summary

# Example usage and testing
async def main():
    """Test the ML predictor"""
    predictor = MLPredictor()
    
    # Test token
    token_address = "0x1234567890123456789012345678901234567890"
    symbol = "TESTCOIN"
    
    print("🧠 Testing ML Performance Predictor")
    print("=" * 50)
    
    try:
        # Test prediction for different timeframes
        timeframes = [
            PredictionTimeframe.HOURS_1,
            PredictionTimeframe.HOURS_24,
            PredictionTimeframe.DAYS_7
        ]
        
        for timeframe in timeframes:
            print(f"\n🔮 Predicting {timeframe.value} performance...")
            
            prediction = await predictor.predict_performance(
                token_address, symbol, timeframe
            )
            
            print(f"Prediction: {prediction.predicted_return:+.2%}")
            print(f"Confidence: {prediction.confidence_level.value} ({prediction.confidence_score:.1%})")
            print(f"Upside potential: {prediction.upside_potential:+.2%}")
            print(f"Downside risk: {prediction.downside_risk:.2%}")
            
            print(f"Bullish signals: {', '.join(prediction.bullish_signals) or 'None'}")
            print(f"Bearish signals: {', '.join(prediction.bearish_signals) or 'None'}")
            print(f"Key factors: {', '.join(prediction.key_factors)}")
        
        # Model summary
        print(f"\n📊 Model Summary:")
        summary = predictor.get_model_summary()
        print(f"Models trained: {summary['models_trained']}")
        print(f"Timeframes: {', '.join(summary['timeframes'])}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())