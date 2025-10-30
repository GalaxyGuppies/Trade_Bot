"""
Feature Engineering Pipeline for ML Model Training
Extracts and processes features from UUID-linked trade data for model training
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Market features
    include_market_features: bool = True
    volatility_windows: List[int] = None  # [1, 4, 24] hours
    momentum_windows: List[int] = None    # [1, 6, 24] hours
    
    # On-chain features
    include_onchain_features: bool = True
    liquidity_ratio_threshold: float = 0.1
    whale_threshold_percent: float = 5.0
    
    # Social features
    include_social_features: bool = True
    sentiment_sources: List[str] = None
    social_velocity_windows: List[int] = None  # [1, 6, 24] hours
    
    # Rugpull features
    include_rugpull_features: bool = True
    risk_score_components: List[str] = None
    
    # Target engineering
    return_windows: List[int] = None      # [1, 6, 24] hours for different return targets
    classification_thresholds: List[float] = None  # [0.05, 0.1, 0.2] for win/loss classification
    
    def __post_init__(self):
        if self.volatility_windows is None:
            self.volatility_windows = [1, 4, 24]
        if self.momentum_windows is None:
            self.momentum_windows = [1, 6, 24]
        if self.sentiment_sources is None:
            self.sentiment_sources = ['twitter', 'reddit', 'telegram', 'news']
        if self.social_velocity_windows is None:
            self.social_velocity_windows = [1, 6, 24]
        if self.risk_score_components is None:
            self.risk_score_components = ['liquidity_risk', 'dev_risk', 'age_risk', 'overall_risk']
        if self.return_windows is None:
            self.return_windows = [1, 6, 24]
        if self.classification_thresholds is None:
            self.classification_thresholds = [0.05, 0.1, 0.2]

class FeatureEngineer:
    """Feature engineering pipeline for trade prediction models"""
    
    def __init__(self, db_path: str = "trade_tracking.db", config: FeatureConfig = None):
        self.db_path = db_path
        self.config = config or FeatureConfig()
        self.feature_importance_cache = {}
        
    def extract_all_features(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Extract all features from UUID-linked trade data"""
        logger.info("Starting feature extraction from trade tracking database")
        
        # Load data from database
        trade_data = self._load_trade_data(start_date, end_date)
        
        if trade_data.empty:
            logger.warning("No trade data found for feature extraction")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(trade_data)} trades for feature engineering")
        
        # Initialize feature dataframe
        features_df = trade_data[['uuid', 'timestamp', 'instrument', 'realized_pnl_percent']].copy()
        
        # Extract different feature categories
        if self.config.include_market_features:
            market_features = self._extract_market_features(trade_data)
            features_df = features_df.merge(market_features, on='uuid', how='left')
        
        if self.config.include_onchain_features:
            onchain_features = self._extract_onchain_features(trade_data)
            features_df = features_df.merge(onchain_features, on='uuid', how='left')
        
        if self.config.include_social_features:
            social_features = self._extract_social_features(trade_data)
            features_df = features_df.merge(social_features, on='uuid', how='left')
        
        if self.config.include_rugpull_features:
            rugpull_features = self._extract_rugpull_features(trade_data)
            features_df = features_df.merge(rugpull_features, on='uuid', how='left')
        
        # Extract execution context features
        execution_features = self._extract_execution_features(trade_data)
        features_df = features_df.merge(execution_features, on='uuid', how='left')
        
        # Generate target labels
        target_labels = self._generate_target_labels(trade_data)
        features_df = features_df.merge(target_labels, on='uuid', how='left')
        
        # Clean and validate features
        features_df = self._clean_features(features_df)
        
        logger.info(f"Generated {len(features_df.columns)} features for {len(features_df)} trades")
        return features_df
    
    def _load_trade_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load complete trade data from database"""
        conn = sqlite3.connect(self.db_path)
        
        # Build query with date filtering
        where_clause = "WHERE o.uuid IS NOT NULL"  # Only completed trades
        params = []
        
        if start_date:
            where_clause += " AND c.timestamp >= ?"
            params.append(start_date)
        if end_date:
            where_clause += " AND c.timestamp <= ?"
            params.append(end_date)
        
        query = f"""
        SELECT 
            c.uuid,
            c.timestamp,
            c.instrument,
            c.contract_address,
            c.model_version,
            c.confidence,
            c.created_by,
            
            r.initial_market_cap,
            r.total_supply,
            r.liquidity_pools,
            r.holder_count,
            r.on_chain_age_days,
            r.audit_status,
            r.verified_source_flag,
            r.social_sentiment_scores,
            r.news_sentiment_scores,
            r.rugpull_heuristic_scores,
            r.trade_rationale,
            r.model_feature_vector,
            r.parameter_snapshot,
            
            e.entry_price,
            e.exit_price,
            e.size,
            e.fees_paid,
            e.slippage,
            e.latency_metrics,
            e.mev_detected,
            
            o.realized_pnl,
            o.realized_pnl_percent,
            o.max_adverse_excursion,
            o.max_favorable_excursion,
            o.time_in_market_seconds,
            o.termination_reason,
            o.execution_quality_score
            
        FROM candidates c
        LEFT JOIN research_docs r ON c.uuid = r.uuid
        LEFT JOIN executions e ON c.uuid = e.uuid
        LEFT JOIN outcomes o ON c.uuid = o.uuid
        {where_clause}
        ORDER BY c.timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Parse JSON columns
        json_columns = ['liquidity_pools', 'social_sentiment_scores', 'news_sentiment_scores',
                       'rugpull_heuristic_scores', 'model_feature_vector', 'parameter_snapshot',
                       'latency_metrics']
        
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if x and isinstance(x, str) else {})
        
        return df
    
    def _extract_market_features(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Extract market-based features"""
        logger.info("Extracting market features")
        
        features = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Market cap features
            market_cap = row.get('initial_market_cap', 0)
            market_cap_log = np.log10(max(market_cap, 1))  # Log transform
            
            # Liquidity features
            liquidity_pools = row.get('liquidity_pools', {})
            total_liquidity = sum(liquidity_pools.values()) if liquidity_pools else 0
            liquidity_to_mcap = total_liquidity / max(market_cap, 1)
            
            # Price features
            entry_price = row.get('entry_price', 0)
            exit_price = row.get('exit_price', 0)
            
            # Spread and execution features
            slippage = row.get('slippage', 0)
            fees_ratio = row.get('fees_paid', 0) / max(row.get('size', 1), 1)
            
            # MEV and execution quality
            mev_detected = int(row.get('mev_detected', False))
            execution_quality = row.get('execution_quality_score', 0.5)
            
            # Latency metrics
            latency_metrics = row.get('latency_metrics', {})
            avg_latency = latency_metrics.get('average_latency_ms', 100)
            
            features.append({
                'uuid': uuid,
                'market_cap_log': market_cap_log,
                'total_liquidity': total_liquidity,
                'liquidity_to_mcap_ratio': liquidity_to_mcap,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'slippage': abs(slippage),
                'fees_ratio': fees_ratio,
                'mev_detected': mev_detected,
                'execution_quality': execution_quality,
                'avg_latency_ms': avg_latency,
                
                # Derived features
                'is_micro_cap': int(market_cap < 1_000_000),
                'is_small_cap': int(1_000_000 <= market_cap < 10_000_000),
                'high_liquidity_ratio': int(liquidity_to_mcap > 0.1),
                'low_slippage': int(abs(slippage) < 0.01),
                'fast_execution': int(avg_latency < 50)
            })
        
        return pd.DataFrame(features)
    
    def _extract_onchain_features(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Extract on-chain based features"""
        logger.info("Extracting on-chain features")
        
        features = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Basic on-chain metrics
            holder_count = row.get('holder_count', 0)
            token_age_days = row.get('on_chain_age_days', 0)
            total_supply = row.get('total_supply', 0)
            market_cap = row.get('initial_market_cap', 0)
            
            # Derived metrics
            holder_density = holder_count / max(market_cap / 1_000_000, 1)  # Holders per $1M mcap
            token_age_score = min(token_age_days / 365.0, 2.0)  # Normalize by years, cap at 2
            
            # Liquidity provider analysis
            liquidity_pools = row.get('liquidity_pools', {})
            num_liquidity_pools = len(liquidity_pools)
            max_pool_dominance = max(liquidity_pools.values()) / max(sum(liquidity_pools.values()), 1) if liquidity_pools else 0
            
            # Contract verification
            audit_status = row.get('audit_status', 'unknown')
            verified_source = int(row.get('verified_source_flag', False))
            
            features.append({
                'uuid': uuid,
                'holder_count': holder_count,
                'token_age_days': token_age_days,
                'total_supply': total_supply,
                'holder_density': holder_density,
                'token_age_score': token_age_score,
                'num_liquidity_pools': num_liquidity_pools,
                'max_pool_dominance': max_pool_dominance,
                'verified_source': verified_source,
                
                # Category features
                'is_audited': int(audit_status == 'audited'),
                'is_unaudited': int(audit_status == 'unaudited'),
                'is_new_token': int(token_age_days < 30),
                'is_established': int(token_age_days > 365),
                'high_holder_count': int(holder_count > 10000),
                'concentrated_liquidity': int(max_pool_dominance > 0.8)
            })
        
        return pd.DataFrame(features)
    
    def _extract_social_features(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Extract social sentiment and momentum features"""
        logger.info("Extracting social features")
        
        features = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Social sentiment scores
            social_scores = row.get('social_sentiment_scores', {})
            news_scores = row.get('news_sentiment_scores', {})
            
            # Aggregate sentiment
            social_sentiment_avg = np.mean(list(social_scores.values())) if social_scores else 0.5
            news_sentiment_avg = np.mean(list(news_scores.values())) if news_scores else 0.5
            combined_sentiment = (social_sentiment_avg + news_sentiment_avg) / 2
            
            # Sentiment variance (disagreement across sources)
            social_sentiment_var = np.var(list(social_scores.values())) if len(social_scores) > 1 else 0
            news_sentiment_var = np.var(list(news_scores.values())) if len(news_scores) > 1 else 0
            
            # Source-specific features
            twitter_sentiment = social_scores.get('twitter', 0.5)
            reddit_sentiment = social_scores.get('reddit', 0.5)
            telegram_sentiment = social_scores.get('telegram', 0.5)
            
            # News sources
            mainstream_news_sentiment = news_scores.get('overall', 0.5)
            
            features.append({
                'uuid': uuid,
                'social_sentiment_avg': social_sentiment_avg,
                'news_sentiment_avg': news_sentiment_avg,
                'combined_sentiment': combined_sentiment,
                'social_sentiment_variance': social_sentiment_var,
                'news_sentiment_variance': news_sentiment_var,
                'twitter_sentiment': twitter_sentiment,
                'reddit_sentiment': reddit_sentiment,
                'telegram_sentiment': telegram_sentiment,
                'mainstream_news_sentiment': mainstream_news_sentiment,
                
                # Derived features
                'bullish_social': int(social_sentiment_avg > 0.6),
                'bearish_social': int(social_sentiment_avg < 0.4),
                'bullish_news': int(news_sentiment_avg > 0.6),
                'bearish_news': int(news_sentiment_avg < 0.4),
                'sentiment_consensus': int(abs(social_sentiment_avg - news_sentiment_avg) < 0.2),
                'high_social_variance': int(social_sentiment_var > 0.1)
            })
        
        return pd.DataFrame(features)
    
    def _extract_rugpull_features(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Extract rugpull risk features"""
        logger.info("Extracting rugpull risk features")
        
        features = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Rugpull risk scores
            rugpull_scores = row.get('rugpull_heuristic_scores', {})
            
            # Overall risk
            overall_risk = rugpull_scores.get('overall_risk', 0.5)
            liquidity_risk = rugpull_scores.get('liquidity_risk', 0.5)
            dev_risk = rugpull_scores.get('dev_risk', 0.5)
            age_risk = rugpull_scores.get('age_risk', 0.5)
            
            # Risk aggregations
            max_risk = max(rugpull_scores.values()) if rugpull_scores else 0.5
            avg_risk = np.mean(list(rugpull_scores.values())) if rugpull_scores else 0.5
            risk_variance = np.var(list(rugpull_scores.values())) if len(rugpull_scores) > 1 else 0
            
            # Combined risk score
            weighted_risk = (
                overall_risk * 0.4 +
                liquidity_risk * 0.3 +
                dev_risk * 0.2 +
                age_risk * 0.1
            )
            
            features.append({
                'uuid': uuid,
                'overall_risk': overall_risk,
                'liquidity_risk': liquidity_risk,
                'dev_risk': dev_risk,
                'age_risk': age_risk,
                'max_risk_score': max_risk,
                'avg_risk_score': avg_risk,
                'risk_variance': risk_variance,
                'weighted_risk': weighted_risk,
                
                # Risk categories
                'low_risk': int(weighted_risk < 0.3),
                'medium_risk': int(0.3 <= weighted_risk <= 0.7),
                'high_risk': int(weighted_risk > 0.7),
                'liquidity_risk_high': int(liquidity_risk > 0.6),
                'dev_risk_high': int(dev_risk > 0.6),
                'age_risk_high': int(age_risk > 0.6)
            })
        
        return pd.DataFrame(features)
    
    def _extract_execution_features(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Extract execution context features"""
        logger.info("Extracting execution features")
        
        features = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Trade timing
            time_in_market = row.get('time_in_market_seconds', 0)
            time_in_market_hours = time_in_market / 3600.0
            
            # Trade size relative to liquidity
            trade_size = row.get('size', 0)
            liquidity_pools = row.get('liquidity_pools', {})
            total_liquidity = sum(liquidity_pools.values()) if liquidity_pools else 1
            size_to_liquidity_ratio = trade_size / max(total_liquidity, 1)
            
            # Model confidence and parameters
            confidence = row.get('confidence', 0.5)
            params = row.get('parameter_snapshot', {})
            stop_loss_pct = params.get('stop_loss_pct', 10.0)
            take_profit_pct = params.get('take_profit_pct', 20.0)
            
            # Termination reason
            termination_reason = row.get('termination_reason', 'unknown')
            
            features.append({
                'uuid': uuid,
                'time_in_market_hours': time_in_market_hours,
                'trade_size': trade_size,
                'size_to_liquidity_ratio': size_to_liquidity_ratio,
                'model_confidence': confidence,
                'stop_loss_pct': stop_loss_pct,
                'take_profit_pct': take_profit_pct,
                
                # Termination reason features
                'exit_profit_target': int(termination_reason == 'profit_target'),
                'exit_stop_loss': int(termination_reason == 'stop_loss'),
                'exit_time_limit': int(termination_reason == 'time_limit'),
                'exit_manual': int(termination_reason == 'manual_exit'),
                'exit_emergency': int(termination_reason == 'emergency_exit'),
                'exit_rug_detected': int(termination_reason == 'rug_detected'),
                
                # Derived features
                'quick_trade': int(time_in_market_hours < 1),
                'medium_trade': int(1 <= time_in_market_hours <= 24),
                'long_trade': int(time_in_market_hours > 24),
                'high_confidence': int(confidence > 0.7),
                'large_trade_size': int(size_to_liquidity_ratio > 0.01),
                'aggressive_targets': int(take_profit_pct > 30 or stop_loss_pct < 10)
            })
        
        return pd.DataFrame(features)
    
    def _generate_target_labels(self, trade_data: pd.DataFrame) -> pd.DataFrame:
        """Generate target labels for supervised learning"""
        logger.info("Generating target labels")
        
        labels = []
        
        for _, row in trade_data.iterrows():
            uuid = row['uuid']
            
            # Primary targets
            realized_pnl_pct = row.get('realized_pnl_percent', 0)
            realized_pnl = row.get('realized_pnl', 0)
            
            # Risk-adjusted returns
            time_in_market_hours = row.get('time_in_market_seconds', 3600) / 3600.0
            max_adverse_excursion = row.get('max_adverse_excursion', 0)
            max_favorable_excursion = row.get('max_favorable_excursion', 0)
            
            # Calculate Sharpe-like ratio (return per unit of max adverse excursion)
            risk_adjusted_return = (realized_pnl_pct / max(abs(max_adverse_excursion), 1.0)) if max_adverse_excursion != 0 else realized_pnl_pct
            
            # Binary classification labels
            profit_5pct = int(realized_pnl_pct > 5.0)
            profit_10pct = int(realized_pnl_pct > 10.0)
            profit_20pct = int(realized_pnl_pct > 20.0)
            
            loss_5pct = int(realized_pnl_pct < -5.0)
            loss_10pct = int(realized_pnl_pct < -10.0)
            loss_15pct = int(realized_pnl_pct < -15.0)
            
            # Multi-class target
            if realized_pnl_pct > 20:
                return_category = 'big_win'
            elif realized_pnl_pct > 10:
                return_category = 'win'
            elif realized_pnl_pct > 0:
                return_category = 'small_win'
            elif realized_pnl_pct > -10:
                return_category = 'small_loss'
            elif realized_pnl_pct > -20:
                return_category = 'loss'
            else:
                return_category = 'big_loss'
            
            labels.append({
                'uuid': uuid,
                'realized_pnl_percent': realized_pnl_pct,
                'realized_pnl': realized_pnl,
                'risk_adjusted_return': risk_adjusted_return,
                'time_in_market_hours': time_in_market_hours,
                'max_adverse_excursion': max_adverse_excursion,
                'max_favorable_excursion': max_favorable_excursion,
                
                # Binary targets
                'profit_5pct': profit_5pct,
                'profit_10pct': profit_10pct,
                'profit_20pct': profit_20pct,
                'loss_5pct': loss_5pct,
                'loss_10pct': loss_10pct,
                'loss_15pct': loss_15pct,
                
                # Multi-class target
                'return_category': return_category,
                
                # Success metrics
                'is_profitable': int(realized_pnl_pct > 0),
                'is_significant_win': int(realized_pnl_pct > 15),
                'is_significant_loss': int(realized_pnl_pct < -15),
                
                # Risk metrics
                'high_risk_adjusted_return': int(risk_adjusted_return > 2.0),
                'quick_profit': int(realized_pnl_pct > 10 and time_in_market_hours < 6)
            })
        
        return pd.DataFrame(labels)
    
    def _clean_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        logger.info("Cleaning and validating features")
        
        # Handle missing values
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(0)
        
        # Handle infinite values
        features_df = features_df.replace([np.inf, -np.inf], 0)
        
        # Remove constant features
        constant_features = []
        for col in numeric_columns:
            if features_df[col].nunique() <= 1:
                constant_features.append(col)
        
        if constant_features:
            features_df = features_df.drop(columns=constant_features)
            logger.info(f"Removed {len(constant_features)} constant features")
        
        # Cap extreme outliers
        for col in numeric_columns:
            if col in features_df.columns:
                q99 = features_df[col].quantile(0.99)
                q01 = features_df[col].quantile(0.01)
                features_df[col] = features_df[col].clip(lower=q01, upper=q99)
        
        logger.info(f"Cleaned features: {features_df.shape}")
        return features_df
    
    def select_features(self, features_df: pd.DataFrame, target_col: str = 'realized_pnl_percent', 
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features using statistical tests"""
        
        # Separate features and target
        feature_cols = [col for col in features_df.columns 
                       if col not in ['uuid', 'timestamp', 'instrument', target_col]]
        
        X = features_df[feature_cols]
        y = features_df[target_col]
        
        # Handle missing target values
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            logger.warning("No valid samples for feature selection")
            return features_df, feature_cols
        
        # Select k best features
        selector = SelectKBest(score_func=f_regression, k=min(k, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        # Calculate feature importance scores
        feature_scores = dict(zip(feature_cols, selector.scores_))
        self.feature_importance_cache[target_col] = feature_scores
        
        logger.info(f"Selected {len(selected_features)} features for {target_col}")
        
        # Return dataframe with selected features
        selected_df = features_df[['uuid', 'timestamp', 'instrument', target_col] + selected_features]
        
        return selected_df, selected_features
    
    def get_feature_importance(self, target_col: str = 'realized_pnl_percent') -> Dict[str, float]:
        """Get cached feature importance scores"""
        return self.feature_importance_cache.get(target_col, {})
    
    def prepare_ml_dataset(self, features_df: pd.DataFrame, 
                          target_col: str = 'realized_pnl_percent',
                          test_size: float = 0.2,
                          validation_size: float = 0.2) -> Dict[str, Any]:
        """Prepare dataset for ML training with proper time-based splits"""
        
        # Sort by timestamp for time-based split
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        
        # Time-based splits (no random shuffling)
        n_samples = len(features_df)
        train_end = int(n_samples * (1 - test_size - validation_size))
        val_end = int(n_samples * (1 - test_size))
        
        train_df = features_df.iloc[:train_end]
        val_df = features_df.iloc[train_end:val_end]
        test_df = features_df.iloc[val_end:]
        
        # Prepare feature matrices
        feature_cols = [col for col in features_df.columns 
                       if col not in ['uuid', 'timestamp', 'instrument'] and not col.startswith('realized_pnl')]
        
        X_train = train_df[feature_cols]
        X_val = val_df[feature_cols]
        X_test = test_df[feature_cols]
        
        y_train = train_df[target_col]
        y_val = val_df[target_col]
        y_test = test_df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train.fillna(0))
        X_val_scaled = scaler.transform(X_val.fillna(0))
        X_test_scaled = scaler.transform(X_test.fillna(0))
        
        logger.info(f"Prepared ML dataset: Train {len(X_train)}, Val {len(X_val)}, Test {len(X_test)}")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values,
            'feature_names': feature_cols,
            'scaler': scaler,
            'train_indices': train_df.index.tolist(),
            'val_indices': val_df.index.tolist(),
            'test_indices': test_df.index.tolist()
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer(db_path="trade_tracking.db")
    
    # Extract all features
    features_df = engineer.extract_all_features()
    
    if not features_df.empty:
        print(f"✅ Extracted features: {features_df.shape}")
        print(f"Feature columns: {len([col for col in features_df.columns if col not in ['uuid', 'timestamp', 'instrument']])}")
        
        # Select top features
        selected_df, selected_features = engineer.select_features(features_df, k=30)
        print(f"✅ Selected {len(selected_features)} top features")
        
        # Prepare ML dataset
        ml_data = engineer.prepare_ml_dataset(selected_df)
        print(f"✅ Prepared ML dataset: {ml_data['X_train'].shape}")
        
        # Show feature importance
        importance = engineer.get_feature_importance()
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print("\n🔝 Top 10 features:")
            for feature, score in top_features:
                print(f"  {feature}: {score:.2f}")
    else:
        print("❌ No trade data found for feature extraction")