"""
Automated ML Training Pipeline with Safety Gates
Implements safe model training, validation, canary deployment, and rollback mechanisms
"""

import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb

# Local imports
from .feature_engineering import FeatureEngineer, FeatureConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str = "lightgbm"  # lightgbm, xgboost, random_forest, gradient_boost
    target_column: str = "realized_pnl_percent"
    max_features: int = 50
    cv_folds: int = 5
    
    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    min_samples: int = 100
    
    # Safety gates
    min_r2_score: float = 0.1
    max_mse_threshold: float = 100.0
    min_improvement_threshold: float = 0.05  # 5% improvement required
    canary_traffic_percent: float = 10.0     # Start with 10% traffic
    
    # Rollback criteria
    performance_window_hours: int = 48
    max_performance_degradation: float = 0.15  # 15% degradation triggers rollback

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    r2_score: float
    mse: float
    mae: float
    cv_score_mean: float
    cv_score_std: float
    feature_count: int
    sample_count: int
    training_time: float

@dataclass
class ModelVersion:
    """Model version tracking"""
    version_id: str
    model_type: str
    created_at: datetime
    metrics: ModelMetrics
    config: Dict[str, Any]
    model_path: str
    feature_names: List[str]
    is_active: bool
    canary_percent: float

class SafeMLPipeline:
    """Safe ML training pipeline with automated deployment and rollback"""
    
    def __init__(self, 
                 db_path: str = "trade_tracking.db",
                 models_dir: str = "models",
                 config: ModelConfig = None):
        
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.config = config or ModelConfig()
        
        # Initialize components
        self.feature_engineer = FeatureEngineer(db_path=db_path)
        
        # Model storage
        self.models_db_path = self.models_dir / "models.db"
        self._init_models_database()
        
        # Performance tracking
        self.performance_cache = {}
        
        logger.info(f"Initialized Safe ML Pipeline with models in {models_dir}")
    
    def _init_models_database(self):
        """Initialize models tracking database"""
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_versions (
            version_id TEXT PRIMARY KEY,
            model_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            r2_score REAL,
            mse REAL,
            mae REAL,
            cv_score_mean REAL,
            cv_score_std REAL,
            feature_count INTEGER,
            sample_count INTEGER,
            training_time REAL,
            config_json TEXT,
            model_path TEXT,
            feature_names TEXT,
            is_active BOOLEAN DEFAULT FALSE,
            canary_percent REAL DEFAULT 0.0
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prediction_accuracy REAL,
            avg_error REAL,
            sample_count INTEGER,
            time_window_hours INTEGER,
            FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS deployment_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version_id TEXT NOT NULL,
            action TEXT NOT NULL, -- deploy, canary, rollback, activate
            timestamp TEXT NOT NULL,
            reason TEXT,
            previous_version_id TEXT,
            FOREIGN KEY (version_id) REFERENCES model_versions(version_id)
        )
        """)
        
        conn.commit()
        conn.close()
    
    def train_new_model(self, start_date: str = None, end_date: str = None) -> Optional[ModelVersion]:
        """Train a new model with safety validation"""
        logger.info("Starting new model training")
        
        try:
            # 1. Extract and prepare features
            features_df = self.feature_engineer.extract_all_features(start_date, end_date)
            
            if len(features_df) < self.config.min_samples:
                logger.warning(f"Insufficient samples: {len(features_df)} < {self.config.min_samples}")
                return None
            
            # 2. Feature selection
            selected_df, selected_features = self.feature_engineer.select_features(
                features_df, 
                target_col=self.config.target_column,
                k=self.config.max_features
            )
            
            # 3. Prepare ML dataset
            ml_data = self.feature_engineer.prepare_ml_dataset(
                selected_df,
                target_col=self.config.target_column,
                test_size=self.config.test_size,
                validation_size=self.config.validation_size
            )
            
            # 4. Train model
            start_time = datetime.now()
            model, metrics = self._train_model(ml_data)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 5. Validate model meets safety criteria
            if not self._validate_model_safety(metrics):
                logger.warning("Model failed safety validation")
                return None
            
            # 6. Save model and create version
            version_id = f"model_{int(datetime.now().timestamp())}"
            model_path = self.models_dir / f"{version_id}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': ml_data['scaler'],
                    'feature_names': ml_data['feature_names'],
                    'config': asdict(self.config),
                    'training_data_info': {
                        'samples': len(features_df),
                        'features': len(selected_features),
                        'start_date': start_date,
                        'end_date': end_date
                    }
                }, f)
            
            # 7. Create model version record
            model_version = ModelVersion(
                version_id=version_id,
                model_type=self.config.model_type,
                created_at=datetime.now(timezone.utc),
                metrics=ModelMetrics(
                    r2_score=metrics['r2'],
                    mse=metrics['mse'],
                    mae=metrics['mae'],
                    cv_score_mean=metrics['cv_mean'],
                    cv_score_std=metrics['cv_std'],
                    feature_count=len(selected_features),
                    sample_count=len(features_df),
                    training_time=training_time
                ),
                config=asdict(self.config),
                model_path=str(model_path),
                feature_names=selected_features,
                is_active=False,
                canary_percent=0.0
            )
            
            # 8. Store in database
            self._store_model_version(model_version)
            
            logger.info(f"Successfully trained model {version_id}")
            logger.info(f"Metrics: R² {metrics['r2']:.3f}, MSE {metrics['mse']:.3f}")
            
            return model_version
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def _train_model(self, ml_data: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train the actual ML model"""
        X_train, y_train = ml_data['X_train'], ml_data['y_train']
        X_val, y_val = ml_data['X_val'], ml_data['y_val']
        X_test, y_test = ml_data['X_test'], ml_data['y_test']
        
        # Select model type
        if self.config.model_type == "lightgbm":
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=-1
            )
        elif self.config.model_type == "xgboost":
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbosity=0
            )
        elif self.config.model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        elif self.config.model_type == "gradient_boost":
            model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        else:
            model = Ridge(alpha=1.0)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=self.config.cv_folds, scoring='r2')
        
        metrics = {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        return model, metrics
    
    def _validate_model_safety(self, metrics: Dict[str, float]) -> bool:
        """Validate model meets safety criteria"""
        safety_checks = []
        
        # R² score check
        r2_check = metrics['r2'] >= self.config.min_r2_score
        safety_checks.append(("R² score", r2_check, f"{metrics['r2']:.3f} >= {self.config.min_r2_score}"))
        
        # MSE check
        mse_check = metrics['mse'] <= self.config.max_mse_threshold
        safety_checks.append(("MSE", mse_check, f"{metrics['mse']:.3f} <= {self.config.max_mse_threshold}"))
        
        # Cross-validation stability check
        cv_stability = metrics['cv_std'] / abs(metrics['cv_mean']) < 0.5 if metrics['cv_mean'] != 0 else False
        safety_checks.append(("CV stability", cv_stability, f"CV std/mean ratio: {metrics['cv_std'] / abs(metrics['cv_mean']):.3f}"))
        
        # Log safety check results
        logger.info("Model safety validation:")
        for check_name, passed, details in safety_checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"  {check_name}: {status} ({details})")
        
        return all(check[1] for check in safety_checks)
    
    def _store_model_version(self, model_version: ModelVersion):
        """Store model version in database"""
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        INSERT INTO model_versions (
            version_id, model_type, created_at, r2_score, mse, mae,
            cv_score_mean, cv_score_std, feature_count, sample_count,
            training_time, config_json, model_path, feature_names,
            is_active, canary_percent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_version.version_id,
            model_version.model_type,
            model_version.created_at.isoformat(),
            model_version.metrics.r2_score,
            model_version.metrics.mse,
            model_version.metrics.mae,
            model_version.metrics.cv_score_mean,
            model_version.metrics.cv_score_std,
            model_version.metrics.feature_count,
            model_version.metrics.sample_count,
            model_version.metrics.training_time,
            json.dumps(model_version.config),
            model_version.model_path,
            json.dumps(model_version.feature_names),
            model_version.is_active,
            model_version.canary_percent
        ))
        
        conn.commit()
        conn.close()
    
    def deploy_canary(self, version_id: str) -> bool:
        """Deploy model as canary with limited traffic"""
        logger.info(f"Deploying canary for model {version_id}")
        
        try:
            # Get current active model
            current_model = self.get_active_model()
            
            # Check if new model shows improvement
            if current_model and not self._check_model_improvement(version_id, current_model.version_id):
                logger.warning("New model doesn't show sufficient improvement")
                return False
            
            # Set canary deployment
            conn = sqlite3.connect(self.models_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            UPDATE model_versions 
            SET canary_percent = ? 
            WHERE version_id = ?
            """, (self.config.canary_traffic_percent, version_id))
            
            # Log deployment
            cursor.execute("""
            INSERT INTO deployment_log (version_id, action, timestamp, reason, previous_version_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                version_id,
                "canary",
                datetime.now(timezone.utc).isoformat(),
                f"Canary deployment with {self.config.canary_traffic_percent}% traffic",
                current_model.version_id if current_model else None
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Canary deployed for {version_id} with {self.config.canary_traffic_percent}% traffic")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying canary: {e}")
            return False
    
    def _check_model_improvement(self, new_version_id: str, current_version_id: str) -> bool:
        """Check if new model shows sufficient improvement"""
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        # Get model metrics
        cursor.execute("SELECT r2_score FROM model_versions WHERE version_id = ?", (new_version_id,))
        new_r2 = cursor.fetchone()[0]
        
        cursor.execute("SELECT r2_score FROM model_versions WHERE version_id = ?", (current_version_id,))
        current_r2 = cursor.fetchone()[0]
        
        conn.close()
        
        improvement = (new_r2 - current_r2) / abs(current_r2) if current_r2 != 0 else 0
        required_improvement = self.config.min_improvement_threshold
        
        logger.info(f"Model improvement: {improvement:.3f} (required: {required_improvement:.3f})")
        
        return improvement >= required_improvement
    
    def promote_canary_to_production(self, version_id: str) -> bool:
        """Promote canary model to full production"""
        logger.info(f"Promoting canary {version_id} to production")
        
        try:
            # Check canary performance
            if not self._validate_canary_performance(version_id):
                logger.warning("Canary performance validation failed")
                return False
            
            conn = sqlite3.connect(self.models_db_path)
            cursor = conn.cursor()
            
            # Deactivate current model
            cursor.execute("UPDATE model_versions SET is_active = FALSE, canary_percent = 0.0")
            
            # Activate new model
            cursor.execute("""
            UPDATE model_versions 
            SET is_active = TRUE, canary_percent = 100.0 
            WHERE version_id = ?
            """, (version_id,))
            
            # Log promotion
            cursor.execute("""
            INSERT INTO deployment_log (version_id, action, timestamp, reason)
            VALUES (?, ?, ?, ?)
            """, (
                version_id,
                "activate",
                datetime.now(timezone.utc).isoformat(),
                "Promoted from canary to production"
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model {version_id} promoted to production")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting canary: {e}")
            return False
    
    def _validate_canary_performance(self, version_id: str) -> bool:
        """Validate canary performance before full deployment"""
        # Check if canary has been running long enough
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT timestamp FROM deployment_log 
        WHERE version_id = ? AND action = 'canary' 
        ORDER BY timestamp DESC LIMIT 1
        """, (version_id,))
        
        result = cursor.fetchone()
        if not result:
            return False
        
        canary_start = datetime.fromisoformat(result[0])
        time_running = (datetime.now(timezone.utc) - canary_start).total_seconds() / 3600
        
        if time_running < 2:  # Minimum 2 hours of canary testing
            logger.info(f"Canary needs more time: {time_running:.1f}h < 2h")
            return False
        
        # Check performance metrics during canary period
        cursor.execute("""
        SELECT AVG(prediction_accuracy), COUNT(*) 
        FROM model_performance 
        WHERE version_id = ? AND timestamp >= ?
        """, (version_id, canary_start.isoformat()))
        
        perf_result = cursor.fetchone()
        conn.close()
        
        if not perf_result[0] or perf_result[1] < 10:  # Need at least 10 predictions
            logger.info("Insufficient canary performance data")
            return False
        
        avg_accuracy = perf_result[0]
        return avg_accuracy > 0.6  # Require 60% prediction accuracy
    
    def rollback_model(self, current_version_id: str, reason: str = "Performance degradation") -> bool:
        """Rollback to previous stable model"""
        logger.warning(f"Rolling back model {current_version_id}: {reason}")
        
        try:
            # Find previous stable model
            conn = sqlite3.connect(self.models_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT version_id FROM deployment_log 
            WHERE action = 'activate' AND version_id != ?
            ORDER BY timestamp DESC LIMIT 1
            """, (current_version_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.error("No previous model found for rollback")
                return False
            
            previous_version_id = result[0]
            
            # Deactivate current model
            cursor.execute("UPDATE model_versions SET is_active = FALSE, canary_percent = 0.0")
            
            # Reactivate previous model
            cursor.execute("""
            UPDATE model_versions 
            SET is_active = TRUE, canary_percent = 100.0 
            WHERE version_id = ?
            """, (previous_version_id,))
            
            # Log rollback
            cursor.execute("""
            INSERT INTO deployment_log (version_id, action, timestamp, reason, previous_version_id)
            VALUES (?, ?, ?, ?, ?)
            """, (
                previous_version_id,
                "rollback",
                datetime.now(timezone.utc).isoformat(),
                reason,
                current_version_id
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Rolled back to model {previous_version_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def get_active_model(self) -> Optional[ModelVersion]:
        """Get currently active model"""
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT * FROM model_versions WHERE is_active = TRUE LIMIT 1
        """)
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        # Parse result into ModelVersion
        columns = ['version_id', 'model_type', 'created_at', 'r2_score', 'mse', 'mae',
                  'cv_score_mean', 'cv_score_std', 'feature_count', 'sample_count',
                  'training_time', 'config_json', 'model_path', 'feature_names',
                  'is_active', 'canary_percent']
        
        row_dict = dict(zip(columns, result))
        
        return ModelVersion(
            version_id=row_dict['version_id'],
            model_type=row_dict['model_type'],
            created_at=datetime.fromisoformat(row_dict['created_at']),
            metrics=ModelMetrics(
                r2_score=row_dict['r2_score'],
                mse=row_dict['mse'],
                mae=row_dict['mae'],
                cv_score_mean=row_dict['cv_score_mean'],
                cv_score_std=row_dict['cv_score_std'],
                feature_count=row_dict['feature_count'],
                sample_count=row_dict['sample_count'],
                training_time=row_dict['training_time']
            ),
            config=json.loads(row_dict['config_json']),
            model_path=row_dict['model_path'],
            feature_names=json.loads(row_dict['feature_names']),
            is_active=row_dict['is_active'],
            canary_percent=row_dict['canary_percent']
        )
    
    def load_model(self, version_id: str = None) -> Optional[Dict[str, Any]]:
        """Load model for predictions"""
        if not version_id:
            active_model = self.get_active_model()
            if not active_model:
                return None
            version_id = active_model.version_id
        
        # Get model path
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT model_path FROM model_versions WHERE version_id = ?", (version_id,))
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        # Load model
        try:
            with open(result[0], 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {version_id}: {e}")
            return None
    
    def run_automated_training_cycle(self):
        """Run automated training and deployment cycle"""
        logger.info("Starting automated training cycle")
        
        # 1. Train new model
        new_model = self.train_new_model()
        
        if not new_model:
            logger.info("No new model trained")
            return
        
        # 2. Deploy as canary
        if self.deploy_canary(new_model.version_id):
            logger.info(f"Canary deployed for {new_model.version_id}")
            
            # Note: In a real system, you would have a separate process
            # monitoring canary performance and deciding when to promote
            # For demo purposes, we'll just log the next steps
            logger.info("Monitor canary performance before promoting to production")
            logger.info("Use promote_canary_to_production() when ready")
        
        # 3. Monitor and potentially rollback
        self._check_and_rollback_if_needed()
    
    def _check_and_rollback_if_needed(self):
        """Check current model performance and rollback if degraded"""
        active_model = self.get_active_model()
        if not active_model:
            return
        
        # Check recent performance
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        since_time = (datetime.now(timezone.utc) - timedelta(hours=self.config.performance_window_hours)).isoformat()
        
        cursor.execute("""
        SELECT AVG(prediction_accuracy) 
        FROM model_performance 
        WHERE version_id = ? AND timestamp >= ?
        """, (active_model.version_id, since_time))
        
        result = cursor.fetchone()
        conn.close()
        
        if result[0] and result[0] < (active_model.metrics.r2_score * (1 - self.config.max_performance_degradation)):
            self.rollback_model(active_model.version_id, "Performance degradation detected")

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of all model performance"""
        conn = sqlite3.connect(self.models_db_path)
        cursor = conn.cursor()
        
        # Get all models
        cursor.execute("""
        SELECT version_id, model_type, created_at, r2_score, is_active, canary_percent
        FROM model_versions 
        ORDER BY created_at DESC
        """)
        
        models = cursor.fetchall()
        
        # Get recent deployments
        cursor.execute("""
        SELECT version_id, action, timestamp, reason 
        FROM deployment_log 
        ORDER BY timestamp DESC LIMIT 10
        """)
        
        deployments = cursor.fetchall()
        
        conn.close()
        
        return {
            'models': [dict(zip(['version_id', 'model_type', 'created_at', 'r2_score', 'is_active', 'canary_percent'], m)) for m in models],
            'recent_deployments': [dict(zip(['version_id', 'action', 'timestamp', 'reason'], d)) for d in deployments]
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = SafeMLPipeline(
        db_path="trade_tracking.db",
        models_dir="trading_models"
    )
    
    # Run automated training cycle
    pipeline.run_automated_training_cycle()
    
    # Get performance summary
    summary = pipeline.get_model_performance_summary()
    print(f"✅ Model Performance Summary:")
    print(f"  Models trained: {len(summary['models'])}")
    print(f"  Recent deployments: {len(summary['recent_deployments'])}")
    
    # Show active model
    active_model = pipeline.get_active_model()
    if active_model:
        print(f"  Active model: {active_model.version_id} (R²: {active_model.metrics.r2_score:.3f})")
    else:
        print("  No active model")