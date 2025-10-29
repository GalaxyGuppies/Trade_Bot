"""
Configuration management for the trading bot
"""
import os
import json
from typing import Dict, Any, Optional
from pathlib import Path

class Config:
    """Configuration manager for the trading bot"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.json"
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with environment variable overrides"""
        # Default configuration
        default_config = {
            "api_keys": {
                "twitter": {
                    "bearer_token": os.getenv("TWITTER_BEARER_TOKEN", "")
                },
                "reddit": {
                    "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
                    "client_secret": os.getenv("REDDIT_CLIENT_SECRET", "")
                },
                "binance": {
                    "api_key": os.getenv("BINANCE_API_KEY", ""),
                    "secret": os.getenv("BINANCE_SECRET", ""),
                    "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true"
                },
                "coinbase": {
                    "api_key": os.getenv("COINBASE_API_KEY", ""),
                    "secret": os.getenv("COINBASE_SECRET", ""),
                    "passphrase": os.getenv("COINBASE_PASSPHRASE", ""),
                    "sandbox": os.getenv("COINBASE_SANDBOX", "true").lower() == "true"
                }
            },
            "database": {
                "url": os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/trading_bot"),
                "timescale_url": os.getenv("TIMESCALE_URL", "postgresql://postgres:password@localhost:5433/timeseries")
            },
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://localhost:6379")
            },
            "trading": {
                "max_position_size": float(os.getenv("MAX_POSITION_SIZE", "1000.0")),
                "max_positions": int(os.getenv("MAX_POSITIONS", "10")),
                "daily_loss_limit": float(os.getenv("DAILY_LOSS_LIMIT", "-500.0")),
                "stop_loss_pct": float(os.getenv("STOP_LOSS_PCT", "0.05")),
                "take_profit_pct": float(os.getenv("TAKE_PROFIT_PCT", "0.15")),
                "position_size_base": float(os.getenv("POSITION_SIZE_BASE", "0.02")),
                "enabled_exchanges": os.getenv("ENABLED_EXCHANGES", "binance").split(","),
                "trading_pairs": os.getenv("TRADING_PAIRS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
            },
            "sentiment": {
                "threshold": float(os.getenv("SENTIMENT_THRESHOLD", "0.6")),
                "model_name": os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
                "collection_interval": int(os.getenv("SENTIMENT_INTERVAL", "300"))  # 5 minutes
            },
            "risk": {
                "rugpull_threshold": float(os.getenv("RUGPULL_THRESHOLD", "0.3")),
                "min_confidence": float(os.getenv("MIN_CONFIDENCE", "0.6")),
                "max_daily_trades": int(os.getenv("MAX_DAILY_TRADES", "50"))
            },
            "monitoring": {
                "log_level": os.getenv("LOG_LEVEL", "INFO"),
                "metrics_port": int(os.getenv("METRICS_PORT", "8001")),
                "health_check_interval": int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
            },
            "data_collection": {
                "market_data_interval": int(os.getenv("MARKET_DATA_INTERVAL", "1")),  # 1 second
                "historical_days": int(os.getenv("HISTORICAL_DAYS", "30")),
                "cleanup_interval": int(os.getenv("CLEANUP_INTERVAL", "3600"))  # 1 hour
            }
        }
        
        # Load from file if exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    # Merge with defaults
                    default_config.update(file_config)
            except Exception as e:
                print(f"Warning: Could not load config file {self.config_file}: {e}")
        
        return default_config
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'trading.max_positions')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any):
        """Set a configuration value using dot notation"""
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the parent dict
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors"""
        errors = {}
        
        # Check required API keys for enabled features
        if not self.get("api_keys.binance.api_key") and "binance" in self.get("trading.enabled_exchanges", []):
            errors["binance_api"] = "Binance API key is required when Binance is enabled"
        
        # Check database connections
        if not self.get("database.url"):
            errors["database"] = "Database URL is required"
        
        # Check trading parameters
        if self.get("trading.max_position_size", 0) <= 0:
            errors["position_size"] = "Max position size must be positive"
        
        if self.get("trading.stop_loss_pct", 0) <= 0:
            errors["stop_loss"] = "Stop loss percentage must be positive"
        
        return errors
    
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    def get_log_level(self) -> str:
        """Get the appropriate log level"""
        return self.get("monitoring.log_level", "INFO").upper()

# Global configuration instance
config = Config()