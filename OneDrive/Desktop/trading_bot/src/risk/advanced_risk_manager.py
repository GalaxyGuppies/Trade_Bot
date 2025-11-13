"""
Advanced Risk Management System for Microcap Trading
Provides sophisticated risk controls with correlation limits and circuit breakers.
"""

import asyncio
import logging
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import time
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"

class AlertType(Enum):
    POSITION_LIMIT = "position_limit"
    PORTFOLIO_RISK = "portfolio_risk"
    CORRELATION_RISK = "correlation_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    CIRCUIT_BREAKER = "circuit_breaker"
    DRAWDOWN_LIMIT = "drawdown_limit"

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    var_95: float  # Value at Risk (95%)
    expected_shortfall: float  # Expected Shortfall (CVaR)
    max_drawdown: float  # Maximum drawdown
    sharpe_ratio: float  # Risk-adjusted returns
    sortino_ratio: float  # Downside risk-adjusted returns
    beta: float  # Market correlation
    volatility: float  # Price volatility
    liquidity_score: float  # Liquidity assessment

@dataclass
class PositionRisk:
    """Risk assessment for individual position"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    risk_level: RiskLevel
    risk_metrics: RiskMetrics
    correlation_risk: float
    liquidity_risk: float
    rugpull_score: float
    time_decay_risk: float
    last_updated: datetime

@dataclass
class PortfolioRisk:
    """Overall portfolio risk assessment"""
    total_exposure: float
    available_capital: float
    portfolio_var: float
    concentration_risk: float
    correlation_risk: float
    liquidity_risk: float
    max_daily_loss: float
    current_drawdown: float
    risk_budget_used: float
    alerts: List[str]
    last_updated: datetime

@dataclass
class RiskAlert:
    """Risk management alert"""
    id: str
    timestamp: datetime
    alert_type: AlertType
    severity: RiskLevel
    message: str
    symbol: Optional[str]
    metric_value: float
    threshold: float
    action_taken: str
    resolved: bool = False

class AdvancedRiskManager:
    """
    Advanced risk management system with:
    - Dynamic position sizing
    - Portfolio correlation limits
    - Circuit breakers
    - Liquidity monitoring
    - Rugpull detection
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.risk_limits = self._load_risk_limits()
        self.correlation_matrix = {}
        self.price_history = {}
        self.alerts = []
        self.circuit_breakers_active = False
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_portfolio_var = 0.10  # 10% max portfolio VaR
        self.max_single_position = 0.02  # 2% max single position
        self.correlation_limit = 0.7  # Max correlation between positions
        
        self._init_database()
        self._load_price_history()
    
    def _init_database(self):
        """Initialize risk management database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Risk metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    var_95 REAL,
                    expected_shortfall REAL,
                    max_drawdown REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    beta REAL,
                    volatility REAL,
                    liquidity_score REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_alerts (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    symbol TEXT,
                    metric_value REAL,
                    threshold REAL,
                    action_taken TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Portfolio risk snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_risk_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    total_exposure REAL,
                    portfolio_var REAL,
                    concentration_risk REAL,
                    correlation_risk REAL,
                    liquidity_risk REAL,
                    current_drawdown REAL,
                    risk_budget_used REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Risk management database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk database: {e}")
    
    def _load_risk_limits(self) -> Dict:
        """Load risk limits configuration"""
        default_limits = {
            "max_position_size": 0.02,  # 2% of portfolio
            "max_daily_loss": 0.05,  # 5% daily loss
            "max_portfolio_var": 0.10,  # 10% portfolio VaR
            "max_correlation": 0.7,  # 70% max correlation
            "min_liquidity_score": 0.3,  # Minimum liquidity
            "max_rugpull_score": 0.3,  # Maximum rugpull risk
            "circuit_breaker_loss": 0.08,  # 8% circuit breaker
            "concentration_limit": 0.25,  # 25% in single sector
        }
        
        try:
            with open("risk_limits.json", "r") as f:
                limits = json.load(f)
            return {**default_limits, **limits}
        except FileNotFoundError:
            logger.info("Using default risk limits")
            return default_limits
    
    def _load_price_history(self):
        """Load recent price history for risk calculations"""
        try:
            conn = sqlite3.connect(self.database_path)
            query = '''
                SELECT symbol, timestamp, close_price
                FROM price_data
                WHERE timestamp > datetime('now', '-30 days')
                ORDER BY symbol, timestamp
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol].copy()
                symbol_data['timestamp'] = pd.to_datetime(symbol_data['timestamp'])
                symbol_data = symbol_data.sort_values('timestamp')
                self.price_history[symbol] = symbol_data
                
        except Exception as e:
            logger.error(f"Error loading price history: {e}")
    
    async def calculate_position_risk(self, symbol: str, position_size: float, 
                                    current_price: float) -> PositionRisk:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Get price history for the symbol
            if symbol not in self.price_history:
                logger.warning(f"No price history for {symbol}")
                return self._create_default_position_risk(symbol, position_size, current_price)
            
            price_data = self.price_history[symbol].copy()
            if len(price_data) < 10:
                logger.warning(f"Insufficient price history for {symbol}")
                return self._create_default_position_risk(symbol, position_size, current_price)
            
            # Calculate returns
            price_data['returns'] = price_data['close_price'].pct_change().dropna()
            returns = price_data['returns'].dropna()
            
            if len(returns) < 5:
                return self._create_default_position_risk(symbol, position_size, current_price)
            
            # Risk metrics calculations
            var_95 = np.percentile(returns, 5) * position_size * current_price
            expected_shortfall = returns[returns <= np.percentile(returns, 5)].mean() * position_size * current_price
            
            # Calculate drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe and Sortino ratios
            mean_return = returns.mean()
            volatility = returns.std()
            sharpe_ratio = (mean_return / volatility) if volatility > 0 else 0
            
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return / downside_volatility) if downside_volatility > 0 else 0
            
            # Beta calculation (vs portfolio average)
            beta = self._calculate_beta(returns)
            
            # Liquidity score
            liquidity_score = self._calculate_liquidity_score(symbol, price_data)
            
            # Rugpull score
            rugpull_score = self._calculate_rugpull_score(symbol, price_data, returns)
            
            # Correlation risk
            correlation_risk = self._calculate_correlation_risk(symbol, returns)
            
            # Time decay risk (for volatile positions)
            time_decay_risk = self._calculate_time_decay_risk(returns, volatility)
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                volatility, liquidity_score, rugpull_score, correlation_risk
            )
            
            risk_metrics = RiskMetrics(
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                beta=beta,
                volatility=volatility,
                liquidity_score=liquidity_score
            )
            
            market_value = position_size * current_price
            unrealized_pnl = market_value - (position_size * price_data['close_price'].iloc[0])
            
            position_risk = PositionRisk(
                symbol=symbol,
                position_size=position_size,
                market_value=market_value,
                unrealized_pnl=unrealized_pnl,
                risk_level=risk_level,
                risk_metrics=risk_metrics,
                correlation_risk=correlation_risk,
                liquidity_risk=1.0 - liquidity_score,
                rugpull_score=rugpull_score,
                time_decay_risk=time_decay_risk,
                last_updated=datetime.now()
            )
            
            # Store risk metrics in database
            await self._store_risk_metrics(symbol, risk_metrics)
            
            return position_risk
            
        except Exception as e:
            logger.error(f"Error calculating position risk for {symbol}: {e}")
            return self._create_default_position_risk(symbol, position_size, current_price)
    
    def _create_default_position_risk(self, symbol: str, position_size: float, 
                                    current_price: float) -> PositionRisk:
        """Create default risk assessment when calculations fail"""
        default_metrics = RiskMetrics(
            var_95=-0.1,
            expected_shortfall=-0.15,
            max_drawdown=-0.2,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            beta=1.0,
            volatility=0.5,
            liquidity_score=0.5
        )
        
        return PositionRisk(
            symbol=symbol,
            position_size=position_size,
            market_value=position_size * current_price,
            unrealized_pnl=0.0,
            risk_level=RiskLevel.HIGH,
            risk_metrics=default_metrics,
            correlation_risk=0.5,
            liquidity_risk=0.5,
            rugpull_score=0.5,
            time_decay_risk=0.5,
            last_updated=datetime.now()
        )
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta vs market/portfolio average"""
        try:
            # Simple beta calculation vs equal-weighted portfolio
            if len(self.price_history) > 1:
                # Calculate average market returns
                all_returns = []
                for symbol_data in self.price_history.values():
                    if len(symbol_data) > 1:
                        symbol_returns = symbol_data['close_price'].pct_change().dropna()
                        all_returns.extend(symbol_returns.tolist())
                
                if len(all_returns) > 10:
                    market_returns = pd.Series(all_returns)
                    covariance = np.cov(returns.values, market_returns.values)[0, 1]
                    market_variance = np.var(market_returns.values)
                    return covariance / market_variance if market_variance > 0 else 1.0
            
            return 1.0  # Default beta
            
        except Exception:
            return 1.0
    
    def _calculate_liquidity_score(self, symbol: str, price_data: pd.DataFrame) -> float:
        """Calculate liquidity score based on price stability and volume patterns"""
        try:
            # Price stability component
            price_changes = price_data['close_price'].pct_change().abs()
            avg_price_change = price_changes.mean()
            price_stability = max(0, 1.0 - (avg_price_change * 10))
            
            # Volume consistency (if available)
            volume_consistency = 0.5  # Default
            
            # Recent trading activity
            recent_activity = min(1.0, len(price_data) / 30.0)
            
            # Combined liquidity score
            liquidity_score = (price_stability * 0.4 + 
                             volume_consistency * 0.3 + 
                             recent_activity * 0.3)
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception:
            return 0.5  # Default moderate liquidity
    
    def _calculate_rugpull_score(self, symbol: str, price_data: pd.DataFrame, 
                               returns: pd.Series) -> float:
        """Calculate rugpull risk score"""
        try:
            rugpull_indicators = 0.0
            
            # Sudden large drops
            large_drops = returns[returns < -0.5]  # >50% drops
            if len(large_drops) > 0:
                rugpull_indicators += 0.3
            
            # Extreme volatility
            if returns.std() > 1.0:  # >100% volatility
                rugpull_indicators += 0.2
            
            # Suspicious price patterns
            price_jumps = price_data['close_price'].pct_change().abs()
            if price_jumps.max() > 2.0:  # >200% single-day moves
                rugpull_indicators += 0.2
            
            # Liquidity concerns
            if len(price_data) < 7:  # Less than a week of data
                rugpull_indicators += 0.3
            
            return min(1.0, rugpull_indicators)
            
        except Exception:
            return 0.5  # Default moderate risk
    
    def _calculate_correlation_risk(self, symbol: str, returns: pd.Series) -> float:
        """Calculate correlation risk with existing positions"""
        try:
            max_correlation = 0.0
            
            for other_symbol, other_data in self.price_history.items():
                if other_symbol == symbol or len(other_data) < 10:
                    continue
                
                other_returns = other_data['close_price'].pct_change().dropna()
                
                # Align data
                min_length = min(len(returns), len(other_returns))
                if min_length > 5:
                    correlation = np.corrcoef(
                        returns.tail(min_length).values,
                        other_returns.tail(min_length).values
                    )[0, 1]
                    
                    if not np.isnan(correlation):
                        max_correlation = max(max_correlation, abs(correlation))
            
            return max_correlation
            
        except Exception:
            return 0.0
    
    def _calculate_time_decay_risk(self, returns: pd.Series, volatility: float) -> float:
        """Calculate time decay risk for volatile positions"""
        try:
            # Higher volatility = higher time decay risk
            volatility_component = min(1.0, volatility * 2)
            
            # Recent performance trend
            recent_returns = returns.tail(5)
            if len(recent_returns) > 0:
                trend_component = max(0.0, -recent_returns.mean())
            else:
                trend_component = 0.5
            
            time_decay = (volatility_component * 0.7 + trend_component * 0.3)
            return min(1.0, time_decay)
            
        except Exception:
            return 0.5
    
    def _determine_risk_level(self, volatility: float, liquidity_score: float,
                            rugpull_score: float, correlation_risk: float) -> RiskLevel:
        """Determine overall risk level for position"""
        risk_score = (
            (volatility * 0.3) +
            ((1.0 - liquidity_score) * 0.25) +
            (rugpull_score * 0.25) +
            (correlation_risk * 0.2)
        )
        
        if risk_score > 0.8:
            return RiskLevel.EXTREME
        elif risk_score > 0.6:
            return RiskLevel.HIGH
        elif risk_score > 0.4:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    async def calculate_portfolio_risk(self, positions: List[PositionRisk]) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if not positions:
                return self._create_empty_portfolio_risk()
            
            # Portfolio totals
            total_exposure = sum(pos.market_value for pos in positions)
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
            
            # Portfolio VaR calculation
            individual_vars = [pos.risk_metrics.var_95 for pos in positions]
            portfolio_var = sum(individual_vars)  # Conservative approach
            
            # Concentration risk
            max_position = max(pos.market_value for pos in positions)
            concentration_risk = max_position / total_exposure if total_exposure > 0 else 0
            
            # Correlation risk
            correlation_risk = max(pos.correlation_risk for pos in positions)
            
            # Liquidity risk
            liquidity_risk = max(pos.liquidity_risk for pos in positions)
            
            # Current drawdown
            current_drawdown = min(pos.risk_metrics.max_drawdown for pos in positions)
            
            # Risk budget utilization
            risk_budget_used = portfolio_var / self.max_portfolio_var
            
            # Generate alerts
            alerts = await self._generate_portfolio_alerts(
                total_exposure, portfolio_var, concentration_risk,
                correlation_risk, liquidity_risk, current_drawdown
            )
            
            portfolio_risk = PortfolioRisk(
                total_exposure=total_exposure,
                available_capital=0.0,  # To be set by caller
                portfolio_var=portfolio_var,
                concentration_risk=concentration_risk,
                correlation_risk=correlation_risk,
                liquidity_risk=liquidity_risk,
                max_daily_loss=self.daily_loss_limit,
                current_drawdown=current_drawdown,
                risk_budget_used=risk_budget_used,
                alerts=alerts,
                last_updated=datetime.now()
            )
            
            # Store portfolio risk snapshot
            await self._store_portfolio_risk(portfolio_risk)
            
            return portfolio_risk
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return self._create_empty_portfolio_risk()
    
    def _create_empty_portfolio_risk(self) -> PortfolioRisk:
        """Create empty portfolio risk when no positions"""
        return PortfolioRisk(
            total_exposure=0.0,
            available_capital=0.0,
            portfolio_var=0.0,
            concentration_risk=0.0,
            correlation_risk=0.0,
            liquidity_risk=0.0,
            max_daily_loss=self.daily_loss_limit,
            current_drawdown=0.0,
            risk_budget_used=0.0,
            alerts=[],
            last_updated=datetime.now()
        )
    
    async def _generate_portfolio_alerts(self, total_exposure: float, portfolio_var: float,
                                       concentration_risk: float, correlation_risk: float,
                                       liquidity_risk: float, current_drawdown: float) -> List[str]:
        """Generate risk alerts for portfolio"""
        alerts = []
        
        # Portfolio VaR alert
        if portfolio_var > self.max_portfolio_var:
            alert = await self._create_alert(
                AlertType.PORTFOLIO_RISK,
                RiskLevel.HIGH,
                f"Portfolio VaR ({portfolio_var:.2%}) exceeds limit ({self.max_portfolio_var:.2%})",
                metric_value=portfolio_var,
                threshold=self.max_portfolio_var
            )
            alerts.append(alert.message)
        
        # Concentration risk alert
        if concentration_risk > self.risk_limits["max_position_size"]:
            alert = await self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.MODERATE,
                f"Position concentration ({concentration_risk:.2%}) exceeds limit",
                metric_value=concentration_risk,
                threshold=self.risk_limits["max_position_size"]
            )
            alerts.append(alert.message)
        
        # Correlation risk alert
        if correlation_risk > self.correlation_limit:
            alert = await self._create_alert(
                AlertType.CORRELATION_RISK,
                RiskLevel.MODERATE,
                f"High correlation risk ({correlation_risk:.2%}) detected",
                metric_value=correlation_risk,
                threshold=self.correlation_limit
            )
            alerts.append(alert.message)
        
        # Liquidity risk alert
        if liquidity_risk > 0.7:
            alert = await self._create_alert(
                AlertType.LIQUIDITY_RISK,
                RiskLevel.HIGH,
                f"High liquidity risk ({liquidity_risk:.2%}) in portfolio",
                metric_value=liquidity_risk,
                threshold=0.7
            )
            alerts.append(alert.message)
        
        # Drawdown alert
        if current_drawdown < -0.15:  # 15% drawdown
            alert = await self._create_alert(
                AlertType.DRAWDOWN_LIMIT,
                RiskLevel.HIGH,
                f"High drawdown ({current_drawdown:.2%}) detected",
                metric_value=current_drawdown,
                threshold=-0.15
            )
            alerts.append(alert.message)
        
        return alerts
    
    async def _create_alert(self, alert_type: AlertType, severity: RiskLevel,
                          message: str, symbol: str = None, metric_value: float = 0.0,
                          threshold: float = 0.0, action_taken: str = "Monitor") -> RiskAlert:
        """Create and store risk alert"""
        alert_id = f"{alert_type.value}_{int(time.time())}"
        
        alert = RiskAlert(
            id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            symbol=symbol,
            metric_value=metric_value,
            threshold=threshold,
            action_taken=action_taken
        )
        
        self.alerts.append(alert)
        await self._store_alert(alert)
        
        logger.warning(f"Risk Alert: {message}")
        return alert
    
    async def check_circuit_breakers(self, portfolio_risk: PortfolioRisk) -> bool:
        """Check if circuit breakers should be triggered"""
        try:
            should_trigger = False
            
            # Daily loss limit
            if portfolio_risk.current_drawdown < -self.daily_loss_limit:
                await self._create_alert(
                    AlertType.CIRCUIT_BREAKER,
                    RiskLevel.EXTREME,
                    f"Daily loss limit exceeded: {portfolio_risk.current_drawdown:.2%}",
                    metric_value=portfolio_risk.current_drawdown,
                    threshold=-self.daily_loss_limit,
                    action_taken="Trading halted"
                )
                should_trigger = True
            
            # Portfolio VaR limit
            if portfolio_risk.portfolio_var > self.risk_limits["circuit_breaker_loss"]:
                await self._create_alert(
                    AlertType.CIRCUIT_BREAKER,
                    RiskLevel.EXTREME,
                    f"Portfolio VaR limit exceeded: {portfolio_risk.portfolio_var:.2%}",
                    metric_value=portfolio_risk.portfolio_var,
                    threshold=self.risk_limits["circuit_breaker_loss"],
                    action_taken="Trading halted"
                )
                should_trigger = True
            
            if should_trigger:
                self.circuit_breakers_active = True
                logger.critical("Circuit breakers activated - trading halted")
            
            return should_trigger
            
        except Exception as e:
            logger.error(f"Error checking circuit breakers: {e}")
            return False
    
    def calculate_optimal_position_size(self, symbol: str, risk_level: RiskLevel,
                                      account_value: float, confidence: float) -> float:
        """Calculate optimal position size based on risk management"""
        try:
            # Base position size by risk level
            base_sizes = {
                RiskLevel.LOW: 0.01,      # 1%
                RiskLevel.MODERATE: 0.015, # 1.5%
                RiskLevel.HIGH: 0.02,     # 2%
                RiskLevel.EXTREME: 0.005  # 0.5%
            }
            
            base_size = base_sizes.get(risk_level, 0.01)
            
            # Adjust for confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x
            
            # Adjust for circuit breakers
            if self.circuit_breakers_active:
                base_size *= 0.5
            
            # Calculate final position size
            position_size = base_size * confidence_multiplier * account_value
            
            # Apply hard limits
            max_position = account_value * self.risk_limits["max_position_size"]
            position_size = min(position_size, max_position)
            
            return max(0.0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    async def _store_risk_metrics(self, symbol: str, metrics: RiskMetrics):
        """Store risk metrics in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_metrics (
                    symbol, timestamp, var_95, expected_shortfall, max_drawdown,
                    sharpe_ratio, sortino_ratio, beta, volatility, liquidity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, datetime.now(), metrics.var_95, metrics.expected_shortfall,
                metrics.max_drawdown, metrics.sharpe_ratio, metrics.sortino_ratio,
                metrics.beta, metrics.volatility, metrics.liquidity_score
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing risk metrics: {e}")
    
    async def _store_alert(self, alert: RiskAlert):
        """Store risk alert in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_alerts (
                    id, timestamp, alert_type, severity, message, symbol,
                    metric_value, threshold, action_taken, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.timestamp, alert.alert_type.value,
                alert.severity.value, alert.message, alert.symbol,
                alert.metric_value, alert.threshold, alert.action_taken, alert.resolved
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    async def _store_portfolio_risk(self, portfolio_risk: PortfolioRisk):
        """Store portfolio risk snapshot"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO portfolio_risk_snapshots (
                    timestamp, total_exposure, portfolio_var, concentration_risk,
                    correlation_risk, liquidity_risk, current_drawdown, risk_budget_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                portfolio_risk.last_updated, portfolio_risk.total_exposure,
                portfolio_risk.portfolio_var, portfolio_risk.concentration_risk,
                portfolio_risk.correlation_risk, portfolio_risk.liquidity_risk,
                portfolio_risk.current_drawdown, portfolio_risk.risk_budget_used
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing portfolio risk: {e}")
    
    def get_risk_summary(self) -> Dict:
        """Get summary of current risk status"""
        return {
            "circuit_breakers_active": self.circuit_breakers_active,
            "active_alerts": len([a for a in self.alerts if not a.resolved]),
            "daily_loss_limit": self.daily_loss_limit,
            "max_portfolio_var": self.max_portfolio_var,
            "max_single_position": self.max_single_position,
            "correlation_limit": self.correlation_limit,
            "recent_alerts": [
                {
                    "type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp.isoformat()
                }
                for alert in self.alerts[-10:]  # Last 10 alerts
            ]
        }

# Example usage and testing
async def main():
    """Test the advanced risk manager"""
    risk_manager = AdvancedRiskManager()
    
    # Test position risk calculation
    position_risk = await risk_manager.calculate_position_risk("EXAMPLE", 1000, 0.001)
    print(f"Position Risk Level: {position_risk.risk_level}")
    print(f"VaR 95%: {position_risk.risk_metrics.var_95:.4f}")
    print(f"Liquidity Score: {position_risk.risk_metrics.liquidity_score:.2f}")
    
    # Test portfolio risk
    portfolio_risk = await risk_manager.calculate_portfolio_risk([position_risk])
    print(f"Portfolio VaR: {portfolio_risk.portfolio_var:.4f}")
    print(f"Risk Budget Used: {portfolio_risk.risk_budget_used:.2%}")
    
    # Test circuit breakers
    circuit_triggered = await risk_manager.check_circuit_breakers(portfolio_risk)
    print(f"Circuit Breakers: {'Active' if circuit_triggered else 'Normal'}")
    
    # Risk summary
    summary = risk_manager.get_risk_summary()
    print(f"Risk Summary: {summary}")

if __name__ == "__main__":
    asyncio.run(main())