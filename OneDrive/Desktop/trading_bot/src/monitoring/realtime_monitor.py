"""
Real-time Monitoring Dashboard for Microcap Trading
Provides rugpull detection, liquidity monitoring, and automated exit triggers.
"""

import asyncio
import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from typing import Dict, List, Optional, Tuple
import logging
import json
import threading
import time
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MonitoringAlert(Enum):
    RUGPULL_DETECTED = "rugpull_detected"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    PRICE_MANIPULATION = "price_manipulation"
    VOLUME_ANOMALY = "volume_anomaly"
    WALLET_CONCENTRATION = "wallet_concentration"
    SMART_MONEY_EXIT = "smart_money_exit"

@dataclass
class PositionHealth:
    """Health status for individual position"""
    symbol: str
    status: HealthStatus
    price: float
    change_24h: float
    volume_24h: float
    liquidity_score: float
    rugpull_score: float
    holder_concentration: float
    smart_money_activity: float
    price_stability: float
    alerts: List[str]
    last_updated: datetime

@dataclass
class RugpullIndicator:
    """Rugpull detection indicator"""
    symbol: str
    indicator_type: str
    severity: float  # 0.0 to 1.0
    description: str
    timestamp: datetime
    action_required: bool

class RealTimeMonitor:
    """
    Real-time monitoring system with:
    - Rugpull detection
    - Liquidity monitoring
    - Smart money tracking
    - Automated exit triggers
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.monitoring_active = False
        self.positions = {}
        self.alerts = []
        self.rugpull_indicators = []
        self.exit_triggers = {}
        
        # Monitoring thresholds
        self.rugpull_threshold = 0.7
        self.liquidity_threshold = 0.3
        self.price_drop_threshold = -0.5  # 50% drop
        self.volume_spike_threshold = 10.0  # 10x volume
        self.holder_concentration_threshold = 0.8  # 80% in few wallets
        
        self._init_database()
    
    def _init_database(self):
        """Initialize monitoring database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Position health table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    price REAL,
                    change_24h REAL,
                    volume_24h REAL,
                    liquidity_score REAL,
                    rugpull_score REAL,
                    holder_concentration REAL,
                    smart_money_activity REAL,
                    price_stability REAL,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Rugpull indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS rugpull_indicators (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    indicator_type TEXT NOT NULL,
                    severity REAL,
                    description TEXT,
                    action_required BOOLEAN,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Exit triggers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exit_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_value REAL,
                    current_value REAL,
                    triggered BOOLEAN DEFAULT FALSE,
                    timestamp DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing monitoring database: {e}")
    
    async def start_monitoring(self, symbols: List[str]):
        """Start real-time monitoring for specified symbols"""
        self.monitoring_active = True
        logger.info(f"Starting real-time monitoring for {len(symbols)} symbols")
        
        while self.monitoring_active:
            try:
                # Update position health for all symbols
                for symbol in symbols:
                    await self._update_position_health(symbol)
                
                # Check for rugpull indicators
                await self._check_rugpull_indicators(symbols)
                
                # Check exit triggers
                await self._check_exit_triggers(symbols)
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next update
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)  # Wait longer on error
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.monitoring_active = False
        logger.info("Real-time monitoring stopped")
    
    async def _update_position_health(self, symbol: str):
        """Update health status for a position"""
        try:
            # Get recent price data
            price_data = await self._get_recent_price_data(symbol)
            if not price_data:
                return
            
            current_price = price_data[-1]['price']
            prev_price = price_data[-2]['price'] if len(price_data) > 1 else current_price
            change_24h = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # Calculate metrics
            liquidity_score = await self._calculate_liquidity_score(symbol, price_data)
            rugpull_score = await self._calculate_rugpull_score(symbol, price_data)
            holder_concentration = await self._get_holder_concentration(symbol)
            smart_money_activity = await self._get_smart_money_activity(symbol)
            price_stability = await self._calculate_price_stability(price_data)
            
            # Determine health status
            status = self._determine_health_status(
                change_24h, liquidity_score, rugpull_score, 
                holder_concentration, smart_money_activity
            )
            
            # Generate alerts
            alerts = self._generate_health_alerts(
                symbol, change_24h, liquidity_score, rugpull_score,
                holder_concentration, smart_money_activity
            )
            
            # Create position health object
            position_health = PositionHealth(
                symbol=symbol,
                status=status,
                price=current_price,
                change_24h=change_24h,
                volume_24h=await self._get_24h_volume(symbol),
                liquidity_score=liquidity_score,
                rugpull_score=rugpull_score,
                holder_concentration=holder_concentration,
                smart_money_activity=smart_money_activity,
                price_stability=price_stability,
                alerts=alerts,
                last_updated=datetime.now()
            )
            
            self.positions[symbol] = position_health
            await self._store_position_health(position_health)
            
            # Log critical alerts
            if status in [HealthStatus.CRITICAL, HealthStatus.EMERGENCY]:
                logger.warning(f"Health Alert for {symbol}: {status.value} - {alerts}")
            
        except Exception as e:
            logger.error(f"Error updating position health for {symbol}: {e}")
    
    async def _get_recent_price_data(self, symbol: str) -> List[Dict]:
        """Get recent price data for symbol"""
        try:
            conn = sqlite3.connect(self.database_path)
            query = '''
                SELECT timestamp, close_price as price
                FROM price_data
                WHERE symbol = ? AND timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 144  -- 10-minute intervals for 24 hours
            '''
            cursor = conn.cursor()
            cursor.execute(query, (symbol,))
            rows = cursor.fetchall()
            conn.close()
            
            return [{'timestamp': row[0], 'price': row[1]} for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return []
    
    async def _calculate_liquidity_score(self, symbol: str, price_data: List[Dict]) -> float:
        """Calculate current liquidity score"""
        try:
            if len(price_data) < 10:
                return 0.5  # Default moderate liquidity
            
            prices = [d['price'] for d in price_data]
            
            # Price stability component
            price_changes = [abs((prices[i] - prices[i+1]) / prices[i+1]) 
                           for i in range(len(prices)-1)]
            avg_change = np.mean(price_changes)
            price_stability = max(0, 1.0 - (avg_change * 20))
            
            # Trading frequency component
            trading_frequency = min(1.0, len(price_data) / 144)  # Ideal: data every 10 min
            
            # Recent activity component
            recent_activity = 1.0 if len(price_data) >= 6 else len(price_data) / 6
            
            liquidity_score = (price_stability * 0.5 + 
                             trading_frequency * 0.3 + 
                             recent_activity * 0.2)
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception:
            return 0.5
    
    async def _calculate_rugpull_score(self, symbol: str, price_data: List[Dict]) -> float:
        """Calculate rugpull risk score"""
        try:
            if len(price_data) < 5:
                return 0.8  # High risk for new tokens
            
            prices = [d['price'] for d in price_data]
            rugpull_score = 0.0
            
            # Sudden price drops
            for i in range(len(prices)-1):
                drop = (prices[i] - prices[i+1]) / prices[i+1]
                if drop < -0.5:  # >50% drop
                    rugpull_score += 0.3
                elif drop < -0.3:  # >30% drop
                    rugpull_score += 0.2
            
            # Extreme volatility
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            volatility = np.std(returns) if len(returns) > 0 else 0
            if volatility > 1.0:  # >100% volatility
                rugpull_score += 0.4
            elif volatility > 0.5:  # >50% volatility
                rugpull_score += 0.2
            
            # Price manipulation patterns
            max_price = max(prices)
            min_price = min(prices)
            if max_price > 0 and (max_price - min_price) / min_price > 5.0:  # >500% range
                rugpull_score += 0.3
            
            return min(1.0, rugpull_score)
            
        except Exception:
            return 0.5
    
    async def _get_holder_concentration(self, symbol: str) -> float:
        """Get wallet holder concentration (simulated)"""
        # In real implementation, this would query blockchain data
        # For now, return simulated data based on symbol characteristics
        try:
            # Simulate holder concentration based on symbol
            hash_value = hash(symbol) % 100
            concentration = 0.3 + (hash_value / 100) * 0.6  # 30% to 90%
            return concentration
        except Exception:
            return 0.5
    
    async def _get_smart_money_activity(self, symbol: str) -> float:
        """Get smart money activity indicator (simulated)"""
        # In real implementation, this would track large wallet movements
        try:
            # Simulate smart money activity
            recent_activity = np.random.uniform(0.0, 1.0)
            return recent_activity
        except Exception:
            return 0.5
    
    async def _calculate_price_stability(self, price_data: List[Dict]) -> float:
        """Calculate price stability score"""
        try:
            if len(price_data) < 5:
                return 0.5
            
            prices = [d['price'] for d in price_data]
            returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
            
            # Lower standard deviation = higher stability
            volatility = np.std(returns) if len(returns) > 0 else 1.0
            stability = max(0.0, 1.0 - min(1.0, volatility * 2))
            
            return stability
            
        except Exception:
            return 0.5
    
    async def _get_24h_volume(self, symbol: str) -> float:
        """Get 24-hour trading volume"""
        try:
            # Simulate volume data
            return np.random.uniform(10000, 1000000)
        except Exception:
            return 0.0
    
    def _determine_health_status(self, change_24h: float, liquidity_score: float,
                               rugpull_score: float, holder_concentration: float,
                               smart_money_activity: float) -> HealthStatus:
        """Determine overall health status"""
        risk_factors = 0
        
        # Price drop factor
        if change_24h < -0.5:  # >50% drop
            risk_factors += 3
        elif change_24h < -0.3:  # >30% drop
            risk_factors += 2
        elif change_24h < -0.15:  # >15% drop
            risk_factors += 1
        
        # Liquidity factor
        if liquidity_score < 0.2:
            risk_factors += 3
        elif liquidity_score < 0.4:
            risk_factors += 2
        elif liquidity_score < 0.6:
            risk_factors += 1
        
        # Rugpull factor
        if rugpull_score > 0.8:
            risk_factors += 3
        elif rugpull_score > 0.6:
            risk_factors += 2
        elif rugpull_score > 0.4:
            risk_factors += 1
        
        # Holder concentration factor
        if holder_concentration > 0.9:
            risk_factors += 2
        elif holder_concentration > 0.8:
            risk_factors += 1
        
        # Smart money exit factor
        if smart_money_activity > 0.8:  # High smart money exit
            risk_factors += 2
        elif smart_money_activity > 0.6:
            risk_factors += 1
        
        # Determine status based on risk factors
        if risk_factors >= 8:
            return HealthStatus.EMERGENCY
        elif risk_factors >= 5:
            return HealthStatus.CRITICAL
        elif risk_factors >= 3:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _generate_health_alerts(self, symbol: str, change_24h: float,
                              liquidity_score: float, rugpull_score: float,
                              holder_concentration: float, smart_money_activity: float) -> List[str]:
        """Generate health alerts for position"""
        alerts = []
        
        if change_24h < -0.5:
            alerts.append(f"Severe price drop: {change_24h:.1%}")
        
        if liquidity_score < 0.3:
            alerts.append(f"Low liquidity: {liquidity_score:.2f}")
        
        if rugpull_score > 0.7:
            alerts.append(f"High rugpull risk: {rugpull_score:.2f}")
        
        if holder_concentration > 0.8:
            alerts.append(f"High wallet concentration: {holder_concentration:.1%}")
        
        if smart_money_activity > 0.7:
            alerts.append(f"Smart money exit detected: {smart_money_activity:.2f}")
        
        return alerts
    
    async def _check_rugpull_indicators(self, symbols: List[str]):
        """Check for rugpull indicators across all positions"""
        try:
            for symbol in symbols:
                if symbol not in self.positions:
                    continue
                
                position = self.positions[symbol]
                indicators = []
                
                # Severe price drop indicator
                if position.change_24h < self.price_drop_threshold:
                    indicators.append(RugpullIndicator(
                        symbol=symbol,
                        indicator_type="severe_price_drop",
                        severity=min(1.0, abs(position.change_24h) / 0.5),
                        description=f"Price dropped {position.change_24h:.1%} in 24h",
                        timestamp=datetime.now(),
                        action_required=True
                    ))
                
                # Liquidity crisis indicator
                if position.liquidity_score < self.liquidity_threshold:
                    indicators.append(RugpullIndicator(
                        symbol=symbol,
                        indicator_type="liquidity_crisis",
                        severity=1.0 - position.liquidity_score,
                        description=f"Liquidity score critically low: {position.liquidity_score:.2f}",
                        timestamp=datetime.now(),
                        action_required=True
                    ))
                
                # High rugpull score indicator
                if position.rugpull_score > self.rugpull_threshold:
                    indicators.append(RugpullIndicator(
                        symbol=symbol,
                        indicator_type="rugpull_pattern",
                        severity=position.rugpull_score,
                        description=f"Rugpull pattern detected: {position.rugpull_score:.2f}",
                        timestamp=datetime.now(),
                        action_required=True
                    ))
                
                # Store indicators
                for indicator in indicators:
                    self.rugpull_indicators.append(indicator)
                    await self._store_rugpull_indicator(indicator)
                    
                    if indicator.action_required:
                        logger.critical(f"Rugpull Alert for {symbol}: {indicator.description}")
            
        except Exception as e:
            logger.error(f"Error checking rugpull indicators: {e}")
    
    async def _check_exit_triggers(self, symbols: List[str]):
        """Check automated exit triggers"""
        try:
            for symbol in symbols:
                if symbol not in self.positions:
                    continue
                
                position = self.positions[symbol]
                
                # Emergency exit triggers
                emergency_exit = (
                    position.status == HealthStatus.EMERGENCY or
                    position.rugpull_score > 0.9 or
                    position.change_24h < -0.7 or
                    position.liquidity_score < 0.1
                )
                
                if emergency_exit:
                    await self._trigger_emergency_exit(symbol, position)
                
                # Warning level triggers
                elif position.status == HealthStatus.CRITICAL:
                    await self._trigger_warning_exit(symbol, position)
            
        except Exception as e:
            logger.error(f"Error checking exit triggers: {e}")
    
    async def _trigger_emergency_exit(self, symbol: str, position: PositionHealth):
        """Trigger emergency exit for position"""
        try:
            logger.critical(f"EMERGENCY EXIT triggered for {symbol}")
            
            # Record exit trigger
            await self._store_exit_trigger(
                symbol, "emergency_exit", 0.9, position.rugpull_score, True
            )
            
            # In real implementation, would execute immediate market sell
            # For now, just log the action
            logger.critical(f"Emergency exit executed for {symbol} at {position.price}")
            
        except Exception as e:
            logger.error(f"Error executing emergency exit for {symbol}: {e}")
    
    async def _trigger_warning_exit(self, symbol: str, position: PositionHealth):
        """Trigger warning-level exit for position"""
        try:
            logger.warning(f"WARNING EXIT triggered for {symbol}")
            
            # Record exit trigger
            await self._store_exit_trigger(
                symbol, "warning_exit", 0.7, position.rugpull_score, True
            )
            
            # In real implementation, would execute partial or staged sell
            logger.warning(f"Warning exit executed for {symbol} at {position.price}")
            
        except Exception as e:
            logger.error(f"Error executing warning exit for {symbol}: {e}")
    
    async def _store_position_health(self, health: PositionHealth):
        """Store position health in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO position_health (
                    symbol, status, price, change_24h, volume_24h, liquidity_score,
                    rugpull_score, holder_concentration, smart_money_activity,
                    price_stability, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                health.symbol, health.status.value, health.price, health.change_24h,
                health.volume_24h, health.liquidity_score, health.rugpull_score,
                health.holder_concentration, health.smart_money_activity,
                health.price_stability, health.last_updated
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing position health: {e}")
    
    async def _store_rugpull_indicator(self, indicator: RugpullIndicator):
        """Store rugpull indicator in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO rugpull_indicators (
                    symbol, indicator_type, severity, description,
                    action_required, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                indicator.symbol, indicator.indicator_type, indicator.severity,
                indicator.description, indicator.action_required, indicator.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing rugpull indicator: {e}")
    
    async def _store_exit_trigger(self, symbol: str, trigger_type: str,
                                trigger_value: float, current_value: float, triggered: bool):
        """Store exit trigger in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO exit_triggers (
                    symbol, trigger_type, trigger_value, current_value, triggered, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, trigger_type, trigger_value, current_value, triggered, datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing exit trigger: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Keep only last 7 days of position health data
            cursor.execute('''
                DELETE FROM position_health
                WHERE created_at < datetime('now', '-7 days')
            ''')
            
            # Keep only last 30 days of indicators
            cursor.execute('''
                DELETE FROM rugpull_indicators
                WHERE created_at < datetime('now', '-30 days')
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
    
    def get_monitoring_summary(self) -> Dict:
        """Get summary of current monitoring status"""
        if not self.positions:
            return {"status": "No positions monitored", "positions": 0}
        
        status_counts = {}
        total_alerts = 0
        critical_positions = []
        
        for symbol, position in self.positions.items():
            status = position.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_alerts += len(position.alerts)
            
            if position.status in [HealthStatus.CRITICAL, HealthStatus.EMERGENCY]:
                critical_positions.append({
                    "symbol": symbol,
                    "status": status,
                    "rugpull_score": position.rugpull_score,
                    "change_24h": position.change_24h,
                    "alerts": position.alerts
                })
        
        return {
            "monitoring_active": self.monitoring_active,
            "total_positions": len(self.positions),
            "status_counts": status_counts,
            "total_alerts": total_alerts,
            "critical_positions": critical_positions,
            "recent_rugpull_indicators": len([
                i for i in self.rugpull_indicators 
                if i.timestamp > datetime.now() - timedelta(hours=1)
            ])
        }

class MonitoringDashboard:
    """
    GUI Dashboard for real-time monitoring
    """
    
    def __init__(self, root: tk.Tk, monitor: RealTimeMonitor):
        self.root = root
        self.monitor = monitor
        self.update_interval = 5000  # 5 seconds
        
        self._setup_gui()
        self._start_updates()
    
    def _setup_gui(self):
        """Setup monitoring dashboard GUI"""
        self.root.title("Real-time Microcap Monitoring Dashboard")
        self.root.geometry("1200x800")
        
        # Main notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Position Health tab
        health_frame = ttk.Frame(notebook)
        notebook.add(health_frame, text="Position Health")
        self._setup_health_tab(health_frame)
        
        # Rugpull Detection tab
        rugpull_frame = ttk.Frame(notebook)
        notebook.add(rugpull_frame, text="Rugpull Detection")
        self._setup_rugpull_tab(rugpull_frame)
        
        # Exit Triggers tab
        triggers_frame = ttk.Frame(notebook)
        notebook.add(triggers_frame, text="Exit Triggers")
        self._setup_triggers_tab(triggers_frame)
        
        # Analytics tab
        analytics_frame = ttk.Frame(notebook)
        notebook.add(analytics_frame, text="Analytics")
        self._setup_analytics_tab(analytics_frame)
    
    def _setup_health_tab(self, parent):
        """Setup position health monitoring tab"""
        # Health status summary
        summary_frame = ttk.LabelFrame(parent, text="Health Summary")
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.health_summary_label = ttk.Label(summary_frame, text="No data available")
        self.health_summary_label.pack(pady=10)
        
        # Positions table
        positions_frame = ttk.LabelFrame(parent, text="Position Details")
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create treeview for positions
        columns = ("Symbol", "Status", "Price", "24h Change", "Liquidity", "Rugpull Score", "Alerts")
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show="headings")
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=120)
        
        # Scrollbar for treeview
        positions_scrollbar = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, 
                                          command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scrollbar.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_rugpull_tab(self, parent):
        """Setup rugpull detection tab"""
        # Rugpull indicators
        indicators_frame = ttk.LabelFrame(parent, text="Rugpull Indicators")
        indicators_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget for indicators
        self.rugpull_text = tk.Text(indicators_frame, wrap=tk.WORD)
        rugpull_scrollbar = ttk.Scrollbar(indicators_frame, orient=tk.VERTICAL,
                                        command=self.rugpull_text.yview)
        self.rugpull_text.configure(yscrollcommand=rugpull_scrollbar.set)
        
        self.rugpull_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        rugpull_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_triggers_tab(self, parent):
        """Setup exit triggers tab"""
        # Exit triggers status
        triggers_frame = ttk.LabelFrame(parent, text="Active Exit Triggers")
        triggers_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.triggers_text = tk.Text(triggers_frame, wrap=tk.WORD)
        triggers_scrollbar = ttk.Scrollbar(triggers_frame, orient=tk.VERTICAL,
                                         command=self.triggers_text.yview)
        self.triggers_text.configure(yscrollcommand=triggers_scrollbar.set)
        
        self.triggers_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        triggers_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_analytics_tab(self, parent):
        """Setup analytics and charts tab"""
        # Placeholder for future analytics
        analytics_label = ttk.Label(parent, text="Analytics charts will be added here")
        analytics_label.pack(pady=20)
    
    def _start_updates(self):
        """Start periodic GUI updates"""
        self._update_display()
        self.root.after(self.update_interval, self._start_updates)
    
    def _update_display(self):
        """Update all dashboard displays"""
        try:
            # Update health summary
            summary = self.monitor.get_monitoring_summary()
            summary_text = f"Monitoring: {'Active' if summary['monitoring_active'] else 'Inactive'}\n"
            summary_text += f"Positions: {summary['total_positions']}\n"
            summary_text += f"Total Alerts: {summary['total_alerts']}\n"
            
            if 'status_counts' in summary:
                for status, count in summary['status_counts'].items():
                    summary_text += f"{status.title()}: {count} "
            
            self.health_summary_label.config(text=summary_text)
            
            # Update positions table
            self._update_positions_table()
            
            # Update rugpull indicators
            self._update_rugpull_display()
            
            # Update exit triggers
            self._update_triggers_display()
            
        except Exception as e:
            logger.error(f"Error updating dashboard display: {e}")
    
    def _update_positions_table(self):
        """Update positions table"""
        try:
            # Clear existing items
            for item in self.positions_tree.get_children():
                self.positions_tree.delete(item)
            
            # Add current positions
            for symbol, position in self.monitor.positions.items():
                alerts_text = f"{len(position.alerts)} alerts" if position.alerts else "No alerts"
                
                item = self.positions_tree.insert("", tk.END, values=(
                    symbol,
                    position.status.value.title(),
                    f"${position.price:.6f}",
                    f"{position.change_24h:.1%}",
                    f"{position.liquidity_score:.2f}",
                    f"{position.rugpull_score:.2f}",
                    alerts_text
                ))
                
                # Color code by health status
                if position.status == HealthStatus.EMERGENCY:
                    self.positions_tree.item(item, tags=("emergency",))
                elif position.status == HealthStatus.CRITICAL:
                    self.positions_tree.item(item, tags=("critical",))
                elif position.status == HealthStatus.WARNING:
                    self.positions_tree.item(item, tags=("warning",))
            
            # Configure tags for colors
            self.positions_tree.tag_configure("emergency", background="#ff4444")
            self.positions_tree.tag_configure("critical", background="#ff8844")
            self.positions_tree.tag_configure("warning", background="#ffaa44")
            
        except Exception as e:
            logger.error(f"Error updating positions table: {e}")
    
    def _update_rugpull_display(self):
        """Update rugpull indicators display"""
        try:
            self.rugpull_text.delete(1.0, tk.END)
            
            recent_indicators = [
                i for i in self.monitor.rugpull_indicators
                if i.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            if not recent_indicators:
                self.rugpull_text.insert(tk.END, "No recent rugpull indicators detected.")
                return
            
            for indicator in recent_indicators[-20:]:  # Show last 20
                timestamp = indicator.timestamp.strftime("%H:%M:%S")
                severity_text = f"{indicator.severity:.1%}"
                
                text = (f"[{timestamp}] {indicator.symbol} - {indicator.indicator_type}\n"
                       f"Severity: {severity_text} | {indicator.description}\n"
                       f"Action Required: {'Yes' if indicator.action_required else 'No'}\n\n")
                
                self.rugpull_text.insert(tk.END, text)
            
            # Scroll to bottom
            self.rugpull_text.see(tk.END)
            
        except Exception as e:
            logger.error(f"Error updating rugpull display: {e}")
    
    def _update_triggers_display(self):
        """Update exit triggers display"""
        try:
            self.triggers_text.delete(1.0, tk.END)
            
            # Show active exit triggers (in real implementation, would query database)
            trigger_text = "Exit triggers will be displayed here when activated.\n\n"
            trigger_text += "Monitoring thresholds:\n"
            trigger_text += f"- Rugpull threshold: {self.monitor.rugpull_threshold:.1%}\n"
            trigger_text += f"- Liquidity threshold: {self.monitor.liquidity_threshold:.1%}\n"
            trigger_text += f"- Price drop threshold: {self.monitor.price_drop_threshold:.1%}\n"
            
            self.triggers_text.insert(tk.END, trigger_text)
            
        except Exception as e:
            logger.error(f"Error updating triggers display: {e}")

# Example usage
async def main():
    """Test the monitoring system"""
    monitor = RealTimeMonitor()
    
    # Test symbols
    test_symbols = ["TEST1", "TEST2", "RISKYCOIN"]
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitor.start_monitoring(test_symbols))
    
    # Let it run for a bit
    await asyncio.sleep(30)
    
    # Get summary
    summary = monitor.get_monitoring_summary()
    print("Monitoring Summary:", summary)
    
    # Stop monitoring
    monitor.stop_monitoring()
    await monitoring_task

def run_gui():
    """Run the monitoring dashboard GUI"""
    root = tk.Tk()
    monitor = RealTimeMonitor()
    dashboard = MonitoringDashboard(root, monitor)
    
    # Start monitoring in background thread
    def start_monitoring():
        asyncio.run(monitor.start_monitoring(["BTC", "ETH", "SOL"]))
    
    monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
    monitoring_thread.start()
    
    root.mainloop()

if __name__ == "__main__":
    # Run GUI version
    run_gui()
    
    # Or run CLI test
    # asyncio.run(main())