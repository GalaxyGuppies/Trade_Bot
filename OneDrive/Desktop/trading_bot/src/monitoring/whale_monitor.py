"""
Whale Wallet Monitoring System
Advanced whale tracking to monitor large holder movements, detect insider trading patterns,
and provide early exit signals for microcap trading.
"""

import asyncio
import logging
import requests
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhaleActivity(Enum):
    ACCUMULATING = "accumulating"
    DISTRIBUTING = "distributing"
    HOLDING = "holding"
    DUMPING = "dumping"
    INACTIVE = "inactive"

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class WhaleWallet:
    """Whale wallet information"""
    address: str
    label: str
    balance: float
    percentage_holdings: float
    first_seen: datetime
    last_activity: datetime
    total_transactions: int
    avg_transaction_size: float
    profit_loss: float
    wallet_age_days: int
    smart_money_score: float
    risk_level: str

@dataclass
class WhaleTransaction:
    """Individual whale transaction"""
    tx_hash: str
    whale_address: str
    token_address: str
    transaction_type: str  # buy, sell, transfer
    amount: float
    value_usd: float
    timestamp: datetime
    gas_price: float
    block_number: int
    from_address: str
    to_address: str

@dataclass
class WhaleAlert:
    """Whale activity alert"""
    id: str
    whale_address: str
    token_address: str
    alert_type: str
    alert_level: AlertLevel
    amount: float
    value_usd: float
    percentage_change: float
    message: str
    timestamp: datetime
    action_recommended: str

@dataclass
class InsiderPattern:
    """Detected insider trading pattern"""
    pattern_id: str
    token_address: str
    wallets_involved: List[str]
    pattern_type: str
    confidence_score: float
    first_detected: datetime
    last_activity: datetime
    total_volume: float
    description: str
    risk_assessment: str

class WhaleMonitor:
    """
    Advanced whale wallet monitoring system for microcap tokens
    Tracks large holders, detects patterns, and provides early signals
    """
    
    def __init__(self, database_path: str = "trading_bot.db"):
        self.database_path = database_path
        self.monitoring_active = False
        self.whale_wallets = {}
        self.transaction_history = defaultdict(deque)
        self.alerts = []
        self.insider_patterns = []
        
        # Whale thresholds
        self.min_whale_threshold = 0.01  # 1% of supply
        self.large_whale_threshold = 0.05  # 5% of supply
        self.mega_whale_threshold = 0.10  # 10% of supply
        
        # Alert thresholds
        self.alert_amount_threshold = 1000  # $1000 transaction
        self.alert_percentage_threshold = 0.005  # 0.5% of supply
        self.whale_dump_threshold = 0.02  # 2% of holdings in single tx
        
        # Smart money detection parameters
        self.profit_threshold = 0.20  # 20% profit for smart money classification
        self.win_rate_threshold = 0.70  # 70% win rate
        self.min_transactions = 10  # Minimum transactions for analysis
        
        # Pattern detection windows
        self.pattern_window_hours = 24
        self.coordination_window_minutes = 30
        
        self._init_database()
        logger.info("🐋 Whale Monitor initialized")
    
    def _init_database(self):
        """Initialize whale monitoring database tables"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Whale wallets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whale_wallets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    address TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    label TEXT,
                    balance REAL,
                    percentage_holdings REAL,
                    first_seen DATETIME,
                    last_activity DATETIME,
                    total_transactions INTEGER,
                    avg_transaction_size REAL,
                    profit_loss REAL,
                    wallet_age_days INTEGER,
                    smart_money_score REAL,
                    risk_level TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(address, token_address)
                )
            ''')
            
            # Whale transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whale_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tx_hash TEXT NOT NULL UNIQUE,
                    whale_address TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    transaction_type TEXT,
                    amount REAL,
                    value_usd REAL,
                    timestamp DATETIME,
                    gas_price REAL,
                    block_number INTEGER,
                    from_address TEXT,
                    to_address TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Whale alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS whale_alerts (
                    id TEXT PRIMARY KEY,
                    whale_address TEXT NOT NULL,
                    token_address TEXT NOT NULL,
                    alert_type TEXT,
                    alert_level TEXT,
                    amount REAL,
                    value_usd REAL,
                    percentage_change REAL,
                    message TEXT,
                    timestamp DATETIME,
                    action_recommended TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insider patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS insider_patterns (
                    id TEXT PRIMARY KEY,
                    token_address TEXT NOT NULL,
                    pattern_type TEXT,
                    confidence_score REAL,
                    first_detected DATETIME,
                    last_activity DATETIME,
                    total_volume REAL,
                    description TEXT,
                    risk_assessment TEXT,
                    wallets_involved TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Whale monitoring database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing whale database: {e}")
    
    async def start_monitoring(self, token_addresses: List[str]):
        """Start monitoring whale activity for given tokens"""
        self.monitoring_active = True
        logger.info(f"🐋 Starting whale monitoring for {len(token_addresses)} tokens")
        
        while self.monitoring_active:
            try:
                for token_address in token_addresses:
                    await self._scan_whale_activity(token_address)
                    await asyncio.sleep(2)  # Rate limiting
                
                # Analyze patterns every cycle
                await self._analyze_insider_patterns(token_addresses)
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # 30-second monitoring cycle
                
            except Exception as e:
                logger.error(f"Error in whale monitoring loop: {e}")
                await asyncio.sleep(60)
    
    def stop_monitoring(self):
        """Stop whale monitoring"""
        self.monitoring_active = False
        logger.info("🐋 Whale monitoring stopped")
    
    async def _scan_whale_activity(self, token_address: str):
        """Scan for whale activity on a specific token"""
        try:
            # Get current whale holders
            whale_holders = await self._get_whale_holders(token_address)
            
            # Update whale wallet information
            for whale_info in whale_holders:
                await self._update_whale_wallet(whale_info, token_address)
            
            # Get recent transactions
            recent_transactions = await self._get_recent_transactions(token_address)
            
            # Process transactions for whale activity
            for tx in recent_transactions:
                if await self._is_whale_transaction(tx, token_address):
                    await self._process_whale_transaction(tx, token_address)
            
        except Exception as e:
            logger.error(f"Error scanning whale activity for {token_address}: {e}")
    
    async def _get_whale_holders(self, token_address: str) -> List[Dict]:
        """Get current whale holders for a token"""
        try:
            # In real implementation, would query blockchain for top holders
            # For demo, simulate whale holder data
            whale_holders = [
                {
                    "address": "0x1234567890abcdef1234567890abcdef12345678",
                    "balance": 1000000,
                    "percentage": 5.5,
                    "label": "Unknown Whale #1"
                },
                {
                    "address": "0xabcdef1234567890abcdef1234567890abcdef12",
                    "balance": 750000,
                    "percentage": 3.8,
                    "label": "Potential DEV wallet"
                },
                {
                    "address": "0x9876543210fedcba9876543210fedcba98765432",
                    "balance": 500000,
                    "percentage": 2.1,
                    "label": "Smart Money Wallet"
                }
            ]
            
            return whale_holders
            
        except Exception as e:
            logger.error(f"Error getting whale holders: {e}")
            return []
    
    async def _get_recent_transactions(self, token_address: str) -> List[Dict]:
        """Get recent transactions for a token"""
        try:
            # In real implementation, would query blockchain for recent transactions
            # For demo, simulate transaction data
            recent_transactions = [
                {
                    "hash": f"0x{'a' * 40}{int(time.time())}",
                    "from": "0x1234567890abcdef1234567890abcdef12345678",
                    "to": "0xDEF171Fe48CF0115B1d80b88dc8eAB59176FEe57",  # PancakeSwap
                    "amount": 50000,
                    "value_usd": 2500,
                    "timestamp": datetime.now(),
                    "gas_price": 5,
                    "block_number": 12345678
                }
            ]
            
            return recent_transactions
            
        except Exception as e:
            logger.error(f"Error getting recent transactions: {e}")
            return []
    
    async def _is_whale_transaction(self, transaction: Dict, token_address: str) -> bool:
        """Determine if transaction involves a whale"""
        try:
            amount = transaction.get("amount", 0)
            from_address = transaction.get("from", "")
            to_address = transaction.get("to", "")
            
            # Check if transaction amount exceeds thresholds
            if transaction.get("value_usd", 0) < self.alert_amount_threshold:
                return False
            
            # Check if either address is a known whale
            whale_addresses = [whale.address for whale in self.whale_wallets.values()]
            
            return from_address in whale_addresses or to_address in whale_addresses
            
        except Exception as e:
            logger.error(f"Error checking whale transaction: {e}")
            return False
    
    async def _update_whale_wallet(self, whale_info: Dict, token_address: str):
        """Update whale wallet information"""
        try:
            address = whale_info["address"]
            
            # Calculate smart money score
            smart_money_score = await self._calculate_smart_money_score(address)
            
            # Determine risk level
            risk_level = self._assess_whale_risk_level(whale_info["percentage"], smart_money_score)
            
            whale_wallet = WhaleWallet(
                address=address,
                label=whale_info.get("label", "Unknown Whale"),
                balance=whale_info["balance"],
                percentage_holdings=whale_info["percentage"],
                first_seen=datetime.now(),  # Would track actual first seen
                last_activity=datetime.now(),
                total_transactions=0,  # Would count from blockchain
                avg_transaction_size=0.0,  # Would calculate from history
                profit_loss=0.0,  # Would calculate based on buy/sell history
                wallet_age_days=365,  # Would calculate from first transaction
                smart_money_score=smart_money_score,
                risk_level=risk_level
            )
            
            self.whale_wallets[f"{token_address}_{address}"] = whale_wallet
            await self._store_whale_wallet(whale_wallet, token_address)
            
        except Exception as e:
            logger.error(f"Error updating whale wallet: {e}")
    
    async def _process_whale_transaction(self, transaction: Dict, token_address: str):
        """Process a whale transaction and generate alerts if needed"""
        try:
            whale_address = transaction.get("from", "")
            amount = transaction.get("amount", 0)
            value_usd = transaction.get("value_usd", 0)
            
            # Determine transaction type
            tx_type = self._determine_transaction_type(transaction)
            
            # Create whale transaction record
            whale_tx = WhaleTransaction(
                tx_hash=transaction["hash"],
                whale_address=whale_address,
                token_address=token_address,
                transaction_type=tx_type,
                amount=amount,
                value_usd=value_usd,
                timestamp=transaction["timestamp"],
                gas_price=transaction.get("gas_price", 0),
                block_number=transaction.get("block_number", 0),
                from_address=transaction.get("from", ""),
                to_address=transaction.get("to", "")
            )
            
            # Store transaction
            await self._store_whale_transaction(whale_tx)
            
            # Check if alert should be generated
            whale_key = f"{token_address}_{whale_address}"
            if whale_key in self.whale_wallets:
                whale = self.whale_wallets[whale_key]
                percentage_change = (amount / whale.balance) * 100 if whale.balance > 0 else 0
                
                if await self._should_generate_alert(whale_tx, whale, percentage_change):
                    await self._generate_whale_alert(whale_tx, whale, percentage_change)
            
            # Add to transaction history for pattern analysis
            self.transaction_history[token_address].append(whale_tx)
            
            # Keep only recent transactions (last 1000)
            if len(self.transaction_history[token_address]) > 1000:
                self.transaction_history[token_address].popleft()
            
        except Exception as e:
            logger.error(f"Error processing whale transaction: {e}")
    
    def _determine_transaction_type(self, transaction: Dict) -> str:
        """Determine if transaction is buy, sell, or transfer"""
        to_address = transaction.get("to", "").lower()
        from_address = transaction.get("from", "").lower()
        
        # Known DEX addresses (simplified)
        dex_addresses = [
            "0xdef171fe48cf0115b1d80b88dc8eab59176fee57",  # PancakeSwap
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
            "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3
        ]
        
        if to_address in dex_addresses:
            return "sell"
        elif from_address in dex_addresses:
            return "buy"
        else:
            return "transfer"
    
    async def _calculate_smart_money_score(self, address: str) -> float:
        """Calculate smart money score for a wallet"""
        try:
            # In real implementation, would analyze historical performance
            # For demo, simulate smart money scoring
            
            # Factors: profit ratio, win rate, timing of entries/exits, etc.
            profit_factor = 0.8  # Simulated 80% profit ratio
            win_rate_factor = 0.75  # Simulated 75% win rate
            timing_factor = 0.6  # Simulated good timing
            consistency_factor = 0.7  # Simulated consistency
            
            smart_money_score = (
                profit_factor * 0.3 +
                win_rate_factor * 0.3 +
                timing_factor * 0.2 +
                consistency_factor * 0.2
            )
            
            return min(1.0, max(0.0, smart_money_score))
            
        except Exception as e:
            logger.error(f"Error calculating smart money score: {e}")
            return 0.5
    
    def _assess_whale_risk_level(self, percentage_holdings: float, smart_money_score: float) -> str:
        """Assess risk level of a whale"""
        if percentage_holdings > self.mega_whale_threshold:
            if smart_money_score < 0.3:
                return "CRITICAL"  # Large hostile whale
            else:
                return "HIGH"  # Large but potentially smart whale
        elif percentage_holdings > self.large_whale_threshold:
            if smart_money_score < 0.4:
                return "HIGH"
            else:
                return "MEDIUM"
        else:
            return "LOW"
    
    async def _should_generate_alert(self, transaction: WhaleTransaction, 
                                   whale: WhaleWallet, percentage_change: float) -> bool:
        """Determine if whale transaction should generate an alert"""
        try:
            # Large transaction threshold
            if transaction.value_usd > self.alert_amount_threshold * 5:  # $5000+
                return True
            
            # Large percentage of holdings
            if percentage_change > self.whale_dump_threshold * 100:  # 2%+ of holdings
                return True
            
            # High risk whale activity
            if whale.risk_level in ["HIGH", "CRITICAL"] and transaction.value_usd > 1000:
                return True
            
            # Smart money activity
            if whale.smart_money_score > 0.8 and transaction.transaction_type == "sell":
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error determining alert necessity: {e}")
            return False
    
    async def _generate_whale_alert(self, transaction: WhaleTransaction, 
                                  whale: WhaleWallet, percentage_change: float):
        """Generate whale activity alert"""
        try:
            alert_id = f"whale_{transaction.tx_hash}_{int(time.time())}"
            
            # Determine alert level
            if whale.risk_level == "CRITICAL" or percentage_change > 5:
                alert_level = AlertLevel.EMERGENCY
            elif whale.risk_level == "HIGH" or percentage_change > 2:
                alert_level = AlertLevel.CRITICAL
            elif transaction.value_usd > 5000 or percentage_change > 1:
                alert_level = AlertLevel.WARNING
            else:
                alert_level = AlertLevel.INFO
            
            # Generate message
            action = transaction.transaction_type.upper()
            message = f"🐋 WHALE {action}: {whale.label} {action.lower()}s ${transaction.value_usd:,.0f} ({percentage_change:.1f}% of holdings)"
            
            # Recommend action
            if transaction.transaction_type == "sell" and alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                action_recommended = "CONSIDER SELLING - Large whale dumping detected"
            elif transaction.transaction_type == "buy" and whale.smart_money_score > 0.8:
                action_recommended = "CONSIDER BUYING - Smart money accumulating"
            else:
                action_recommended = "MONITOR - Continue watching whale activity"
            
            alert = WhaleAlert(
                id=alert_id,
                whale_address=whale.address,
                token_address=transaction.token_address,
                alert_type=f"whale_{transaction.transaction_type}",
                alert_level=alert_level,
                amount=transaction.amount,
                value_usd=transaction.value_usd,
                percentage_change=percentage_change,
                message=message,
                timestamp=datetime.now(),
                action_recommended=action_recommended
            )
            
            self.alerts.append(alert)
            await self._store_whale_alert(alert)
            
            logger.warning(f"🚨 {alert_level.value.upper()} WHALE ALERT: {message}")
            
        except Exception as e:
            logger.error(f"Error generating whale alert: {e}")
    
    async def _analyze_insider_patterns(self, token_addresses: List[str]):
        """Analyze for insider trading patterns"""
        try:
            for token_address in token_addresses:
                recent_transactions = list(self.transaction_history[token_address])
                
                if len(recent_transactions) < 5:
                    continue
                
                # Look for coordinated activity
                coordinated_pattern = await self._detect_coordinated_activity(recent_transactions)
                if coordinated_pattern:
                    await self._store_insider_pattern(coordinated_pattern)
                
                # Look for pre-announcement accumulation
                accumulation_pattern = await self._detect_pre_announcement_accumulation(recent_transactions)
                if accumulation_pattern:
                    await self._store_insider_pattern(accumulation_pattern)
                
                # Look for suspicious timing patterns
                timing_pattern = await self._detect_suspicious_timing(recent_transactions)
                if timing_pattern:
                    await self._store_insider_pattern(timing_pattern)
            
        except Exception as e:
            logger.error(f"Error analyzing insider patterns: {e}")
    
    async def _detect_coordinated_activity(self, transactions: List[WhaleTransaction]) -> Optional[InsiderPattern]:
        """Detect coordinated whale activity"""
        try:
            # Group transactions by time windows
            time_windows = {}
            window_size = timedelta(minutes=self.coordination_window_minutes)
            
            for tx in transactions[-50:]:  # Analyze last 50 transactions
                window_start = tx.timestamp.replace(minute=(tx.timestamp.minute // 30) * 30, second=0, microsecond=0)
                window_key = window_start.isoformat()
                
                if window_key not in time_windows:
                    time_windows[window_key] = []
                time_windows[window_key].append(tx)
            
            # Look for windows with multiple large transactions
            for window_key, window_txs in time_windows.items():
                if len(window_txs) >= 3:  # 3+ transactions in window
                    total_volume = sum(tx.value_usd for tx in window_txs)
                    unique_wallets = len(set(tx.whale_address for tx in window_txs))
                    
                    if total_volume > 10000 and unique_wallets >= 2:  # Coordinated activity
                        pattern = InsiderPattern(
                            pattern_id=f"coord_{window_key}_{int(time.time())}",
                            token_address=window_txs[0].token_address,
                            wallets_involved=[tx.whale_address for tx in window_txs],
                            pattern_type="coordinated_activity",
                            confidence_score=0.7,
                            first_detected=datetime.now(),
                            last_activity=max(tx.timestamp for tx in window_txs),
                            total_volume=total_volume,
                            description=f"Coordinated activity: {unique_wallets} wallets, ${total_volume:,.0f} volume",
                            risk_assessment="HIGH - Potential manipulation"
                        )
                        return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting coordinated activity: {e}")
            return None
    
    async def _detect_pre_announcement_accumulation(self, transactions: List[WhaleTransaction]) -> Optional[InsiderPattern]:
        """Detect pre-announcement accumulation patterns"""
        try:
            # Look for unusual buying patterns before potential announcements
            recent_buys = [tx for tx in transactions[-20:] if tx.transaction_type == "buy"]
            
            if len(recent_buys) >= 3:
                buy_volume = sum(tx.value_usd for tx in recent_buys)
                unique_buyers = len(set(tx.whale_address for tx in recent_buys))
                time_span = (max(tx.timestamp for tx in recent_buys) - min(tx.timestamp for tx in recent_buys)).total_seconds() / 3600
                
                # Unusual accumulation: high volume, multiple wallets, short time
                if buy_volume > 15000 and unique_buyers >= 2 and time_span < 12:
                    pattern = InsiderPattern(
                        pattern_id=f"accumulation_{int(time.time())}",
                        token_address=recent_buys[0].token_address,
                        wallets_involved=[tx.whale_address for tx in recent_buys],
                        pattern_type="pre_announcement_accumulation",
                        confidence_score=0.6,
                        first_detected=datetime.now(),
                        last_activity=max(tx.timestamp for tx in recent_buys),
                        total_volume=buy_volume,
                        description=f"Unusual accumulation: ${buy_volume:,.0f} by {unique_buyers} wallets in {time_span:.1f}h",
                        risk_assessment="MEDIUM - Potential insider knowledge"
                    )
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting pre-announcement accumulation: {e}")
            return None
    
    async def _detect_suspicious_timing(self, transactions: List[WhaleTransaction]) -> Optional[InsiderPattern]:
        """Detect suspicious timing patterns"""
        try:
            # Look for transactions right before major price movements
            # This would require price data integration
            
            # For demo, detect rapid sell-offs
            recent_sells = [tx for tx in transactions[-10:] if tx.transaction_type == "sell"]
            
            if len(recent_sells) >= 3:
                sell_volume = sum(tx.value_usd for tx in recent_sells)
                time_span = (max(tx.timestamp for tx in recent_sells) - min(tx.timestamp for tx in recent_sells)).total_seconds() / 3600
                
                if sell_volume > 20000 and time_span < 2:  # Large sells in short time
                    pattern = InsiderPattern(
                        pattern_id=f"timing_{int(time.time())}",
                        token_address=recent_sells[0].token_address,
                        wallets_involved=[tx.whale_address for tx in recent_sells],
                        pattern_type="suspicious_timing",
                        confidence_score=0.8,
                        first_detected=datetime.now(),
                        last_activity=max(tx.timestamp for tx in recent_sells),
                        total_volume=sell_volume,
                        description=f"Rapid sell-off: ${sell_volume:,.0f} in {time_span:.1f}h",
                        risk_assessment="HIGH - Potential exit before bad news"
                    )
                    return pattern
            
            return None
            
        except Exception as e:
            logger.error(f"Error detecting suspicious timing: {e}")
            return None
    
    async def _store_whale_wallet(self, whale: WhaleWallet, token_address: str):
        """Store whale wallet information in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO whale_wallets (
                    address, token_address, label, balance, percentage_holdings,
                    first_seen, last_activity, total_transactions, avg_transaction_size,
                    profit_loss, wallet_age_days, smart_money_score, risk_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                whale.address, token_address, whale.label, whale.balance,
                whale.percentage_holdings, whale.first_seen.isoformat(),
                whale.last_activity.isoformat(), whale.total_transactions,
                whale.avg_transaction_size, whale.profit_loss, whale.wallet_age_days,
                whale.smart_money_score, whale.risk_level
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing whale wallet: {e}")
    
    async def _store_whale_transaction(self, transaction: WhaleTransaction):
        """Store whale transaction in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR IGNORE INTO whale_transactions (
                    tx_hash, whale_address, token_address, transaction_type, amount,
                    value_usd, timestamp, gas_price, block_number, from_address, to_address
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                transaction.tx_hash, transaction.whale_address, transaction.token_address,
                transaction.transaction_type, transaction.amount, transaction.value_usd,
                transaction.timestamp.isoformat(), transaction.gas_price,
                transaction.block_number, transaction.from_address, transaction.to_address
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing whale transaction: {e}")
    
    async def _store_whale_alert(self, alert: WhaleAlert):
        """Store whale alert in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO whale_alerts (
                    id, whale_address, token_address, alert_type, alert_level,
                    amount, value_usd, percentage_change, message, timestamp, action_recommended
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id, alert.whale_address, alert.token_address, alert.alert_type,
                alert.alert_level.value, alert.amount, alert.value_usd,
                alert.percentage_change, alert.message, alert.timestamp.isoformat(),
                alert.action_recommended
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing whale alert: {e}")
    
    async def _store_insider_pattern(self, pattern: InsiderPattern):
        """Store insider pattern in database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO insider_patterns (
                    id, token_address, pattern_type, confidence_score, first_detected,
                    last_activity, total_volume, description, risk_assessment, wallets_involved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pattern.pattern_id, pattern.token_address, pattern.pattern_type,
                pattern.confidence_score, pattern.first_detected.isoformat(),
                pattern.last_activity.isoformat(), pattern.total_volume,
                pattern.description, pattern.risk_assessment,
                json.dumps(pattern.wallets_involved)
            ))
            
            conn.commit()
            conn.close()
            
            logger.warning(f"🕵️ Insider pattern detected: {pattern.description}")
            
        except Exception as e:
            logger.error(f"Error storing insider pattern: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old whale monitoring data"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Keep only last 30 days of transactions
            cursor.execute('''
                DELETE FROM whale_transactions
                WHERE timestamp < datetime('now', '-30 days')
            ''')
            
            # Keep only last 7 days of alerts
            cursor.execute('''
                DELETE FROM whale_alerts
                WHERE timestamp < datetime('now', '-7 days')
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error cleaning up whale data: {e}")
    
    def get_whale_summary(self, token_address: str = None) -> Dict:
        """Get summary of whale activity"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            if token_address:
                # Token-specific summary
                cursor.execute('''
                    SELECT COUNT(*), AVG(smart_money_score), AVG(percentage_holdings)
                    FROM whale_wallets WHERE token_address = ?
                ''', (token_address,))
                
                whale_stats = cursor.fetchone()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM whale_alerts 
                    WHERE token_address = ? AND timestamp > datetime('now', '-24 hours')
                ''', (token_address,))
                
                recent_alerts = cursor.fetchone()[0]
                
            else:
                # Overall summary
                cursor.execute('''
                    SELECT COUNT(*), AVG(smart_money_score), AVG(percentage_holdings)
                    FROM whale_wallets
                ''')
                
                whale_stats = cursor.fetchone()
                
                cursor.execute('''
                    SELECT COUNT(*) FROM whale_alerts 
                    WHERE timestamp > datetime('now', '-24 hours')
                ''')
                
                recent_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                "total_whales": whale_stats[0] if whale_stats[0] else 0,
                "avg_smart_money_score": whale_stats[1] if whale_stats[1] else 0,
                "avg_holdings_percentage": whale_stats[2] if whale_stats[2] else 0,
                "alerts_24h": recent_alerts,
                "monitoring_active": self.monitoring_active
            }
            
        except Exception as e:
            logger.error(f"Error getting whale summary: {e}")
            return {"error": str(e)}
    
    async def get_top_whales(self, token_address: str, limit: int = 10) -> List[WhaleWallet]:
        """Get top whales for a token"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM whale_wallets 
                WHERE token_address = ?
                ORDER BY percentage_holdings DESC
                LIMIT ?
            ''', (token_address, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            whales = []
            for row in rows:
                whale = WhaleWallet(
                    address=row[1], label=row[3], balance=row[4],
                    percentage_holdings=row[5], first_seen=datetime.fromisoformat(row[6]),
                    last_activity=datetime.fromisoformat(row[7]), total_transactions=row[8],
                    avg_transaction_size=row[9], profit_loss=row[10],
                    wallet_age_days=row[11], smart_money_score=row[12], risk_level=row[13]
                )
                whales.append(whale)
            
            return whales
            
        except Exception as e:
            logger.error(f"Error getting top whales: {e}")
            return []

# Example usage and testing
async def main():
    """Test the whale monitoring system"""
    whale_monitor = WhaleMonitor()
    
    # Test tokens
    test_tokens = ["0xTEST1", "0xTEST2"]
    
    print("🐋 Testing Whale Monitoring System")
    print("=" * 50)
    
    # Simulate monitoring for a short period
    monitoring_task = asyncio.create_task(whale_monitor.start_monitoring(test_tokens))
    
    # Let it run for 30 seconds
    await asyncio.sleep(30)
    
    # Stop monitoring
    whale_monitor.stop_monitoring()
    
    # Get summary
    summary = whale_monitor.get_whale_summary()
    print(f"\nWhale Summary:")
    print(f"Total Whales: {summary.get('total_whales', 0)}")
    print(f"Avg Smart Money Score: {summary.get('avg_smart_money_score', 0):.2f}")
    print(f"Alerts in 24h: {summary.get('alerts_24h', 0)}")
    
    # Get top whales
    top_whales = await whale_monitor.get_top_whales(test_tokens[0])
    print(f"\nTop Whales for {test_tokens[0]}:")
    for whale in top_whales[:3]:
        print(f"- {whale.label}: {whale.percentage_holdings:.1f}% holdings (Smart Score: {whale.smart_money_score:.2f})")

if __name__ == "__main__":
    asyncio.run(main())