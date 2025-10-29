"""
Private Trading Ledger System
Tracks all trades with sentiment analysis and statistical data
"""

import json
import csv
import sqlite3
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TradeStatus(Enum):
    PENDING = "pending"
    EXECUTED = "executed"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"

class TradeSide(Enum):
    BUY = "buy"
    SELL = "sell"

@dataclass
class TradeRecord:
    """Complete trade record with all metadata"""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: TradeSide
    quantity: float
    price: float
    total_value: float
    fees: float
    status: TradeStatus
    
    # Sentiment data
    market_sentiment: str  # bullish/neutral/bearish
    social_sentiment: float  # -1 to 1
    defi_sentiment: str
    whale_activity: str
    
    # Technical indicators
    rsi: float
    macd: float
    bollinger_position: float
    volume_profile: str
    
    # AI predictions
    ai_confidence: float
    predicted_direction: str
    signal_strength: float
    
    # Risk metrics
    position_size_factor: float
    risk_score: float
    max_drawdown_risk: float
    
    # Performance tracking
    pnl_realized: float = 0.0
    pnl_unrealized: float = 0.0
    holding_period: Optional[timedelta] = None
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    
    # Metadata
    strategy_used: str = ""
    notes: str = ""
    execution_venue: str = ""
    slippage: float = 0.0

class TradingLedger:
    """
    Comprehensive trading ledger with advanced analytics
    """
    
    def __init__(self, db_path: str = "data/trading_ledger.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.init_database()
        logger.info(f"Trading ledger initialized at {self.db_path}")
    
    def init_database(self):
        """Initialize SQLite database with proper schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    price REAL,
                    total_value REAL,
                    fees REAL,
                    status TEXT,
                    
                    market_sentiment TEXT,
                    social_sentiment REAL,
                    defi_sentiment TEXT,
                    whale_activity TEXT,
                    
                    rsi REAL,
                    macd REAL,
                    bollinger_position REAL,
                    volume_profile TEXT,
                    
                    ai_confidence REAL,
                    predicted_direction TEXT,
                    signal_strength REAL,
                    
                    position_size_factor REAL,
                    risk_score REAL,
                    max_drawdown_risk REAL,
                    
                    pnl_realized REAL,
                    pnl_unrealized REAL,
                    holding_period TEXT,
                    exit_price REAL,
                    exit_timestamp TEXT,
                    
                    strategy_used TEXT,
                    notes TEXT,
                    execution_venue TEXT,
                    slippage REAL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON trades(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strategy ON trades(strategy_used)")
    
    def log_trade(self, trade: TradeRecord) -> bool:
        """Log a trade to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                trade_dict = asdict(trade)
                
                # Convert enums and datetime objects to strings
                trade_dict['side'] = trade_dict['side'].value if hasattr(trade_dict['side'], 'value') else str(trade_dict['side'])
                trade_dict['status'] = trade_dict['status'].value if hasattr(trade_dict['status'], 'value') else str(trade_dict['status'])
                trade_dict['timestamp'] = trade.timestamp.isoformat()
                trade_dict['exit_timestamp'] = trade.exit_timestamp.isoformat() if trade.exit_timestamp else None
                trade_dict['holding_period'] = str(trade.holding_period) if trade.holding_period else None
                
                # Insert trade
                columns = ', '.join(trade_dict.keys())
                placeholders = ', '.join(['?' for _ in trade_dict.keys()])
                query = f"INSERT OR REPLACE INTO trades ({columns}) VALUES ({placeholders})"
                
                conn.execute(query, list(trade_dict.values()))
                
            logger.info(f"Trade logged: {trade.trade_id} - {trade.symbol} {trade.side.value} {trade.quantity} @ ${trade.price}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log trade {trade.trade_id}: {e}")
            return False
    
    def update_trade_exit(self, trade_id: str, exit_price: float, pnl_realized: float) -> bool:
        """Update trade with exit information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                exit_timestamp = datetime.now().isoformat()
                
                # Calculate holding period
                trade_data = conn.execute(
                    "SELECT timestamp FROM trades WHERE trade_id = ?", 
                    (trade_id,)
                ).fetchone()
                
                if trade_data:
                    entry_time = datetime.fromisoformat(trade_data[0])
                    holding_period = datetime.now() - entry_time
                    
                    conn.execute("""
                        UPDATE trades 
                        SET exit_price = ?, pnl_realized = ?, exit_timestamp = ?, 
                            holding_period = ?, status = ?
                        WHERE trade_id = ?
                    """, (exit_price, pnl_realized, exit_timestamp, str(holding_period), 
                          TradeStatus.EXECUTED.value, trade_id))
                    
                    logger.info(f"Trade exit updated: {trade_id} - P&L: ${pnl_realized:.2f}")
                    return True
            
        except Exception as e:
            logger.error(f"Failed to update trade exit {trade_id}: {e}")
            
        return False
    
    def get_trade_history(self, limit: int = 100, symbol: str = None) -> List[Dict]:
        """Get trade history with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT * FROM trades"
                params = []
                
                if symbol:
                    query += " WHERE symbol = ?"
                    params.append(symbol)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor = conn.execute(query, params)
                columns = [desc[0] for desc in cursor.description]
                
                trades = []
                for row in cursor.fetchall():
                    trade_dict = dict(zip(columns, row))
                    trades.append(trade_dict)
                
                return trades
                
        except Exception as e:
            logger.error(f"Failed to get trade history: {e}")
            return []
    
    def get_performance_metrics(self, days: int = 30) -> Dict:
        """Calculate comprehensive performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                
                # Get all trades in the period
                trades = conn.execute("""
                    SELECT * FROM trades 
                    WHERE timestamp >= ? AND status = ?
                    ORDER BY timestamp
                """, (cutoff_date, TradeStatus.EXECUTED.value)).fetchall()
                
                if not trades:
                    return {'error': 'No trades found in the specified period'}
                
                # Calculate metrics
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if trade[19] > 0)  # pnl_realized
                losing_trades = total_trades - winning_trades
                
                total_pnl = sum(trade[19] for trade in trades)  # pnl_realized
                total_fees = sum(trade[7] for trade in trades)  # fees
                
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Calculate average holding periods
                holding_periods = []
                for trade in trades:
                    if trade[20]:  # holding_period
                        try:
                            # Parse timedelta string
                            period_str = trade[20]
                            if 'days' in period_str:
                                days_part = float(period_str.split(' days')[0])
                                holding_periods.append(days_part * 24)  # Convert to hours
                            else:
                                # Assume it's in hours format
                                parts = period_str.split(':')
                                hours = float(parts[0]) if len(parts) > 0 else 0
                                holding_periods.append(hours)
                        except:
                            continue
                
                avg_holding_period = np.mean(holding_periods) if holding_periods else 0
                
                # Sentiment analysis
                sentiment_counts = {}
                for trade in trades:
                    sentiment = trade[9]  # market_sentiment
                    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
                
                # Most profitable strategy
                strategy_pnl = {}
                for trade in trades:
                    strategy = trade[24] or 'unknown'  # strategy_used
                    pnl = trade[19]  # pnl_realized
                    if strategy not in strategy_pnl:
                        strategy_pnl[strategy] = []
                    strategy_pnl[strategy].append(pnl)
                
                best_strategy = max(strategy_pnl.keys(), 
                                  key=lambda k: sum(strategy_pnl[k])) if strategy_pnl else 'N/A'
                
                return {
                    'period_days': days,
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': round(win_rate, 2),
                    'total_pnl': round(total_pnl, 2),
                    'total_fees': round(total_fees, 2),
                    'net_profit': round(total_pnl - total_fees, 2),
                    'avg_holding_period_hours': round(avg_holding_period, 2),
                    'sentiment_distribution': sentiment_counts,
                    'best_strategy': best_strategy,
                    'best_strategy_pnl': round(sum(strategy_pnl.get(best_strategy, [0])), 2)
                }
                
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {'error': str(e)}
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export trade data to CSV"""
        if not filename:
            filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp", conn)
                df.to_csv(filename, index=False)
                logger.info(f"Trades exported to {filename}")
                return filename
                
        except Exception as e:
            logger.error(f"Failed to export trades: {e}")
            return ""
    
    def get_sentiment_analysis(self) -> Dict:
        """Analyze sentiment patterns in trading"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                trades = conn.execute("""
                    SELECT market_sentiment, social_sentiment, defi_sentiment, pnl_realized
                    FROM trades WHERE status = ?
                """, (TradeStatus.EXECUTED.value,)).fetchall()
                
                if not trades:
                    return {'error': 'No completed trades found'}
                
                # Analyze sentiment vs performance
                sentiment_performance = {
                    'bullish': {'count': 0, 'total_pnl': 0, 'avg_pnl': 0},
                    'neutral': {'count': 0, 'total_pnl': 0, 'avg_pnl': 0},
                    'bearish': {'count': 0, 'total_pnl': 0, 'avg_pnl': 0}
                }
                
                social_sentiments = []
                for trade in trades:
                    market_sentiment, social_sentiment, defi_sentiment, pnl = trade
                    
                    if market_sentiment in sentiment_performance:
                        sentiment_performance[market_sentiment]['count'] += 1
                        sentiment_performance[market_sentiment]['total_pnl'] += pnl
                    
                    if social_sentiment is not None:
                        social_sentiments.append(social_sentiment)
                
                # Calculate averages
                for sentiment in sentiment_performance:
                    data = sentiment_performance[sentiment]
                    if data['count'] > 0:
                        data['avg_pnl'] = data['total_pnl'] / data['count']
                
                return {
                    'sentiment_performance': sentiment_performance,
                    'avg_social_sentiment': np.mean(social_sentiments) if social_sentiments else 0,
                    'social_sentiment_std': np.std(social_sentiments) if social_sentiments else 0,
                    'total_trades_analyzed': len(trades)
                }
                
        except Exception as e:
            logger.error(f"Failed to analyze sentiment: {e}")
            return {'error': str(e)}

# Test function
def test_trading_ledger():
    """Test trading ledger functionality"""
    ledger = TradingLedger()
    
    # Create sample trade
    sample_trade = TradeRecord(
        trade_id="TEST_001",
        timestamp=datetime.now(),
        symbol="BTC/USDT",
        side=TradeSide.BUY,
        quantity=0.1,
        price=45000.0,
        total_value=4500.0,
        fees=4.5,
        status=TradeStatus.EXECUTED,
        market_sentiment="bullish",
        social_sentiment=0.7,
        defi_sentiment="positive",
        whale_activity="accumulating",
        rsi=65.0,
        macd=120.5,
        bollinger_position=0.8,
        volume_profile="high",
        ai_confidence=0.85,
        predicted_direction="up",
        signal_strength=0.9,
        position_size_factor=1.5,
        risk_score=0.3,
        max_drawdown_risk=0.05,
        strategy_used="momentum_scalping"
    )
    
    # Log the trade
    success = ledger.log_trade(sample_trade)
    print(f"Trade logged: {success}")
    
    # Get trade history
    history = ledger.get_trade_history(10)
    print(f"Trade history: {len(history)} trades")
    
    # Update with exit
    ledger.update_trade_exit("TEST_001", 46000.0, 100.0)
    
    # Get performance metrics
    metrics = ledger.get_performance_metrics()
    print(f"Performance metrics: {metrics}")

if __name__ == "__main__":
    test_trading_ledger()