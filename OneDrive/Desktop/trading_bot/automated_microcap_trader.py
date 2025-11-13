#!/usr/bin/env python3
"""
Automated Microcap Token Discovery and Trading System
Integrates with existing high volatility strategy and GUI controls
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from strategies.high_volatility_low_cap import HighVolatilityLowCapStrategy, LowCapCandidate
from data.social_sentiment import EnhancedSocialSentimentCollector
from risk.adaptive_scaling import AdaptiveProfitScaling

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MicrocapOpportunity:
    """Enhanced microcap opportunity with automation scoring"""
    symbol: str
    contract_address: str
    market_cap: float
    daily_volume: float
    price: float
    price_change_24h: float
    
    # Risk metrics
    rugpull_risk_score: float
    liquidity_score: float
    volatility_score: float
    holder_concentration: float
    
    # Sentiment data
    social_sentiment_score: float
    news_sentiment_score: float
    community_strength: float
    
    # Technical indicators
    rsi: float
    momentum_score: float
    volume_surge: bool
    
    # Automation scoring
    automation_score: float
    confidence_level: float
    recommended_allocation: float
    
    # Metadata
    discovery_source: str
    first_seen: datetime
    last_updated: datetime

@dataclass
class RiskLimits:
    """Risk limits for automated trading"""
    max_position_size_usd: float = 500.0
    max_portfolio_risk_pct: float = 15.0
    max_daily_trades: int = 10
    min_confidence_threshold: float = 0.65
    max_rugpull_risk: float = 0.6
    min_liquidity_score: float = 2.0

class AutomatedMicrocapTrader:
    """
    Fully automated microcap token discovery and trading system
    """
    
    def __init__(self, 
                 config_path: str = "config.json",
                 initial_capital: float = 10000.0,
                 enable_live_trading: bool = False):
        
        self.config = self._load_config(config_path)
        self.initial_capital = initial_capital
        self.available_capital = initial_capital
        self.enable_live_trading = enable_live_trading
        
        # Initialize components
        self.risk_limits = RiskLimits()
        self.opportunities = {}  # symbol -> MicrocapOpportunity
        self.active_positions = {}
        self.trade_history = []
        
        # Initialize strategies and data collectors
        self._init_strategies()
        self._init_database()
        
        # Performance tracking
        self.daily_trades_count = 0
        self.last_reset_date = datetime.now().date()
        self.total_pnl = 0.0
        
        logger.info(f"🤖 Automated Microcap Trader initialized")
        logger.info(f"💰 Capital: ${initial_capital:,.2f}")
        logger.info(f"🎯 Live trading: {'ENABLED' if enable_live_trading else 'SIMULATION'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {
                'api_keys': {
                    'coinmarketcap': '6cad35f36d7b4e069b8dcb0eb9d17d56',
                    'coingecko': 'CG-uKph8trS6RiycsxwVQtxfxvF',
                    'dappradar': 'xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA',
                    'worldnews': '46af273710a543ee8e821382082bb08e'
                },
                'automation': {
                    'scan_interval_minutes': 3,
                    'position_monitoring_seconds': 30,
                    'max_concurrent_positions': 5,
                    'enable_sentiment_analysis': True,
                    'enable_technical_analysis': True
                }
            }
    
    def _init_strategies(self):
        """Initialize trading strategies and data collectors"""
        # High volatility low cap strategy - configured for 500k-1.5M market cap
        self.hv_strategy = HighVolatilityLowCapStrategy(
            small_fund_usd=self.initial_capital * 0.3,  # 30% for high risk trades
            coinmarketcap_api_key=self.config['api_keys']['coinmarketcap'],
            coingecko_api_key=self.config['api_keys']['coingecko'],
            dappradar_api_key=self.config['api_keys']['dappradar'],
            min_market_cap=500000.0,     # 500k minimum market cap
            max_market_cap=1500000.0,    # 1.5M maximum market cap
            min_daily_volume=100000.0,   # Higher volume requirement for liquidity
            min_liquidity_score=0.7      # High liquidity requirement
        )
        
        # Enhanced sentiment collector
        self.sentiment_collector = EnhancedSocialSentimentCollector(
            worldnewsapi_key=self.config['api_keys']['worldnews']
        )
        
        # Adaptive scaling for position sizing
        self.adaptive_scaler = AdaptiveProfitScaling()
        
        logger.info("✅ Strategies and data collectors initialized")
    
    def _init_database(self):
        """Initialize database for storing opportunities and trades"""
        os.makedirs('data', exist_ok=True)
        self.db_path = 'data/automated_microcap_trading.db'
        
        with sqlite3.connect(self.db_path) as conn:
            # Opportunities table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS opportunities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE,
                    contract_address TEXT,
                    market_cap REAL,
                    daily_volume REAL,
                    price REAL,
                    rugpull_risk_score REAL,
                    automation_score REAL,
                    confidence_level REAL,
                    discovery_source TEXT,
                    first_seen TEXT,
                    last_updated TEXT,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Automated trades table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS automated_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT,
                    action TEXT,
                    quantity REAL,
                    price REAL,
                    position_size_usd REAL,
                    confidence REAL,
                    automation_score REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    entry_time TEXT,
                    exit_time TEXT,
                    exit_reason TEXT,
                    pnl REAL,
                    status TEXT DEFAULT 'active'
                )
            ''')
            
            # Performance tracking
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    trades_count INTEGER,
                    total_pnl REAL,
                    win_rate REAL,
                    avg_holding_time_hours REAL,
                    max_drawdown REAL
                )
            ''')
    
    async def run_automation_loop(self):
        \"\"\"Main automation loop\"\"\"\n        logger.info(\"🚀 Starting automated microcap trading loop\")\n        \n        while True:\n            try:\n                cycle_start = time.time()\n                \n                # Reset daily counters if new day\n                self._check_daily_reset()\n                \n                # 1. Discover new microcap opportunities\n                await self._discover_opportunities()\n                \n                # 2. Analyze and score opportunities\n                await self._analyze_opportunities()\n                \n                # 3. Execute trades based on automation criteria\n                await self._execute_automated_trades()\n                \n                # 4. Monitor existing positions\n                await self._monitor_positions()\n                \n                # 5. Update performance metrics\n                self._update_performance_metrics()\n                \n                cycle_time = time.time() - cycle_start\n                logger.info(f\"🔄 Automation cycle completed in {cycle_time:.2f}s\")\n                \n                # Wait for next cycle\n                scan_interval = self.config.get('automation', {}).get('scan_interval_minutes', 3)\n                await asyncio.sleep(scan_interval * 60)\n                \n            except Exception as e:\n                logger.error(f\"❌ Automation loop error: {e}\")\n                await asyncio.sleep(60)  # Wait 1 minute before retrying\n    \n    async def _discover_opportunities(self):\n        \"\"\"Discover new microcap opportunities using multiple sources\"\"\"\n        logger.info(\"🔍 Discovering microcap opportunities...\")\n        \n        try:\n            # Use existing high volatility strategy to find candidates\n            low_cap_candidates = await self.hv_strategy.scan_low_cap_opportunities_with_sentiment()\n            \n            opportunities_found = 0\n            \n            for candidate in low_cap_candidates[:20]:  # Process top 20\n                # Convert to MicrocapOpportunity\n                opportunity = await self._convert_to_opportunity(candidate)\n                \n                if opportunity and self._meets_discovery_criteria(opportunity):\n                    # Store or update opportunity\n                    self.opportunities[opportunity.symbol] = opportunity\n                    await self._store_opportunity_in_db(opportunity)\n                    opportunities_found += 1\n            \n            logger.info(f\"💎 Found {opportunities_found} new/updated opportunities\")\n            \n        except Exception as e:\n            logger.error(f\"Discovery error: {e}\")\n    \n    async def _convert_to_opportunity(self, candidate: LowCapCandidate) -> Optional[MicrocapOpportunity]:\n        \"\"\"Convert LowCapCandidate to MicrocapOpportunity with enhanced data\"\"\"\n        try:\n            # Get additional market data\n            market_data = await self._fetch_enhanced_market_data(candidate.symbol)\n            \n            if not market_data:\n                return None\n            \n            # Calculate enhanced scores\n            automation_score = self._calculate_automation_score(candidate, market_data)\n            \n            # Get sentiment data\n            sentiment_data = await self._get_sentiment_data(candidate.symbol)\n            \n            return MicrocapOpportunity(\n                symbol=candidate.symbol,\n                contract_address=candidate.contract_address,\n                market_cap=candidate.market_cap,\n                daily_volume=candidate.daily_volume,\n                price=market_data.get('price', 0.0),\n                price_change_24h=market_data.get('price_change_24h', 0.0),\n                \n                # Risk metrics\n                rugpull_risk_score=candidate.rugpull_risk_score,\n                liquidity_score=candidate.liquidity_score,\n                volatility_score=candidate.volatility_score,\n                holder_concentration=market_data.get('holder_concentration', 0.5),\n                \n                # Sentiment\n                social_sentiment_score=sentiment_data.get('social_sentiment', 0.0),\n                news_sentiment_score=sentiment_data.get('news_sentiment', 0.0),\n                community_strength=sentiment_data.get('community_strength', 0.5),\n                \n                # Technical indicators\n                rsi=market_data.get('rsi', 50.0),\n                momentum_score=market_data.get('momentum_score', 0.5),\n                volume_surge=market_data.get('volume_surge', False),\n                \n                # Automation scoring\n                automation_score=automation_score,\n                confidence_level=self._calculate_confidence(candidate, market_data, sentiment_data),\n                recommended_allocation=self._calculate_recommended_allocation(automation_score),\n                \n                # Metadata\n                discovery_source='high_volatility_strategy',\n                first_seen=datetime.now(),\n                last_updated=datetime.now()\n            )\n            \n        except Exception as e:\n            logger.error(f\"Error converting candidate {candidate.symbol}: {e}\")\n            return None\n    \n    async def _fetch_enhanced_market_data(self, symbol: str) -> Dict:\n        \"\"\"Fetch enhanced market data for a symbol\"\"\"\n        # Simulate enhanced market data (in real implementation, use multiple APIs)\n        import random\n        \n        return {\n            'price': random.uniform(0.001, 10.0),\n            'price_change_24h': random.uniform(-20.0, 30.0),\n            'holder_concentration': random.uniform(0.2, 0.8),\n            'rsi': random.uniform(20.0, 80.0),\n            'momentum_score': random.uniform(0.0, 1.0),\n            'volume_surge': random.choice([True, False])\n        }\n    \n    async def _get_sentiment_data(self, symbol: str) -> Dict:\n        \"\"\"Get sentiment data for a symbol\"\"\"\n        try:\n            if self.config.get('automation', {}).get('enable_sentiment_analysis', True):\n                # Get combined sentiment from multiple sources\n                sentiment_result = await self.sentiment_collector.get_professional_combined_sentiment([symbol])\n                \n                if symbol in sentiment_result:\n                    data = sentiment_result[symbol]\n                    return {\n                        'social_sentiment': data.get('sentiment_score', 0.0),\n                        'news_sentiment': data.get('detailed_data', {}).get('worldnews', {}).get('sentiment_score', 0.0),\n                        'community_strength': data.get('confidence', 0.5)\n                    }\n            \n            # Default neutral sentiment\n            return {\n                'social_sentiment': 0.0,\n                'news_sentiment': 0.0,\n                'community_strength': 0.5\n            }\n            \n        except Exception as e:\n            logger.error(f\"Sentiment analysis error for {symbol}: {e}\")\n            return {'social_sentiment': 0.0, 'news_sentiment': 0.0, 'community_strength': 0.5}\n    \n    def _calculate_automation_score(self, candidate: LowCapCandidate, market_data: Dict) -> float:\n        \"\"\"Calculate automation score for trading decision\"\"\"\n        score = 0.0\n        \n        # Volatility component (higher volatility = higher score for microcaps)\n        volatility_score = min(candidate.volatility_score / 10.0, 1.0)\n        score += volatility_score * 0.25\n        \n        # Liquidity component (higher liquidity = higher score)\n        liquidity_score = min(candidate.liquidity_score / 5.0, 1.0)\n        score += liquidity_score * 0.20\n        \n        # Risk component (lower rugpull risk = higher score)\n        risk_score = 1.0 - candidate.rugpull_risk_score\n        score += risk_score * 0.25\n        \n        # Technical momentum\n        momentum_score = market_data.get('momentum_score', 0.5)\n        score += momentum_score * 0.15\n        \n        # Volume surge bonus\n        if market_data.get('volume_surge', False):\n            score += 0.10\n        \n        # RSI positioning (favor oversold conditions)\n        rsi = market_data.get('rsi', 50.0)\n        if rsi < 30:  # Oversold\n            score += 0.05\n        elif rsi > 70:  # Overbought - penalty\n            score -= 0.05\n        \n        return min(max(score, 0.0), 1.0)\n    \n    def _calculate_confidence(self, candidate: LowCapCandidate, market_data: Dict, sentiment_data: Dict) -> float:\n        \"\"\"Calculate confidence level for the opportunity\"\"\"\n        confidence = 0.5  # Base confidence\n        \n        # Social momentum boost\n        social_sentiment = sentiment_data.get('social_sentiment', 0.0)\n        if social_sentiment > 0.1:\n            confidence += 0.2\n        elif social_sentiment < -0.1:\n            confidence -= 0.1\n        \n        # Community strength\n        community_strength = sentiment_data.get('community_strength', 0.5)\n        confidence += (community_strength - 0.5) * 0.3\n        \n        # Technical confirmation\n        momentum = market_data.get('momentum_score', 0.5)\n        confidence += (momentum - 0.5) * 0.2\n        \n        # Liquidity adjustment\n        if candidate.liquidity_score > 3.0:\n            confidence += 0.1\n        \n        return min(max(confidence, 0.0), 1.0)\n    \n    def _calculate_recommended_allocation(self, automation_score: float) -> float:\n        \"\"\"Calculate recommended position allocation\"\"\"\n        base_allocation = self.risk_limits.max_position_size_usd\n        \n        # Scale allocation based on automation score\n        scaled_allocation = base_allocation * automation_score\n        \n        return min(scaled_allocation, self.risk_limits.max_position_size_usd)\n    \n    def _meets_discovery_criteria(self, opportunity: MicrocapOpportunity) -> bool:\n        \"\"\"Check if opportunity meets discovery criteria\"\"\"\n        return (\n            opportunity.automation_score >= 0.4 and\n            opportunity.confidence_level >= 0.3 and\n            opportunity.rugpull_risk_score <= 0.8 and\n            opportunity.liquidity_score >= 1.0\n        )\n    \n    async def _analyze_opportunities(self):\n        \"\"\"Analyze and re-score existing opportunities\"\"\"\n        logger.info(f\"📊 Analyzing {len(self.opportunities)} opportunities...\")\n        \n        for symbol, opportunity in list(self.opportunities.items()):\n            try:\n                # Re-fetch market data for updated scoring\n                market_data = await self._fetch_enhanced_market_data(symbol)\n                \n                if market_data:\n                    # Update opportunity with fresh data\n                    opportunity.price = market_data.get('price', opportunity.price)\n                    opportunity.price_change_24h = market_data.get('price_change_24h', 0.0)\n                    opportunity.rsi = market_data.get('rsi', 50.0)\n                    opportunity.last_updated = datetime.now()\n                    \n                    # Recalculate scores\n                    dummy_candidate = LowCapCandidate(\n                        symbol=opportunity.symbol,\n                        contract_address=opportunity.contract_address,\n                        market_cap=opportunity.market_cap,\n                        daily_volume=opportunity.daily_volume,\n                        liquidity_score=opportunity.liquidity_score,\n                        volatility_score=opportunity.volatility_score,\n                        rugpull_risk_score=opportunity.rugpull_risk_score,\n                        holder_count=1000,\n                        token_age_days=30,\n                        social_momentum=0.5,\n                        dev_activity_score=0.5,\n                        audit_status=\"unknown\"\n                    )\n                    \n                    opportunity.automation_score = self._calculate_automation_score(dummy_candidate, market_data)\n                    \n            except Exception as e:\n                logger.error(f\"Error analyzing {symbol}: {e}\")\n    \n    async def _execute_automated_trades(self):\n        \"\"\"Execute trades based on automation criteria\"\"\"\n        if not self.enable_live_trading:\n            logger.info(\"📋 Trade execution disabled (simulation mode)\")\n            return\n        \n        # Check daily trade limits\n        if self.daily_trades_count >= self.risk_limits.max_daily_trades:\n            logger.info(f\"🚫 Daily trade limit reached ({self.daily_trades_count}/{self.risk_limits.max_daily_trades})\")\n            return\n        \n        # Sort opportunities by automation score\n        sorted_opportunities = sorted(\n            self.opportunities.values(),\n            key=lambda x: x.automation_score,\n            reverse=True\n        )\n        \n        trades_executed = 0\n        \n        for opportunity in sorted_opportunities:\n            try:\n                # Check if we should trade this opportunity\n                if self._should_execute_trade(opportunity):\n                    # Calculate position size\n                    position_size = self._calculate_position_size(opportunity)\n                    \n                    if position_size > 0 and self._check_risk_limits(position_size):\n                        # Execute the trade\n                        trade_result = await self._execute_trade(opportunity, position_size)\n                        \n                        if trade_result['success']:\n                            trades_executed += 1\n                            self.daily_trades_count += 1\n                            \n                            logger.info(f\"✅ Executed trade: {opportunity.symbol} - ${position_size:.2f}\")\n                            \n                            # Break if we've hit concurrent position limit\n                            if len(self.active_positions) >= self.config.get('automation', {}).get('max_concurrent_positions', 5):\n                                break\n                        else:\n                            logger.warning(f\"❌ Trade execution failed for {opportunity.symbol}: {trade_result['error']}\")\n                \n            except Exception as e:\n                logger.error(f\"Trade execution error for {opportunity.symbol}: {e}\")\n        \n        if trades_executed > 0:\n            logger.info(f\"🎯 Executed {trades_executed} automated trades\")\n    \n    def _should_execute_trade(self, opportunity: MicrocapOpportunity) -> bool:\n        \"\"\"Determine if we should execute a trade for this opportunity\"\"\"\n        return (\n            opportunity.automation_score >= 0.6 and\n            opportunity.confidence_level >= self.risk_limits.min_confidence_threshold and\n            opportunity.rugpull_risk_score <= self.risk_limits.max_rugpull_risk and\n            opportunity.liquidity_score >= self.risk_limits.min_liquidity_score and\n            opportunity.symbol not in self.active_positions  # Not already trading\n        )\n    \n    def _calculate_position_size(self, opportunity: MicrocapOpportunity) -> float:\n        \"\"\"Calculate position size for the opportunity\"\"\"\n        # Base position size from risk limits\n        base_size = self.risk_limits.max_position_size_usd\n        \n        # Scale by automation score and confidence\n        scaling_factor = (opportunity.automation_score * opportunity.confidence_level)\n        scaled_size = base_size * scaling_factor\n        \n        # Apply adaptive scaling if available\n        if hasattr(self.adaptive_scaler, 'get_position_size'):\n            adaptive_size = self.adaptive_scaler.get_position_size(\n                opportunity.confidence_level, \n                self.available_capital\n            )\n            scaled_size = min(scaled_size, adaptive_size)\n        \n        # Ensure we don't exceed available capital\n        return min(scaled_size, self.available_capital * 0.1)  # Max 10% of capital per trade\n    \n    def _check_risk_limits(self, position_size: float) -> bool:\n        \"\"\"Check if trade meets risk limits\"\"\"\n        # Check available capital\n        if position_size > self.available_capital:\n            return False\n        \n        # Check portfolio risk\n        current_risk = sum(pos['position_size'] for pos in self.active_positions.values())\n        total_risk_after = (current_risk + position_size) / self.initial_capital * 100\n        \n        if total_risk_after > self.risk_limits.max_portfolio_risk_pct:\n            return False\n        \n        # Check position size limit\n        if position_size > self.risk_limits.max_position_size_usd:\n            return False\n        \n        return True\n    \n    async def _execute_trade(self, opportunity: MicrocapOpportunity, position_size: float) -> Dict:\n        \"\"\"Execute a trade (simulated for now)\"\"\"\n        try:\n            trade_id = f\"{opportunity.symbol}_{int(time.time())}\"\n            \n            # Simulate trade execution\n            entry_price = opportunity.price\n            quantity = position_size / entry_price\n            \n            # Calculate stop loss and take profit\n            stop_loss = entry_price * 0.85  # 15% stop loss\n            take_profit = entry_price * 1.4  # 40% take profit\n            \n            # Create position record\n            position = {\n                'trade_id': trade_id,\n                'symbol': opportunity.symbol,\n                'quantity': quantity,\n                'entry_price': entry_price,\n                'current_price': entry_price,\n                'position_size': position_size,\n                'stop_loss': stop_loss,\n                'take_profit': take_profit,\n                'entry_time': datetime.now(),\n                'automation_score': opportunity.automation_score,\n                'confidence': opportunity.confidence_level,\n                'pnl': 0.0,\n                'status': 'active'\n            }\n            \n            # Add to active positions\n            self.active_positions[opportunity.symbol] = position\n            \n            # Update available capital\n            self.available_capital -= position_size\n            \n            # Store in database\n            await self._store_trade_in_db(position)\n            \n            return {'success': True, 'trade_id': trade_id}\n            \n        except Exception as e:\n            logger.error(f\"Trade execution error: {e}\")\n            return {'success': False, 'error': str(e)}\n    \n    async def _monitor_positions(self):\n        \"\"\"Monitor active positions for exit conditions\"\"\"\n        if not self.active_positions:\n            return\n        \n        logger.info(f\"👀 Monitoring {len(self.active_positions)} active positions...\")\n        \n        positions_to_close = []\n        \n        for symbol, position in self.active_positions.items():\n            try:\n                # Simulate price update (in real implementation, fetch from exchange)\n                import random\n                price_change = random.uniform(-0.08, 0.12)  # ±8% to +12% change\n                new_price = position['entry_price'] * (1 + price_change)\n                \n                # Update position\n                position['current_price'] = new_price\n                position['pnl'] = (new_price - position['entry_price']) * position['quantity']\n                \n                # Check exit conditions\n                should_exit, exit_reason = self._check_exit_conditions(position)\n                \n                if should_exit:\n                    positions_to_close.append((symbol, exit_reason))\n                    \n            except Exception as e:\n                logger.error(f\"Position monitoring error for {symbol}: {e}\")\n        \n        # Close positions that met exit criteria\n        for symbol, exit_reason in positions_to_close:\n            await self._close_position(symbol, exit_reason)\n    \n    def _check_exit_conditions(self, position: Dict) -> Tuple[bool, str]:\n        \"\"\"Check if position should be closed\"\"\"\n        current_price = position['current_price']\n        \n        # Stop loss\n        if current_price <= position['stop_loss']:\n            return True, \"stop_loss\"\n        \n        # Take profit\n        if current_price >= position['take_profit']:\n            return True, \"take_profit\"\n        \n        # Time limit (12 hours for quick microcap testing)\n        time_in_position = datetime.now() - position['entry_time']\n        if time_in_position.total_seconds() > 12 * 3600:\n            return True, \"time_limit\"\n        \n        # Emergency exit if massive loss (circuit breaker)\n        pnl_pct = (position['pnl'] / position['position_size']) * 100\n        if pnl_pct < -25:  # 25% loss\n            return True, \"emergency_exit\"\n        \n        return False, \"\"\n    \n    async def _close_position(self, symbol: str, exit_reason: str):\n        \"\"\"Close a position\"\"\"\n        try:\n            position = self.active_positions[symbol]\n            \n            # Calculate final values\n            exit_value = position['current_price'] * position['quantity']\n            final_pnl = position['pnl']\n            \n            # Update capital\n            self.available_capital += exit_value\n            self.total_pnl += final_pnl\n            \n            # Log closure\n            pnl_pct = (final_pnl / position['position_size']) * 100\n            logger.info(f\"🔄 CLOSED {symbol}: {exit_reason} - P&L: ${final_pnl:+.2f} ({pnl_pct:+.1f}%)\")\n            \n            # Update database\n            await self._update_trade_in_db(position, exit_reason)\n            \n            # Remove from active positions\n            del self.active_positions[symbol]\n            \n            # Add to trade history\n            position['exit_time'] = datetime.now()\n            position['exit_reason'] = exit_reason\n            self.trade_history.append(position.copy())\n            \n        except Exception as e:\n            logger.error(f\"Position closing error for {symbol}: {e}\")\n    \n    async def _store_opportunity_in_db(self, opportunity: MicrocapOpportunity):\n        \"\"\"Store opportunity in database\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.execute('''\n                    INSERT OR REPLACE INTO opportunities \n                    (symbol, contract_address, market_cap, daily_volume, price, \n                     rugpull_risk_score, automation_score, confidence_level, \n                     discovery_source, first_seen, last_updated)\n                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                ''', (\n                    opportunity.symbol,\n                    opportunity.contract_address,\n                    opportunity.market_cap,\n                    opportunity.daily_volume,\n                    opportunity.price,\n                    opportunity.rugpull_risk_score,\n                    opportunity.automation_score,\n                    opportunity.confidence_level,\n                    opportunity.discovery_source,\n                    opportunity.first_seen.isoformat(),\n                    opportunity.last_updated.isoformat()\n                ))\n        except Exception as e:\n            logger.error(f\"Database storage error: {e}\")\n    \n    async def _store_trade_in_db(self, position: Dict):\n        \"\"\"Store trade in database\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.execute('''\n                    INSERT INTO automated_trades \n                    (trade_id, symbol, action, quantity, price, position_size_usd,\n                     confidence, automation_score, stop_loss, take_profit, entry_time)\n                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)\n                ''', (\n                    position['trade_id'],\n                    position['symbol'],\n                    'buy',\n                    position['quantity'],\n                    position['entry_price'],\n                    position['position_size'],\n                    position['confidence'],\n                    position['automation_score'],\n                    position['stop_loss'],\n                    position['take_profit'],\n                    position['entry_time'].isoformat()\n                ))\n        except Exception as e:\n            logger.error(f\"Trade storage error: {e}\")\n    \n    async def _update_trade_in_db(self, position: Dict, exit_reason: str):\n        \"\"\"Update trade with exit information\"\"\"\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.execute('''\n                    UPDATE automated_trades \n                    SET exit_time = ?, exit_reason = ?, pnl = ?, status = 'closed'\n                    WHERE trade_id = ?\n                ''', (\n                    datetime.now().isoformat(),\n                    exit_reason,\n                    position['pnl'],\n                    position['trade_id']\n                ))\n        except Exception as e:\n            logger.error(f\"Trade update error: {e}\")\n    \n    def _check_daily_reset(self):\n        \"\"\"Check if we need to reset daily counters\"\"\"\n        current_date = datetime.now().date()\n        if current_date != self.last_reset_date:\n            self.daily_trades_count = 0\n            self.last_reset_date = current_date\n            logger.info(f\"📅 Daily counters reset for {current_date}\")\n    \n    def _update_performance_metrics(self):\n        \"\"\"Update performance tracking\"\"\"\n        # Calculate current performance\n        total_pnl_today = sum(\n            trade['pnl'] for trade in self.trade_history \n            if trade.get('exit_time', datetime.now()).date() == datetime.now().date()\n        )\n        \n        # Calculate win rate\n        trades_today = [t for t in self.trade_history if t.get('exit_time', datetime.now()).date() == datetime.now().date()]\n        win_rate = sum(1 for t in trades_today if t['pnl'] > 0) / len(trades_today) if trades_today else 0\n        \n        # Store in database\n        try:\n            with sqlite3.connect(self.db_path) as conn:\n                conn.execute('''\n                    INSERT OR REPLACE INTO daily_performance\n                    (date, trades_count, total_pnl, win_rate)\n                    VALUES (?, ?, ?, ?)\n                ''', (\n                    datetime.now().date().isoformat(),\n                    self.daily_trades_count,\n                    total_pnl_today,\n                    win_rate\n                ))\n        except Exception as e:\n            logger.error(f\"Performance update error: {e}\")\n    \n    def get_status_summary(self) -> Dict:\n        \"\"\"Get current status summary for GUI display\"\"\"\n        return {\n            'available_capital': self.available_capital,\n            'total_pnl': self.total_pnl,\n            'active_positions': len(self.active_positions),\n            'daily_trades': self.daily_trades_count,\n            'opportunities_count': len(self.opportunities),\n            'top_opportunities': sorted(\n                self.opportunities.values(),\n                key=lambda x: x.automation_score,\n                reverse=True\n            )[:5],\n            'recent_trades': self.trade_history[-10:] if self.trade_history else []\n        }\n    \n    def update_risk_limits(self, \n                          max_position_size: float = None,\n                          max_portfolio_risk: float = None,\n                          max_daily_trades: int = None,\n                          min_confidence: float = None):\n        \"\"\"Update risk limits from GUI\"\"\"\n        if max_position_size is not None:\n            self.risk_limits.max_position_size_usd = max_position_size\n        if max_portfolio_risk is not None:\n            self.risk_limits.max_portfolio_risk_pct = max_portfolio_risk\n        if max_daily_trades is not None:\n            self.risk_limits.max_daily_trades = max_daily_trades\n        if min_confidence is not None:\n            self.risk_limits.min_confidence_threshold = min_confidence\n        \n        logger.info(f\"🎯 Risk limits updated: {asdict(self.risk_limits)}\")\n\nasync def main():\n    \"\"\"Test the automated microcap trader\"\"\"\n    trader = AutomatedMicrocapTrader(\n        initial_capital=5000.0,\n        enable_live_trading=True  # Set to False for pure simulation\n    )\n    \n    # Run a few cycles for testing\n    logger.info(\"🧪 Running test cycles...\")\n    \n    for cycle in range(3):\n        logger.info(f\"📋 Test cycle {cycle + 1}/3\")\n        \n        # Run one automation cycle\n        await trader._discover_opportunities()\n        await trader._analyze_opportunities()\n        await trader._execute_automated_trades()\n        await trader._monitor_positions()\n        \n        # Print status\n        status = trader.get_status_summary()\n        logger.info(f\"💰 Capital: ${status['available_capital']:.2f} | P&L: ${status['total_pnl']:+.2f}\")\n        logger.info(f\"📊 Positions: {status['active_positions']} | Opportunities: {status['opportunities_count']}\")\n        \n        # Wait between cycles\n        await asyncio.sleep(10)\n    \n    logger.info(\"✅ Test completed\")\n\nif __name__ == \"__main__\":\n    asyncio.run(main())\n
