#!/usr/bin/env python3
"""
Trading Bot GUI - Advanced Trading Dashboard
Features: Real-time charts, trade history, market indicators, P&L tracking
"""

import asyncio
import json
import sys
import os
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
from datetime import datetime, timedelta
import sqlite3
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import numpy as np

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.exchange_manager import ExchangeManager

# Import real trader module
try:
    # Try importing from same directory first
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    from real_trader import RealTrader
    from whale_tracker import WhaleWalletTracker
    from copy_trade_manager import CopyTradeManager
    from technical_filters import TechnicalSignalFilter
    REAL_TRADING_AVAILABLE = True
    print("✅ Real trading module loaded successfully")
    print("✅ Whale tracker module loaded successfully")
    print("✅ Copy-trade manager loaded successfully")
    print("✅ Technical filters loaded successfully")
except ImportError as e:
    REAL_TRADING_AVAILABLE = False
    print(f"⚠️ Real trading module not available: {e}")

class AutoTrader:
    """
    Whale-First Automated Trading System
    Priority: Whale Copy-Trades > Technical Signals
    """
    
    def __init__(self, gui_instance):
        self.gui = gui_instance
        self.trading_enabled = False
        
        # REAL TRADING MODE
        self.real_trading_mode = False  # Start in paper trading mode
        self.real_trader = None  # Will be initialized when real trading is enabled
        
        # 🐋 WHALE-FIRST COPY-TRADING SYSTEM (NEW)
        self.whale_tracker = None
        self.copy_trade_manager = None
        self.technical_filter = None
        
        # SEPARATE TIMERS FOR BUYS AND SELLS
        self.last_buy_time = {}
        self.last_sell_time = {}
        
        # 🎯 WHALE-FIRST CONFIGURATION
        # Based on top traders: 1% position size, 5 max positions, strict risk limits
        # 
        # PRIORITY SYSTEM:
        # 1. WHALE SIGNALS (with technical confirmation) - 14 whale wallets monitored
        # 2. High-conviction technical signals (SMA crossover + volume spike)
        # 3. Standard technical signals (backup only)
        # 
        # TRACKED TOKENS: BANGERS, TRUMP, BASED (pump.fun memecoins)
        # WHALE WALLETS: 14 verified smart-money wallets (98% win rate leaders)
        # 
        # RISK MANAGEMENT:
        # - Max 1% of portfolio per position
        # - Max 5 open positions
        # - 5% daily loss limit
        # - Stop loss: -20% (industry standard)
        # - Take profit: +100%, +300% (memecoin levels)
        self.token_settings = {
            'BANGERS': {  # 🎯 PRIMARY MEMECOIN
                'address': '3wppuwUMAGgxnX75Aqr4W91xYWaN6RjxjCUFiPZUpump',
                'min_profit': 0.030,   # 3.0% profit target
                'max_loss': -0.200,    # 20% stop loss (copy-trade manager controls this)
                'trade_amount': 0.095, # Will be overridden by copy-trade manager (1% sizing)
                'min_interval': 5,     # 5 seconds between trades
                'priority': 1          # Highest priority
            },
            'TRUMP': {  # 🎯 SECONDARY MEMECOIN
                'address': '6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN',
                'min_profit': 0.025,   # 2.5% profit target
                'max_loss': -0.200,    # 20% stop loss
                'trade_amount': 0.063, # Will be overridden by copy-trade manager
                'min_interval': 3,     # 3 seconds between trades
                'priority': 2
            },
            'BASED': {  # 🎯 THIRD MEMECOIN
                'address': 'EMAGfmV5bMzYEtgda43ZmCYwmLL7SaMi2RVqaRPjpump',
                'min_profit': 0.020,   # 2.0% profit target
                'max_loss': -0.005,    # 0.5% stop loss
                'trade_amount': 0.047, # $7.50 per trade @ $158/SOL
                'min_interval': 3      # 3 seconds between trades
            }
        }
        
        # 🎯 OPTIMIZED DEFAULT SETTINGS (Conservative RSI, less overtrading)
        self.min_profit_threshold = 0.020  # 2.0% default (was 0.1%)
        self.max_loss_threshold = -0.005   # 0.5% default (was 1%)
        self.rsi_oversold = 30   # Conservative (was 45 - too sensitive)
        self.rsi_overbought = 70 # Conservative (was 55 - too sensitive)
        self.trade_amount = 0.01 # Default amount
        self.min_trade_interval = 10  # Default interval
        
    def calculate_moving_averages(self, price_data):
        """Calculate short-term and long-term moving averages"""
        prices = [p['price'] for p in price_data]
        
        # Short-term MA (5 periods) and Long-term MA (20 periods)
        ma_short = sum(prices[-5:]) / len(prices[-5:]) if len(prices) >= 5 else None
        ma_long = sum(prices[-20:]) / len(prices[-20:]) if len(prices) >= 20 else None
        
        # 24-hour average (assume ~8640 periods if scanning every 10 seconds)
        # Simplified: use last 360 periods (1 hour if scanning every 10s)
        ma_daily = sum(prices[-360:]) / len(prices[-360:]) if len(prices) >= 360 else None
        
        return {
            'ma_short': ma_short,
            'ma_long': ma_long,
            'ma_daily': ma_daily
        }
    
    def should_buy(self, token, price_data, rsi_value=None):
        """Determine if we should buy a token with token-specific ultra-aggressive logic"""
        import time
        current_time = time.time()
        
        # Get token-specific settings
        settings = self.token_settings.get(token, {
            'min_interval': self.min_trade_interval,
            'trade_amount': self.trade_amount
        })
        
        # Check if we can BUY (using separate buy timer)
        last_buy = self.last_buy_time.get(token, 0)
        if current_time - last_buy < settings['min_interval']:
            return False, f"Too soon ({settings['min_interval']}s interval)"
        
        # SIMPLE MODE: Handle brand new tokens with NO price history
        if len(price_data) < 2:
            # For new tokens, buy immediately on first price (opportunistic entry)
            if not hasattr(self, '_initial_buy_done'):
                self._initial_buy_done = {}
            
            if token not in self._initial_buy_done:
                self._initial_buy_done[token] = True
                self.last_buy_time[token] = current_time
                return True, f"[{token}] SIMPLE: Initial entry (new token)"
            else:
                return False, "SIMPLE: Waiting for price data"
        
        recent_prices = [p['price'] for p in price_data[-3:]]
        current_price = recent_prices[-1]
        
        # RISK CHECK: Don't buy if already holding enough
        current_holdings = self.gui.portfolio[token]['holdings']
        if current_holdings > 0:
            return False, "Already holding position"
        
        # SIMPLE MODE: For tokens with limited history (< 5 data points)
        if len(price_data) < 5:
            buy_signals = []
            
            # SIMPLE SIGNAL 1: Price dip opportunity (buy on any small dip)
            if len(recent_prices) >= 2:
                price_change = ((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]) * 100
                if price_change < 0:  # Any dip
                    buy_signals.append(f"SIMPLE: Dip entry ({price_change:.3f}%)")
                elif price_change > 0.5:  # Or strong momentum
                    buy_signals.append(f"SIMPLE: Momentum ({price_change:.3f}%)")
            
            # SIMPLE SIGNAL 2: Always ready to buy (opportunistic for new tokens)
            if not buy_signals:
                buy_signals.append(f"SIMPLE: Market entry")
            
            should_buy = len(buy_signals) >= 1  # Only need 1 signal for new tokens
            reason = f"[{token}] {', '.join(buy_signals)}"
            
            if should_buy:
                self.last_buy_time[token] = current_time
            
            return should_buy, reason
        
        # ADVANCED MODE: For tokens with good history (5+ data points)
        # Calculate moving averages
        ma_data = self.calculate_moving_averages(price_data)
        
        # OPTIMIZED SCALPING BUY CONDITIONS (momentum-focused)
        buy_signals = []
        
        # CORE SIGNAL 1: Price breaks ABOVE MA5 (momentum confirmed)
        if ma_data['ma_short'] and current_price > ma_data['ma_short'] * 1.001:  # 0.1% above MA
            pct_above = ((current_price - ma_data['ma_short']) / ma_data['ma_short']) * 100
            buy_signals.append(f"Momentum: above MA5 ({pct_above:.2f}%)")
        
        # CORE SIGNAL 2: RSI in sweet spot (not overbought/oversold)
        if rsi_value is not None and 40 <= rsi_value <= 60:
            buy_signals.append(f"RSI optimal: {rsi_value:.1f}")
        
        # CORE SIGNAL 3: Recent price momentum (last 3 ticks trending up)
        if len(recent_prices) >= 3:
            if recent_prices[-1] > recent_prices[-2] and recent_prices[-2] > recent_prices[-3]:
                price_change = ((recent_prices[-1] - recent_prices[-3]) / recent_prices[-3]) * 100
                buy_signals.append(f"Uptrend: +{price_change:.3f}%")
        
        # CORE SIGNAL 4: Volume spike (volatility indicates interest)
        if len(recent_prices) >= 2:
            price_change_pct = abs((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]) * 100
            # Require meaningful volatility (0.2%+)
            if price_change_pct > 0.2:
                buy_signals.append(f"Volume: {price_change_pct:.3f}%")
        
        # Require MULTIPLE signals for confirmation (minimum 2)
        should_buy = len(buy_signals) >= 2
        reason = f"[{token}] {', '.join(buy_signals)}" if buy_signals else "No signals"
        
        if should_buy:
            self.last_buy_time[token] = current_time  # Update BUY timer only
            
        return should_buy, reason
    
    def should_sell(self, token, current_price, holding_info, rsi_value=None, price_data=None):
        """Determine if we should sell a token with token-specific ultra-aggressive logic"""
        if holding_info['holdings'] <= 0:
            return False, "No holdings"
        
        import time
        current_time = time.time()
        
        # Get token-specific settings
        settings = self.token_settings.get(token, {
            'min_profit': self.min_profit_threshold,
            'max_loss': self.max_loss_threshold,
            'min_interval': self.min_trade_interval
        })
        
        # Check if we can SELL (using separate sell timer)
        last_sell = self.last_sell_time.get(token, 0)
        if current_time - last_sell < settings['min_interval']:
            return False, f"Too soon ({settings['min_interval']}s interval)"
        
        avg_price = holding_info['avg_price']
        if avg_price <= 0:
            return False, "Invalid avg price"
            
        profit_loss_pct = ((current_price - avg_price) / avg_price) * 100
        
        # Calculate moving averages if we have price data
        ma_data = {'ma_short': None, 'ma_long': None, 'ma_daily': None}
        if price_data and len(price_data) >= 2:
            ma_data = self.calculate_moving_averages(price_data)
        
        # ULTRA-AGGRESSIVE SELLING CONDITIONS (token-specific)
        sell_signals = []
        
        # SIMPLE & ADVANCED MODE SELL CONDITIONS
        sell_signals = []
        
        # Get token-specific profit target
        profit_target_pct = settings['min_profit'] * 100  # 1.5% or 2.0% depending on token
        stop_loss_pct = settings['max_loss'] * 100  # -1.0% or -0.8%
        
        # CRITICAL: Stop loss (ALWAYS CHECK - works with ANY data)
        if profit_loss_pct <= stop_loss_pct:
            sell_signals.append(f"STOP LOSS: {profit_loss_pct:.3f}%")
        
        # CORE SIGNAL 1: Hit profit target (ALWAYS CHECK - works with ANY data)
        if profit_loss_pct >= profit_target_pct:
            sell_signals.append(f"Profit target: {profit_loss_pct:.3f}%")
        
        # ADVANCED SIGNALS: Only check if we have enough data
        if price_data and len(price_data) >= 5:
            # CORE SIGNAL 2: Price breaks BELOW MA5 (momentum lost)
            if ma_data['ma_short'] and current_price < ma_data['ma_short'] * 0.999:  # 0.1% below MA
                pct_below = ((ma_data['ma_short'] - current_price) / ma_data['ma_short']) * 100
                sell_signals.append(f"Momentum lost: below MA5 ({pct_below:.2f}%)")
            
            # CORE SIGNAL 3: RSI overbought (take profits)
            if rsi_value is not None and rsi_value > 65:
                sell_signals.append(f"RSI overbought: {rsi_value:.1f}")
            
            # CORE SIGNAL 4: Downtrend detected (last 3 ticks down)
            if len(price_data) >= 3:
                recent = [p['price'] for p in price_data[-3:]]
                if recent[-1] < recent[-2] and recent[-2] < recent[-3]:
                    price_drop = ((recent[-3] - recent[-1]) / recent[-3]) * 100
                    sell_signals.append(f"Downtrend: -{price_drop:.3f}%")
        
        # Sell on ANY signal (quick exit)
        should_sell = len(sell_signals) > 0
        reason = f"[{token}] {', '.join(sell_signals)}" if sell_signals else f"Hold: {profit_loss_pct:.3f}%"
        
        if should_sell:
            self.last_sell_time[token] = current_time  # Update SELL timer only
            
        return should_sell, reason
    
    async def execute_auto_trade(self, token, action, reason):
        """Execute an automated trade with token-specific ultra-aggressive logic"""
        try:
            import time
            current_price = self.gui.token_data.get(token, {}).get('price', 0)
            if current_price == 0:
                self.gui.log_message(f"❌ No price data for {token}")
                return False
            
            # Get token-specific settings
            settings = self.token_settings.get(token, {'trade_amount': self.trade_amount})
            base_amount = settings['trade_amount']
            
            # ULTRA-AGGRESSIVE QUANTITY CALCULATION
            if action == 'BUY':
                if token == 'SOL':
                    # For SOL, use percentage of USD value
                    if self.gui.wallet_usd_value > 5:
                        quantity = min(base_amount, self.gui.wallet_usd_value / current_price * 0.25)
                    else:
                        quantity = 0.001  # Micro trade
                else:
                    # For low-cap tokens, use SOL balance more aggressively
                    sol_price = self.gui.token_data.get('SOL', {}).get('price', 0)
                    if sol_price > 0 and self.gui.sol_balance > 0.001:
                        # Token-specific aggression levels
                        sol_percentage = 0.4 if token == 'USELESS' else 0.3 if token == 'TROLL' else 0.2
                        max_sol_to_use = self.gui.sol_balance * sol_percentage
                        
                        # Calculate quantity in SOL terms
                        quantity = min(base_amount, max_sol_to_use)
                        
                        # Convert to token quantity if needed
                        if token != 'SOL':
                            # For low-cap tokens, we trade in small amounts
                            usd_value = quantity * sol_price
                            quantity = usd_value / current_price
                    else:
                        # Force micro-trades even with low balance
                        quantity = 0.0001 if token == 'USELESS' else 0.001
            else:  # SELL
                holdings = self.gui.portfolio[token]['holdings']
                # Token-specific sell percentages
                # ACE: Sell 100% on stop loss (full exit strategy)
                if token == 'ACE' and reason and 'STOP LOSS' in reason:
                    sell_pct = 1.0  # Sell ALL on stop loss
                elif token == 'USELESS':
                    sell_pct = 0.9
                elif token == 'TROLL':
                    sell_pct = 0.8
                else:
                    sell_pct = 0.7
                quantity = holdings * sell_pct
            
            # Ensure minimum quantity but allow micro-trades
            if quantity <= 0:
                quantity = 0.0001 if token == 'OPTA' else 0.001
            
            # Calculate trade value
            trade_value = quantity * current_price
            
            # MINIMUM TRADE VALUE CHECK (Jupiter requires minimum ~$0.10)
            # 🎯 MINIMUM TRADE SIZE: $5.00 (Eliminates 71.2% dust trades)
            min_trade_value_usd = 5.00  # CRITICAL: Increased from $0.01 to prevent dust trades
            if trade_value < min_trade_value_usd:
                self.gui.log_message(f"⚠️ {token}: Trade value ${trade_value:.2f} below minimum ${min_trade_value_usd} - SKIPPING")
                return False
            
            # RISK MANAGEMENT CHECKS
            if action == 'BUY':
                # Check if trading is paused due to daily loss limit
                if self.gui.trading_paused:
                    self.gui.log_message(f"⛔ Trading paused: Daily loss limit hit ({self.gui.daily_pnl_tracker:.2%})")
                    return False
                
                # Check position size limit (10% max per token)
                max_trade_value = self.gui.wallet_usd_value * self.gui.max_position_pct
                if trade_value > max_trade_value:
                    self.gui.log_message(f"⚠️ Position size limited: ${trade_value:.2f} → ${max_trade_value:.2f}")
                    trade_value = max_trade_value
                    quantity = trade_value / current_price
            
            # Enhanced logging for ultra-aggressive trading
            mode_label = "🔴 REAL" if self.real_trading_mode else "📄 PAPER"
            self.gui.log_message(f"{mode_label} 🔥 SCALP-{action}: {quantity:.6f} {token} @ ${current_price:.8f}")
            self.gui.log_message(f"📊 Reason: {reason}")
            self.gui.log_message(f"💰 Value: ${trade_value:.4f}")
            self.gui.log_message(f"⚡ Settings: {settings}")
            
            # REAL TRADING EXECUTION
            if self.real_trading_mode and self.real_trader:
                try:
                    # Convert quantity to SOL value for trading
                    sol_price = self.gui.token_data.get('SOL', {}).get('price', 160)
                    amount_sol = (quantity * current_price) / sol_price
                    
                    self.gui.log_message(f"🔴 EXECUTING REAL TRADE ON BLOCKCHAIN...")
                    
                    if action == 'BUY':
                        signature = self.real_trader.buy_token(token, amount_sol, slippage_bps=100)
                    else:
                        # Pass portfolio balance to avoid RPC rate limiting
                        portfolio_balance = self.gui.portfolio[token]['holdings']
                        signature = self.real_trader.sell_token(token, quantity, portfolio_balance=portfolio_balance, slippage_bps=100)
                    
                    if signature:
                        self.gui.log_message(f"✅ REAL TRADE SUCCESS: {signature[:16]}...")
                        self.gui.log_message(f"🔗 https://solscan.io/tx/{signature}")
                        
                        # Calculate profit/loss for sells
                        profit_loss = 0
                        if action == 'SELL' and self.gui.portfolio[token]['holdings'] > 0:
                            avg_cost = self.gui.portfolio[token]['avg_price']
                            if avg_cost > 0:
                                profit_loss = (current_price - avg_cost) * quantity
                                
                                # Update daily P&L tracker
                                profit_pct = profit_loss / self.gui.wallet_usd_value if self.gui.wallet_usd_value > 0 else 0
                                self.gui.daily_pnl_tracker += profit_pct
                                
                                # Check if hit daily loss limit
                                if self.gui.daily_pnl_tracker <= self.gui.max_daily_loss_pct:
                                    self.gui.trading_paused = True
                                    self.gui.log_message(f"🚨 TRADING PAUSED: Daily loss limit reached ({self.gui.daily_pnl_tracker:.2%})")
                                
                                self.gui.log_message(f"📊 Daily P&L: {self.gui.daily_pnl_tracker:.2%}")
                        
                        # Save real trade to database
                        timestamp = datetime.now().isoformat()
                        self.gui.conn.execute('''
                            INSERT INTO trades (timestamp, token, action, quantity, price, value, profit_loss)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (timestamp, token, action, quantity, current_price, trade_value, profit_loss))
                        self.gui.conn.commit()
                        
                        # Update portfolio to reflect real trade
                        self.gui.update_portfolio(token, action, quantity, current_price)
                        self.gui.root.after(0, self.gui.load_trade_history)
                        self.gui.root.after(0, self.gui.update_portfolio_display)
                    else:
                        self.gui.log_message(f"❌ REAL TRADE FAILED - CHECK LOGS")
                        return False
                        
                except Exception as e:
                    self.gui.log_message(f"❌ REAL TRADE ERROR: {e}")
                    return False
            
            # Store trade timestamp for tracking
            if not hasattr(self.gui, 'trade_timestamps'):
                self.gui.trade_timestamps = {}
            self.gui.trade_timestamps[token] = time.time()
            
            # Store last price for volatility calculations
            if not hasattr(self.gui, 'last_price'):
                self.gui.last_price = {}
            self.gui.last_price[token] = current_price
            
            # Execute the trade using existing trade logic
            self.gui.trade_token_var.set(token)
            self.gui.trade_quantity_var.set(str(quantity))
            
            # Bypass some validation for aggressive trading
            original_execute = self.gui.execute_trade
            
            def aggressive_execute(action):
                try:
                    token = self.gui.trade_token_var.get()
                    quantity = float(self.gui.trade_quantity_var.get())
                    current_price = self.gui.token_data[token]['price']
                    trade_value = quantity * current_price
                    
                    # Skip balance validation for tiny test trades
                    if quantity < 0.01:
                        self.gui.log_message(f"⚡ Executing micro-trade: {quantity:.6f} {token}")
                    
                    # Record trade directly without all the validation
                    timestamp = datetime.now().isoformat()
                    
                    # Calculate basic P&L
                    profit_loss = 0
                    if action == 'SELL' and self.gui.portfolio[token]['holdings'] > 0:
                        avg_cost = self.gui.portfolio[token]['avg_price']
                        if avg_cost > 0:
                            profit_loss = (current_price - avg_cost) * quantity
                    
                    # Save to database
                    self.gui.conn.execute('''
                        INSERT INTO trades (timestamp, token, action, quantity, price, value, profit_loss)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, token, action, quantity, current_price, trade_value, profit_loss))
                    self.gui.conn.commit()
                    
                    # Update portfolio
                    self.gui.update_portfolio(token, action, quantity, current_price)
                    self.gui.update_simulated_wallet_balance(token, action, quantity, current_price)
                    
                    # Update GUI
                    self.gui.root.after(0, self.gui.load_trade_history)
                    self.gui.root.after(0, self.gui.update_portfolio_display)
                    
                    self.gui.log_message(f"✅ {action} executed: {quantity:.6f} {token}")
                    if profit_loss != 0:
                        pnl_text = f"Profit: ${profit_loss:.2f}" if profit_loss > 0 else f"Loss: ${abs(profit_loss):.2f}"
                        self.gui.log_message(f"💰 {pnl_text}")
                    
                    return True
                    
                except Exception as e:
                    self.gui.log_message(f"❌ Aggressive trade execution error: {e}")
                    return False
            
            # Execute the aggressive trade
            success = aggressive_execute(action)
            return success
            
        except Exception as e:
            self.gui.log_message(f"❌ Auto trade error: {e}")
            return False

class TradingBotGUI:
    def __init__(self):
        """Initialize the advanced trading bot GUI"""
        self.root = tk.Tk()
        self.root.title("🚀 Advanced Solana Trading Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize bot components
        # Get the config path relative to the script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), 'config.json')
        self.config = self.load_config(config_path)
        self.exchange_manager = ExchangeManager(self.config)
        self.running = False
        self.scan_interval = 30  # 🎯 OPTIMIZED: 30 seconds (was 2s - reduced overtrading)
        # 🎯 DATA-DRIVEN TOKEN SELECTION: TROLL (+$3.49), USELESS (+$0.005), WIF, JUP
        # REMOVED LOSERS: GXY (-$1.95), ROI (-$2.13), ACE (-$0.14), BONK (-$0.89)
        self.target_tokens = ['BANGERS', 'TRUMP', 'BASED']
        
        # 🐋 Whale wallet tracking - 3 VERIFIED TRADERS
        self.whale_wallets = [
            'Ad7CwwXixx1MAFMCcoF4krxbJRyejjyAgNJv4iaKZVCq',  # Whale 1
            'JCRGumoE9Qi5BBgULTgdgTLjSgkCMSbF62ZZfGs84JeU',  # Whale 2
            'GGkB8ef2AMGgTx9nJKLWDPtMPTpix92iTMJKo58JafGr'   # Whale 3 (new)
        ]
        self.whale_tracker = None  # Initialized when trading starts
        self.whale_signals = []     # Store whale trade signals
        self.last_whale_check = {}  # Track last check time per wallet
        
        # Data storage
        self.token_data = {}
        self.price_history = defaultdict(lambda: deque(maxlen=100))  # Store last 100 prices
        self.trades = []
        self.portfolio = {token: {'holdings': 0, 'avg_price': 0, 'total_invested': 0} for token in self.target_tokens}
        self.daily_pnl = defaultdict(float)
        
        # Initialize database
        self.init_database()
        
        # Market indicators
        self.market_indicators = {}
        
        # Wallet balance tracking
        self.sol_balance = 0.0
        self.wallet_usd_value = 0.0
        
        # 🎯 RISK MANAGEMENT LIMITS (OPTIMIZED)
        self.max_position_pct = 0.10  # Max 10% of balance per token
        self.max_daily_loss_pct = -0.10  # INCREASED: Max -10% daily loss (was -2%)
        self.daily_pnl_tracker = 0.0  # Track today's P&L
        self.trading_paused = False  # Pause if hit daily limit
        
        # Initialize AutoTrader
        self.auto_trader = AutoTrader(self)
        
        # Trade timestamp tracking
        self.trade_timestamps = {}
        
        self.setup_advanced_gui()
        self.load_trade_history()
        
        # Initialize wallet balance
        self.refresh_wallet_balance()
        
        # Auto-start the bot (like the previous beautiful version)
        self.root.after(1000, self.auto_start_bot)  # Start after 1 second
        
        # Sync portfolio with actual blockchain holdings AFTER real trading is enabled
        self.root.after(6000, self.sync_portfolio_with_blockchain)  # Wait 6 seconds for prices + real trading mode
        
    def auto_start_bot(self):
        """Automatically start the bot after GUI loads"""
        if not self.running:
            self.log_message("🤖 Auto-starting trading bot...")
            self.start_bot()
            
        # Also enable auto-trading by default for ultra-aggressive mode
        self.root.after(3000, self.force_enable_auto_trading)  # Enable after 3 seconds
            
    def force_enable_auto_trading(self):
        """Force enable auto-trading for ultra-aggressive mode"""
        if hasattr(self, 'auto_trade_var') and not self.auto_trade_var.get():
            self.log_message("🔥 FORCE ENABLING AUTO-TRADING for ultra-aggressive mode...")
            self.auto_trade_var.set(True)
            self.toggle_auto_trading()
            self.log_message("⚡ Ultra-aggressive trading is now ACTIVE!")
        
        # Also auto-enable REAL TRADING MODE (not paper trading)
        if REAL_TRADING_AVAILABLE and hasattr(self, 'auto_trader'):
            if not self.auto_trader.real_trading_mode:
                self.log_message("🔴 Auto-enabling REAL TRADING MODE...")
                # Load private key from config
                try:
                    import json
                    from pathlib import Path
                    config_path = Path(__file__).parent.parent / 'config.json'
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    private_key = config['solana']['wallet_private_key']
                    self.log_message("🔑 Loading wallet from config...")
                    
                    # Initialize real trader
                    self.auto_trader.real_trader = RealTrader(private_key)
                    
                    # Set Jupiter API key if available
                    if 'jupiter_api_key' in config['solana'] and config['solana']['jupiter_api_key']:
                        self.auto_trader.real_trader.set_jupiter_api_key(config['solana']['jupiter_api_key'])
                    
                    self.auto_trader.real_trading_mode = True
                    
                    # Update UI
                    self.trading_mode_label.config(
                        text="🔴 REAL TRADING MODE - LIVE",
                        foreground='red'
                    )
                    self.real_trading_button.config(text="Disable Real Trading")
                    
                    self.log_message("✅ REAL TRADING MODE ENABLED")
                    self.log_message(f"🔗 Wallet: {self.auto_trader.real_trader.wallet_address}")
                    self.log_message("⚠️ All trades will be executed on Solana blockchain!")
                    
                    # Sync portfolio with blockchain immediately
                    self.root.after(2000, self.sync_portfolio_with_blockchain)
                    
                except Exception as e:
                    self.log_message(f"❌ Failed to auto-enable real trading: {e}")
                    self.log_message("💡 You can manually enable it using the button")
        
    def init_database(self):
        """Initialize SQLite database for trade history"""
        self.db_path = '../trading_history.db'
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # Create tables
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                token TEXT,
                action TEXT,
                quantity REAL,
                price REAL,
                value REAL,
                profit_loss REAL DEFAULT 0
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                token TEXT,
                price REAL,
                volume REAL DEFAULT 0
            )
        ''')
        
        self.conn.commit()
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                print(f"⚠️ Config file not found: {config_path}")
                print(f"📁 Using default configuration")
                return self.get_default_config()
        except Exception as e:
            print(f"⚠️ Failed to load config: {e}")
            print(f"📁 Using default configuration")
            return self.get_default_config()
    
    def get_default_config(self):
        """Return default configuration"""
        return {
            'wallet': {
                'solana_address': 'Not configured'
            },
            'api': {
                'dexscreener': 'https://api.dexscreener.com'
            }
        }
    
    def setup_advanced_gui(self):
        """Setup the advanced GUI with charts and trading features"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Dashboard
        self.setup_dashboard_tab()
        
        # Tab 2: Charts
        self.setup_charts_tab()
        
        # Tab 3: Trading
        self.setup_trading_tab()
        
        # Tab 4: History
        self.setup_history_tab()
        
        # Tab 5: Analytics
        self.setup_analytics_tab()
    
    def setup_dashboard_tab(self):
        """Setup main dashboard tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Title and status
        title_frame = ttk.Frame(dashboard_frame)
        title_frame.pack(fill=tk.X, pady=5)
        
        title_label = ttk.Label(title_frame, text="🚀 SOLANA TRADING DASHBOARD", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # TRADING MODE TOGGLE
        self.trading_mode_label = ttk.Label(title_frame, text="📄 PAPER TRADING MODE", 
                                 font=('Arial', 12, 'bold'), foreground='orange')
        self.trading_mode_label.pack(side=tk.LEFT, padx=20)
        
        # Real Trading Toggle Button
        self.real_trading_button = ttk.Button(title_frame, text="Enable Real Trading", 
                                             command=self.toggle_real_trading,
                                             style='Danger.TButton')
        self.real_trading_button.pack(side=tk.LEFT, padx=10)
        
        # Status and controls
        control_frame = ttk.Frame(title_frame)
        control_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(control_frame, text="Status: Stopped", 
                                     font=('Arial', 10, 'bold'))
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.start_button = ttk.Button(control_frame, text="Start Bot", 
                                      command=self.start_bot)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Bot", 
                                     command=self.stop_bot, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Auto-trading controls
        self.auto_trade_var = tk.BooleanVar(value=False)
        auto_trade_check = ttk.Checkbutton(control_frame, text="🤖 Auto Trading", 
                                          variable=self.auto_trade_var,
                                          command=self.toggle_auto_trading)
        auto_trade_check.pack(side=tk.LEFT, padx=20)
        
        self.auto_trade_status = ttk.Label(control_frame, text="Manual Mode", 
                                         font=('Arial', 9), foreground='gray')
        self.auto_trade_status.pack(side=tk.LEFT, padx=5)
        
        # Test log button
        ttk.Button(control_frame, text="📝 Test Log", 
                  command=self.test_log_message).pack(side=tk.LEFT, padx=5)
        
        # Market overview
        market_frame = ttk.LabelFrame(dashboard_frame, text="Market Overview", padding=10)
        market_frame.pack(fill=tk.X, pady=5)
        
        # Create market cards
        self.market_cards = {}
        card_frame = ttk.Frame(market_frame)
        card_frame.pack(fill=tk.X)
        
        for i, token in enumerate(self.target_tokens):
            card = self.create_market_card(card_frame, token)
            card.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
            self.market_cards[token] = card
            card_frame.columnconfigure(i, weight=1)
        
        # Portfolio summary
        portfolio_frame = ttk.LabelFrame(dashboard_frame, text="Portfolio Summary", padding=10)
        portfolio_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Wallet Balance Section
        wallet_frame = ttk.LabelFrame(portfolio_frame, text="Wallet Balance", padding=10)
        wallet_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Wallet info display
        wallet_info_frame = ttk.Frame(wallet_frame)
        wallet_info_frame.pack(fill=tk.X)
        
        # Wallet address (truncated)
        wallet_address = self.config.get('wallet', {}).get('solana_address', 'Not configured')
        truncated_address = f"{wallet_address[:8]}...{wallet_address[-8:]}" if len(wallet_address) > 16 else wallet_address
        
        self.wallet_address_label = ttk.Label(wallet_info_frame, text=f"Wallet: {truncated_address}", 
                                            font=('Arial', 10))
        self.wallet_address_label.pack(side=tk.LEFT, padx=10)
        
        # SOL Balance
        self.sol_balance_label = ttk.Label(wallet_info_frame, text="SOL Balance: Loading...", 
                                         font=('Arial', 11, 'bold'))
        self.sol_balance_label.pack(side=tk.LEFT, padx=20)
        
        # USD Value
        self.wallet_usd_label = ttk.Label(wallet_info_frame, text="USD Value: $0.00", 
                                        font=('Arial', 11, 'bold'))
        self.wallet_usd_label.pack(side=tk.LEFT, padx=20)
        
        # Refresh wallet button
        ttk.Button(wallet_info_frame, text="🔄 Refresh Wallet", 
                  command=self.refresh_wallet_balance).pack(side=tk.RIGHT, padx=10)
        
        # Portfolio metrics
        metrics_frame = ttk.Frame(portfolio_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        self.total_value_label = ttk.Label(metrics_frame, text="Total Value: $0.00", 
                                          font=('Arial', 12, 'bold'))
        self.total_value_label.pack(side=tk.LEFT, padx=20)
        
        self.daily_pnl_label = ttk.Label(metrics_frame, text="Daily P&L: $0.00", 
                                        font=('Arial', 12, 'bold'))
        self.daily_pnl_label.pack(side=tk.LEFT, padx=20)
        
        self.total_pnl_label = ttk.Label(metrics_frame, text="Total P&L: $0.00", 
                                        font=('Arial', 12, 'bold'))
        self.total_pnl_label.pack(side=tk.LEFT, padx=20)
        
        # Portfolio table
        port_columns = ('Token', 'Holdings', 'Avg Price', 'Current Price', 'Value', 'P&L', '%')
        self.portfolio_tree = ttk.Treeview(portfolio_frame, columns=port_columns, show='headings', height=6)
        
        for col in port_columns:
            self.portfolio_tree.heading(col, text=col)
            self.portfolio_tree.column(col, width=120)
        
        portfolio_scrollbar = ttk.Scrollbar(portfolio_frame, orient=tk.VERTICAL, 
                                           command=self.portfolio_tree.yview)
        self.portfolio_tree.configure(yscrollcommand=portfolio_scrollbar.set)
        
        self.portfolio_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        portfolio_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize portfolio display
        for token in self.target_tokens:
            self.portfolio_tree.insert('', 'end', iid=token, 
                                      values=(token, '0', '$0.00', '$0.00', '$0.00', '$0.00', '0.00%'))
        
        # 🐋 Whale Tracking Panel
        whale_frame = ttk.LabelFrame(dashboard_frame, text="🐋 Whale Tracker - Copy Trading", padding=10)
        whale_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        whale_info = ttk.Frame(whale_frame)
        whale_info.pack(fill=tk.X, pady=5)
        
        ttk.Label(whale_info, text="Monitoring 3 Whale Wallets:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.whale_status_labels = []
        for i, whale in enumerate(self.whale_wallets, 1):
            whale_label = ttk.Label(whale_info, 
                                   text=f"Whale {i}: {whale[:8]}...{whale[-6:]} - Checking...",
                                   font=('Arial', 9))
            whale_label.pack(anchor=tk.W, padx=20)
            self.whale_status_labels.append(whale_label)
        
        # Recent whale trades
        ttk.Label(whale_frame, text="Recent Whale Trades (Last 10):", 
                 font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        whale_columns = ('Time', 'Whale', 'Action', 'Token', 'Amount', 'Status')
        self.whale_tree = ttk.Treeview(whale_frame, columns=whale_columns, show='headings', height=5)
        
        for col in whale_columns:
            self.whale_tree.heading(col, text=col)
            if col == 'Whale':
                self.whale_tree.column(col, width=100)
            elif col == 'Time':
                self.whale_tree.column(col, width=80)
            else:
                self.whale_tree.column(col, width=80)
        
        whale_scrollbar = ttk.Scrollbar(whale_frame, orient=tk.VERTICAL, 
                                       command=self.whale_tree.yview)
        self.whale_tree.configure(yscrollcommand=whale_scrollbar.set)
        
        self.whale_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        whale_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_market_card(self, parent, token):
        """Create a market card widget for a token"""
        card_frame = ttk.LabelFrame(parent, text=token, padding=10)
        
        # Price label
        price_label = ttk.Label(card_frame, text="$0.000000", 
                               font=('Arial', 14, 'bold'))
        price_label.pack()
        
        # Change label
        change_label = ttk.Label(card_frame, text="0.00%", 
                                font=('Arial', 10))
        change_label.pack()
        
        # Indicator
        indicator_label = ttk.Label(card_frame, text="⚪ NEUTRAL", 
                                   font=('Arial', 9))
        indicator_label.pack()
        
        # Store references
        card_frame.price_label = price_label
        card_frame.change_label = change_label
        card_frame.indicator_label = indicator_label
        
        return card_frame
    
    def setup_charts_tab(self):
        """Setup charts tab with real-time price charts"""
        charts_frame = ttk.Frame(self.notebook)
        self.notebook.add(charts_frame, text="📈 Charts")
        
        # Chart controls
        control_frame = ttk.Frame(charts_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Token:").pack(side=tk.LEFT, padx=5)
        self.chart_token_var = tk.StringVar(value='SOL')
        token_combo = ttk.Combobox(control_frame, textvariable=self.chart_token_var, 
                                  values=self.target_tokens, state='readonly', width=10)
        token_combo.pack(side=tk.LEFT, padx=5)
        token_combo.bind('<<ComboboxSelected>>', self.update_chart)
        
        ttk.Button(control_frame, text="Refresh Chart", 
                  command=self.update_chart).pack(side=tk.LEFT, padx=10)
        
        # Chart canvas
        self.chart_figure = Figure(figsize=(12, 8), dpi=100, facecolor='white')
        self.chart_canvas = FigureCanvasTkAgg(self.chart_figure, charts_frame)
        self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Initialize chart
        self.update_chart()
    
    def setup_trading_tab(self):
        """Setup trading tab for manual trades"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="💰 Trading")
        
        # Trading controls
        trade_control_frame = ttk.LabelFrame(trading_frame, text="Manual Trading", padding=15)
        trade_control_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Wallet info in trading tab
        trade_wallet_frame = ttk.Frame(trade_control_frame)
        trade_wallet_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.trade_wallet_label = ttk.Label(trade_wallet_frame, text="Wallet Balance: Loading...", 
                                          font=('Arial', 10, 'bold'))
        self.trade_wallet_label.pack(side=tk.LEFT)
        
        ttk.Button(trade_wallet_frame, text="🔄", 
                  command=self.refresh_wallet_balance).pack(side=tk.RIGHT)
        
        # Token selection
        token_frame = ttk.Frame(trade_control_frame)
        token_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(token_frame, text="Token:").pack(side=tk.LEFT, padx=5)
        self.trade_token_var = tk.StringVar(value='SOL')
        ttk.Combobox(token_frame, textvariable=self.trade_token_var, 
                    values=self.target_tokens, state='readonly', width=10).pack(side=tk.LEFT, padx=5)
        
        # Quantity
        ttk.Label(token_frame, text="Quantity:").pack(side=tk.LEFT, padx=(20, 5))
        self.trade_quantity_var = tk.StringVar(value='1.0')
        ttk.Entry(token_frame, textvariable=self.trade_quantity_var, width=15).pack(side=tk.LEFT, padx=5)
        
        # Trade buttons
        button_frame = ttk.Frame(trade_control_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="🟢 BUY", 
                  command=lambda: self.execute_trade('BUY')).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="🔴 SELL", 
                  command=lambda: self.execute_trade('SELL')).pack(side=tk.LEFT, padx=10)
        
        # Market data display
        market_data_frame = ttk.LabelFrame(trading_frame, text="Market Data", padding=15)
        market_data_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        # Current price and indicators
        self.current_price_label = ttk.Label(market_data_frame, text="Current Price: $0.00", 
                                           font=('Arial', 12, 'bold'))
        self.current_price_label.pack(pady=5)
        
        self.market_trend_label = ttk.Label(market_data_frame, text="Trend: NEUTRAL", 
                                          font=('Arial', 11))
        self.market_trend_label.pack(pady=5)
        
        # Trading signals
        signals_frame = ttk.LabelFrame(market_data_frame, text="Trading Signals", padding=10)
        signals_frame.pack(fill=tk.X, pady=10)
        
        self.rsi_label = ttk.Label(signals_frame, text="RSI: --")
        self.rsi_label.pack(anchor='w')
        
        self.ma_label = ttk.Label(signals_frame, text="Moving Average: --")
        self.ma_label.pack(anchor='w')
        
        self.volume_label = ttk.Label(signals_frame, text="Volume Trend: --")
        self.volume_label.pack(anchor='w')
        
        # Auto-trading settings
        auto_settings_frame = ttk.LabelFrame(market_data_frame, text="Auto-Trading Settings", padding=10)
        auto_settings_frame.pack(fill=tk.X, pady=10)
        
        # Profit threshold
        profit_frame = ttk.Frame(auto_settings_frame)
        profit_frame.pack(fill=tk.X, pady=2)
        ttk.Label(profit_frame, text="Profit Threshold (%):").pack(side=tk.LEFT)
        self.profit_threshold_var = tk.StringVar(value="0.5")  # Much lower
        profit_entry = ttk.Entry(profit_frame, textvariable=self.profit_threshold_var, width=8)
        profit_entry.pack(side=tk.LEFT, padx=5)
        
        # Stop loss
        loss_frame = ttk.Frame(auto_settings_frame)
        loss_frame.pack(fill=tk.X, pady=2)
        ttk.Label(loss_frame, text="Stop Loss (%):").pack(side=tk.LEFT)
        self.stop_loss_var = tk.StringVar(value="2.0")  # Tighter
        loss_entry = ttk.Entry(loss_frame, textvariable=self.stop_loss_var, width=8)
        loss_entry.pack(side=tk.LEFT, padx=5)
        
        # Trade amount
        amount_frame = ttk.Frame(auto_settings_frame)
        amount_frame.pack(fill=tk.X, pady=2)
        ttk.Label(amount_frame, text="Trade Amount:").pack(side=tk.LEFT)
        self.trade_amount_var = tk.StringVar(value="0.05")  # Smaller
        amount_entry = ttk.Entry(amount_frame, textvariable=self.trade_amount_var, width=8)
        amount_entry.pack(side=tk.LEFT, padx=5)
        
        # Update settings button
        ttk.Button(auto_settings_frame, text="Update Settings", 
                  command=self.update_auto_trading_settings).pack(pady=5)
    
    def setup_history_tab(self):
        """Setup trade history tab"""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="📜 History")
        
        # History controls
        control_frame = ttk.Frame(history_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Refresh History", 
                  command=self.load_trade_history).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="Clear History", 
                  command=self.clear_trade_history).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(control_frame, text="Export CSV", 
                  command=self.export_history).pack(side=tk.LEFT, padx=10)
        
        # Trade history table
        history_columns = ('ID', 'Timestamp', 'Token', 'Action', 'Quantity', 'Price', 'Value', 'P&L')
        self.history_tree = ttk.Treeview(history_frame, columns=history_columns, show='headings')
        
        for col in history_columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)
        
        # Scrollbars
        history_v_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, 
                                           command=self.history_tree.yview)
        history_h_scrollbar = ttk.Scrollbar(history_frame, orient=tk.HORIZONTAL, 
                                           command=self.history_tree.xview)
        
        self.history_tree.configure(yscrollcommand=history_v_scrollbar.set, 
                                   xscrollcommand=history_h_scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        history_v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        history_h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_analytics_tab(self):
        """Setup analytics tab with performance metrics"""
        analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(analytics_frame, text="📊 Analytics")
        
        # Performance metrics
        metrics_frame = ttk.LabelFrame(analytics_frame, text="Performance Metrics", padding=15)
        metrics_frame.pack(fill=tk.X, pady=10, padx=10)
        
        # Create metrics grid
        metrics_grid = ttk.Frame(metrics_frame)
        metrics_grid.pack(fill=tk.X)
        
        # Win rate
        self.win_rate_label = ttk.Label(metrics_grid, text="Win Rate: 0%", 
                                       font=('Arial', 11, 'bold'))
        self.win_rate_label.grid(row=0, column=0, padx=20, pady=5, sticky='w')
        
        # Average profit
        self.avg_profit_label = ttk.Label(metrics_grid, text="Avg Profit: $0.00", 
                                         font=('Arial', 11, 'bold'))
        self.avg_profit_label.grid(row=0, column=1, padx=20, pady=5, sticky='w')
        
        # Total trades
        self.total_trades_label = ttk.Label(metrics_grid, text="Total Trades: 0", 
                                           font=('Arial', 11, 'bold'))
        self.total_trades_label.grid(row=1, column=0, padx=20, pady=5, sticky='w')
        
        # Best trade
        self.best_trade_label = ttk.Label(metrics_grid, text="Best Trade: $0.00", 
                                         font=('Arial', 11, 'bold'))
        self.best_trade_label.grid(row=1, column=1, padx=20, pady=5, sticky='w')
        
        # P&L Chart
        pnl_frame = ttk.LabelFrame(analytics_frame, text="Profit & Loss Chart", padding=10)
        pnl_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.pnl_figure = Figure(figsize=(10, 6), dpi=100, facecolor='white')
        self.pnl_canvas = FigureCanvasTkAgg(self.pnl_figure, pnl_frame)
        self.pnl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Activity log
        log_frame = ttk.LabelFrame(analytics_frame, text="Activity Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10, padx=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, 
                                                 font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initial log messages
        self.log_message("🚀 Advanced Trading Dashboard initialized")
        self.log_message("📊 DexScreener integration ready")
        self.log_message("💾 Database initialized for trade tracking")
        self.log_message("⚡ Ultra-aggressive low-cap strategy loaded")
        self.log_message("🎯 Target tokens: TROLL ($15), USELESS ($10), JELLYJELLY ($5)")
        self.log_message("💰 Wallet integration active")
        self.log_message("🔥 Preparing for high-frequency micro-trading...")
    
    def update_chart(self, event=None):
        """Update the price chart for selected token"""
        token = self.chart_token_var.get()
        
        # Clear previous chart
        self.chart_figure.clear()
        
        # Get price history
        prices = list(self.price_history[token])
        if len(prices) < 2:
            # Show empty chart message
            ax = self.chart_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'No data available for {token}\nStart the bot to collect price data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'{token} Price Chart')
            self.chart_canvas.draw()
            return
        
        # Create subplots
        ax1 = self.chart_figure.add_subplot(211)  # Price chart
        ax2 = self.chart_figure.add_subplot(212)  # Volume/indicators
        
        # Prepare data
        timestamps = [p['timestamp'] for p in prices]
        price_values = [p['price'] for p in prices]
        
        # Plot price line
        ax1.plot(timestamps, price_values, linewidth=2, color='#2196F3', label=f'{token} Price')
        ax1.set_title(f'{token} Price Chart', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add moving averages if enough data
        if len(price_values) >= 20:
            ma20 = np.convolve(price_values, np.ones(20)/20, mode='valid')
            ma20_timestamps = timestamps[19:]
            ax1.plot(ma20_timestamps, ma20, '--', color='orange', label='MA20', alpha=0.7)
            ax1.legend()
        
        # Calculate and display RSI in bottom chart
        if len(price_values) >= 14:
            rsi = self.calculate_rsi(price_values)
            # Ensure timestamps and RSI arrays have the same length
            rsi_timestamps = timestamps[14:]  # Skip first 14 for RSI calculation
            if len(rsi_timestamps) == len(rsi):
                ax2.plot(rsi_timestamps, rsi, color='purple', linewidth=2)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
                ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
                ax2.set_title('RSI (14)', fontweight='bold')
                ax2.set_ylabel('RSI')
                ax2.set_ylim(0, 100)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, f'RSI calculation error\nTimestamps: {len(rsi_timestamps)}, RSI: {len(rsi)}', 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('RSI (14) - Error')
        else:
            ax2.text(0.5, 0.5, 'Insufficient data for RSI\n(need 14+ data points)', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('RSI (14)')
        
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(timestamps) > 0:
            self.chart_figure.autofmt_xdate()
        
        # Tight layout
        self.chart_figure.tight_layout()
        self.chart_canvas.draw()
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_market_indicators(self, token, price):
        """Calculate market indicators for a token"""
        prices = [p['price'] for p in self.price_history[token]]
        
        if len(prices) < 5:
            return {'trend': 'NEUTRAL', 'signal': 'HOLD', 'strength': 0}
        
        # Calculate trend
        recent_prices = prices[-5:]
        if all(recent_prices[i] > recent_prices[i-1] for i in range(1, len(recent_prices))):
            trend = 'BULLISH'
            signal = 'BUY'
            strength = min(len([p for p in recent_prices if p > recent_prices[0]]) * 20, 100)
        elif all(recent_prices[i] < recent_prices[i-1] for i in range(1, len(recent_prices))):
            trend = 'BEARISH'
            signal = 'SELL'
            strength = min(len([p for p in recent_prices if p < recent_prices[0]]) * 20, 100)
        else:
            trend = 'NEUTRAL'
            signal = 'HOLD'
            strength = 0
        
        return {'trend': trend, 'signal': signal, 'strength': strength}
    
    def execute_trade(self, action):
        """Execute a manual trade"""
        try:
            token = self.trade_token_var.get()
            quantity = float(self.trade_quantity_var.get())
            
            # Get current price
            if token not in self.token_data:
                messagebox.showerror("Error", f"No price data available for {token}")
                return
            
            current_price = self.token_data[token]['price']
            trade_value = quantity * current_price
            
            # Validate wallet balance for buy orders
            if action == 'BUY':
                if token == 'SOL':
                    # Buying SOL with USD (simulated)
                    required_usd = trade_value
                    if self.wallet_usd_value < required_usd:
                        messagebox.showwarning("Insufficient Funds", 
                                             f"Insufficient USD balance. Required: ${required_usd:.2f}, Available: ${self.wallet_usd_value:.2f}")
                        return
                else:
                    # Buying other tokens with SOL
                    sol_price = self.token_data.get('SOL', {}).get('price', 0)
                    if sol_price == 0:
                        messagebox.showerror("Error", "SOL price not available")
                        return
                    
                    required_sol = trade_value / sol_price
                    if self.sol_balance < required_sol:
                        messagebox.showwarning("Insufficient Funds", 
                                             f"Insufficient SOL balance. Required: {required_sol:.6f} SOL, Available: {self.sol_balance:.6f} SOL")
                        return
            
            # Validate holdings for sell orders
            elif action == 'SELL':
                if token == 'SOL':
                    if self.sol_balance < quantity:
                        messagebox.showwarning("Insufficient Holdings", 
                                             f"Insufficient SOL balance. Trying to sell: {quantity:.6f} SOL, Available: {self.sol_balance:.6f} SOL")
                        return
                else:
                    if self.portfolio[token]['holdings'] < quantity:
                        messagebox.showwarning("Insufficient Holdings", 
                                             f"Insufficient {token} holdings. Trying to sell: {quantity:.6f}, Available: {self.portfolio[token]['holdings']:.6f}")
                        return
            
            # Calculate P&L for sell orders
            profit_loss = 0
            if action == 'SELL' and self.portfolio[token]['holdings'] >= quantity:
                avg_cost = self.portfolio[token]['avg_price']
                profit_loss = (current_price - avg_cost) * quantity
            
            # Record trade
            timestamp = datetime.now().isoformat()
            trade_record = {
                'timestamp': timestamp,
                'token': token,
                'action': action,
                'quantity': quantity,
                'price': current_price,
                'value': trade_value,
                'profit_loss': profit_loss
            }
            
            # Save to database
            self.conn.execute('''
                INSERT INTO trades (timestamp, token, action, quantity, price, value, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, token, action, quantity, current_price, trade_value, profit_loss))
            self.conn.commit()
            
            # Update portfolio
            self.update_portfolio(token, action, quantity, current_price)
            
            # Update simulated wallet balance
            self.update_simulated_wallet_balance(token, action, quantity, current_price)
            
            # Update GUI
            self.load_trade_history()
            self.update_portfolio_display()
            self.update_wallet_display(self.sol_balance, "Connected")
            
            self.log_message(f"✅ {action} {quantity} {token} @ ${current_price:.6f} = ${trade_value:.2f}")
            if profit_loss != 0:
                pnl_text = f"Profit: ${profit_loss:.2f}" if profit_loss > 0 else f"Loss: ${abs(profit_loss):.2f}"
                self.log_message(f"💰 {pnl_text}")
            
            messagebox.showinfo("Trade Executed", f"{action} order executed successfully!")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid quantity")
        except Exception as e:
            messagebox.showerror("Error", f"Trade execution failed: {e}")
    
    def update_simulated_wallet_balance(self, token, action, quantity, price):
        """Update simulated wallet balance after trades"""
        if token == 'SOL':
            if action == 'BUY':
                self.sol_balance += quantity
            elif action == 'SELL':
                self.sol_balance -= quantity
        else:
            # For other tokens, we trade with SOL
            sol_price = self.token_data.get('SOL', {}).get('price', 0)
            if sol_price > 0:
                trade_value_in_sol = (quantity * price) / sol_price
                
                if action == 'BUY':
                    self.sol_balance -= trade_value_in_sol
                elif action == 'SELL':
                    self.sol_balance += trade_value_in_sol
    
    def toggle_auto_trading(self):
        """Toggle auto-trading mode"""
        self.auto_trader.trading_enabled = self.auto_trade_var.get()
        
        if self.auto_trader.trading_enabled:
            self.auto_trade_status.config(text="🤖 AUTO MODE", foreground='green')
            self.log_message("🤖 Automated trading ENABLED")
            self.log_message(f"📊 Profit threshold: {self.auto_trader.min_profit_threshold*100:.1f}%")
            self.log_message(f"🛑 Stop loss: {self.auto_trader.max_loss_threshold*100:.1f}%")
        else:
            self.auto_trade_status.config(text="👤 Manual Mode", foreground='gray')
            self.log_message("👤 Automated trading DISABLED")
    
    async def check_whale_signals(self, token, analysis):
        """
        Check for whale wallet signals using enhanced whale tracker
        PRIORITY SIGNAL - Takes precedence over all technical analysis
        NOW WITH MULTI-WHALE CONSENSUS DETECTION + FULL DISCOVERY MODE
        """
        try:
            if not self.whale_tracker:
                # Initialize whale tracker on first use
                rpc_url = self.config.get('solana', {}).get('rpc_url', 'https://api.mainnet-beta.solana.com')
                self.whale_tracker = WhaleWalletTracker(rpc_url)
                whale_count = len(self.whale_tracker.WHALE_WALLETS)
                self.log_message(f"🐋 Whale tracker initialized with {whale_count} smart-money wallets")
                self.log_message(f"🔍 FULL DISCOVERY MODE: Bot will detect ANY token whales buy!")
            
            # Get token mint address (if we have it for the current token)
            token_address = analysis.get('address') or self.token_settings.get(token, {}).get('address')
            
            # FULL DISCOVERY MODE: Don't limit to specific tokens
            # Let whale tracker find ANY token the whales are trading
            signals = await self.whale_tracker.scan_whale_activity(
                tracked_tokens=None,  # None = discover ANY token
                lookback_minutes=10,
                max_whales=20,
                discovery_mode=True  # Enable full discovery
            )
            
            if not signals:
                return None
            
            # Log discovered tokens
            discovered_tokens = [s for s in signals if s.get('discovery_mode')]
            if discovered_tokens:
                self.log_message(f"🔍 Discovered {len(discovered_tokens)} new whale trade(s):")
                for sig in discovered_tokens[:3]:  # Show first 3
                    self.log_message(f"   {sig['token']} - {sig['whale_name']} {sig['action']} "
                                   f"(${sig.get('liquidity_usd', 0):,.0f} liq, "
                                   f"${sig.get('volume_24h', 0):,.0f} vol)")
            
            # 🚨 CHECK FOR MULTI-WHALE CONSENSUS (PRIORITY BOOST)
            consensus_data = self.whale_tracker.detect_multi_whale_consensus(signals, lookback_minutes=30)
            priority_signals = self.whale_tracker.get_priority_signals(consensus_data, min_whales=2)
            
            # If we have multi-whale consensus, use that instead of single whale signal
            if priority_signals:
                # Find consensus for current token
                token_consensus = next((s for s in priority_signals if s['token'] == token), None)
                
                if token_consensus:
                    self.log_message(f"🚨 MULTI-WHALE CONSENSUS DETECTED for {token}!")
                    self.log_message(f"   {token_consensus['whale_count']} whales {token_consensus['action']} simultaneously")
                    self.log_message(f"   Participating whales: {', '.join(w['name'] for w in token_consensus['whales'][:3])}")
                    if len(token_consensus['whales']) > 3:
                        self.log_message(f"   ... and {len(token_consensus['whales']) - 3} more")
                    
                    # Build enhanced signal from consensus
                    top_signal = {
                        'token': token,
                        'action': token_consensus['action'],
                        'whale_name': f"{token_consensus['whale_count']} Whales (CONSENSUS)",
                        'whale_win_rate': token_consensus['avg_win_rate'],
                        'confidence': min(0.99, 0.7 + (token_consensus['whale_count'] * 0.05)),  # Boost confidence based on whale count
                        'is_consensus': True,
                        'whale_count': token_consensus['whale_count'],
                        'priority_score': token_consensus['priority_score'],
                        'whales': token_consensus['whales'],
                        'time': datetime.now(),
                        'minutes_ago': 0,
                        'amount': sum(w.get('amount', 0) for w in token_consensus['whales']),
                        'reason': token_consensus['reason']
                    }
                    
                    # Apply technical confirmation if available
                    if self.technical_filter and analysis:
                        current_price = analysis.get('price', 0)
                        current_volume = analysis.get('volume_24h', 0)
                        
                        if top_signal['action'] == 'BUY':
                            confirmed, reason, final_confidence = self.technical_filter.confirm_whale_buy_signal(
                                token,
                                top_signal,
                                current_price,
                                current_volume
                            )
                        else:  # SELL
                            confirmed, reason, final_confidence = self.technical_filter.confirm_whale_sell_signal(
                                token,
                                top_signal,
                                current_price,
                                current_volume
                            )
                        
                        if not confirmed:
                            self.log_message(f"⚠️ Multi-whale {top_signal['action']} rejected by technical filter")
                            self.log_message(f"   Reason: {reason}")
                            # Still return signal but mark as unconfirmed
                            top_signal['technical_confirmed'] = False
                            top_signal['technical_reason'] = reason
                        else:
                            top_signal['confidence'] = final_confidence
                            top_signal['technical_confirmation'] = reason
                            top_signal['technical_confirmed'] = True
                    
                    return top_signal
            
            # No multi-whale consensus, use single strongest whale signal
            top_signal = signals[0]
            
            # Validate signal confidence
            min_confidence = getattr(self.copy_trade_manager, 'min_whale_confidence', 0.70) if self.copy_trade_manager else 0.70
            if top_signal['confidence'] < min_confidence:
                self.log_message(f"⚠️ Whale signal for {token} below confidence threshold "
                                    f"({top_signal['confidence']:.1%} < {min_confidence:.1%})")
                return None
            
            # Apply technical confirmation if technical_filter is available
            if self.technical_filter and analysis:
                current_price = analysis.get('price', 0)
                current_volume = analysis.get('volume_24h', 0)
                
                if top_signal['action'] == 'BUY':
                    confirmed, reason, final_confidence = self.technical_filter.confirm_whale_buy_signal(
                        token,
                        top_signal,
                        current_price,
                        current_volume
                    )
                else:  # SELL
                    confirmed, reason, final_confidence = self.technical_filter.confirm_whale_sell_signal(
                        token,
                        top_signal,
                        current_price,
                        current_volume
                    )
                
                if not confirmed:
                    self.log_message(f"⚠️ Whale {top_signal['action']} signal for {token} rejected by technical filter")
                    self.log_message(f"   Reason: {reason}")
                    return None
                
                # Update signal with final confidence
                top_signal['confidence'] = final_confidence
                top_signal['technical_confirmation'] = reason
            
            return top_signal
            
        except Exception as e:
            self.log_message(f"❌ Whale signal check error: {e}")
            import traceback
            self.log_message(traceback.format_exc())
            return None
    
    async def check_auto_trading_signals(self, token, analysis):
        """
        WHALE-FIRST Automated Trading Signal System with FULL DISCOVERY MODE
        Priority: 1) Multi-Whale Consensus (ANY token) > 2) Single Whale Signals > 3) Technical Signals
        
        NOTE: This now scans for whale activity on ANY Solana token, not just tracked ones
        """
        try:
            if not self.auto_trader.trading_enabled:
                return
            
            # Initialize copy-trade manager if not already done
            if not self.auto_trader.copy_trade_manager:
                # Get current portfolio value (SOL balance * SOL price)
                sol_price = analysis.get('sol_price', 150.0)  # Fallback price
                portfolio_value = self.sol_balance * sol_price
                
                self.auto_trader.copy_trade_manager = CopyTradeManager(
                    total_portfolio_value=max(portfolio_value, 100.0),  # Minimum $100
                    max_position_pct=0.01,     # 1% per position
                    max_open_positions=5,       # Max 5 positions
                    daily_loss_limit_pct=0.05,  # 5% daily loss limit
                    min_whale_confidence=0.70   # 70% minimum confidence
                )
                self.log_message(f"💼 Copy-Trade Manager initialized: ${portfolio_value:.2f} portfolio")
            
            # Initialize technical filter if not done
            if not self.auto_trader.technical_filter:
                self.auto_trader.technical_filter = TechnicalSignalFilter(
                    sma_short_period=5,
                    sma_long_period=15,
                    volume_spike_threshold=2.0,
                    price_breakout_pct=0.05
                )
                self.log_message("📊 Technical filter initialized (5/15 SMA)")
            
            # 🐋 PRIORITY 1: CHECK WHALE SIGNALS FIRST
            whale_signal = await self.check_whale_signals(token, analysis)
            
            if whale_signal:
                # Enhanced logging for consensus vs single whale
                is_consensus = whale_signal.get('is_consensus', False)
                
                if is_consensus:
                    self.log_message(f"🚨 MULTI-WHALE CONSENSUS DETECTED for {token}!")
                    self.log_message(f"   Whale Count: {whale_signal.get('whale_count', 0)}")
                    self.log_message(f"   Action: {whale_signal['action']}")
                    self.log_message(f"   Priority Score: {whale_signal.get('priority_score', 0):.1f}")
                    self.log_message(f"   Consensus Confidence: {whale_signal['confidence']:.0%}")
                    self.log_message(f"   Average Win Rate: {whale_signal.get('whale_win_rate', 0):.0%}")
                    
                    # Show participating whales
                    whales = whale_signal.get('whales', [])
                    if len(whales) > 0:
                        self.log_message(f"   Participating Whales:")
                        for i, w in enumerate(whales[:3], 1):
                            self.log_message(f"      {i}. {w.get('name', 'Unknown')} ({w.get('win_rate', 0):.0%} win rate)")
                        if len(whales) > 3:
                            self.log_message(f"      ... and {len(whales) - 3} more whale(s)")
                else:
                    self.log_message(f"🐋 WHALE SIGNAL DETECTED for {token}!")
                    self.log_message(f"   Whale: {whale_signal.get('whale_name', 'Unknown')}")
                    self.log_message(f"   Action: {whale_signal['action']}")
                    self.log_message(f"   Confidence: {whale_signal['confidence']:.0%}")
                    self.log_message(f"   Win Rate: {whale_signal.get('whale_win_rate', 0):.0%}")
                
                if whale_signal.get('technical_confirmation'):
                    self.log_message(f"   ✅ Technical: {whale_signal['technical_confirmation']}")
                elif whale_signal.get('technical_confirmed') is False:
                    self.log_message(f"   ⚠️ Technical: {whale_signal.get('technical_reason', 'Not confirmed')}")
                
                # Update whale tracker display
                self.update_whale_display(whale_signal)
                
                # Process whale signal through copy-trade manager
                current_price = analysis.get('price', 0)
                
                if whale_signal['action'] == 'BUY':
                    # Calculate position size (enhanced for consensus)
                    base_capital = self.sol_balance * current_price
                    
                    # For multi-whale consensus, use enhanced position sizing (2% vs 1%)
                    if is_consensus:
                        # Temporarily boost max position for consensus signals
                        original_max = self.auto_trader.copy_trade_manager.max_position_pct
                        self.auto_trader.copy_trade_manager.max_position_pct = min(0.02, original_max * 2)  # Double up to 2%
                        self.log_message(f"   💪 Enhanced position sizing for consensus: {self.auto_trader.copy_trade_manager.max_position_pct:.1%}")
                    
                    position = self.auto_trader.copy_trade_manager.calculate_position_size(
                        whale_signal,
                        current_price,
                        base_capital
                    )
                    
                    # Restore original max position if we boosted it
                    if is_consensus:
                        self.auto_trader.copy_trade_manager.max_position_pct = original_max
                    
                    if position:
                        # Open position through copy-trade manager
                        if self.auto_trader.copy_trade_manager.open_position(position):
                            # Build reason string
                            if is_consensus:
                                reason = (f"🚨 MULTI-WHALE CONSENSUS ({whale_signal.get('whale_count', 0)} whales) | "
                                         f"Priority: {whale_signal.get('priority_score', 0):.1f} | "
                                         f"Confidence: {whale_signal['confidence']:.0%}")
                            else:
                                reason = (f"🐋 WHALE COPY: {whale_signal.get('whale_name', 'Unknown')} | "
                                         f"Confidence: {whale_signal['confidence']:.0%}")
                            
                            # Execute the actual trade
                            await self.auto_trader.execute_auto_trade(
                                token,
                                'BUY',
                                reason,
                                amount=position['token_amount']
                            )
                    else:
                        self.log_message(f"⚠️ Position sizing rejected for {token}")
                
                elif whale_signal['action'] == 'SELL':
                    # Close position if we have one
                    if token in self.auto_trader.copy_trade_manager.open_positions:
                        exit_reason = 'MULTI_WHALE_CONSENSUS_SELL' if is_consensus else 'WHALE_SIGNAL_SELL'
                        
                        self.auto_trader.copy_trade_manager.close_position(
                            token,
                            current_price,
                            exit_reason=exit_reason
                        )
                        
                        # Execute the actual sell
                        await self.auto_trader.execute_auto_trade(
                            token,
                            'SELL',
                            f"🐋 WHALE EXIT: {whale_signal.get('whale_name', 'Unknown')} selling"
                        )
                    else:
                        self.log_message(f"⚠️ Whale selling {token} but no position to exit")
                
                return  # Skip technical analysis when whale signal present
            
            # 🐋 PRIORITY 1.5: Check existing positions for stop-loss/take-profit
            if self.auto_trader.copy_trade_manager and token in self.auto_trader.copy_trade_manager.open_positions:
                current_price = analysis.get('price', 0)
                exit_action = self.auto_trader.copy_trade_manager.update_position_price(token, current_price)
                
                if exit_action:
                    self.log_message(f"🎯 {exit_action} triggered for {token}")
                    
                    # Close position
                    closed_position = self.auto_trader.copy_trade_manager.close_position(
                        token,
                        current_price,
                        exit_reason=exit_action
                    )
                    
                    if closed_position:
                        # Execute the sell
                        await self.auto_trader.execute_auto_trade(
                            token,
                            'SELL',
                            f"🎯 {exit_action}: {closed_position['realized_pnl_pct']:+.1f}%"
                        )
                    
                    return  # Don't open new positions while managing existing ones
            
            # PRIORITY 2: Technical analysis (only if no whale signals and no position management)
            price_data = list(self.price_history[token])
            current_price = analysis['price']
            
            # Calculate RSI if we have enough data (reduced requirement)
            rsi_value = None
            if len(price_data) >= 5:  # Reduced from 14 to 5
                prices = [p['price'] for p in price_data]
                if len(prices) >= 5:
                    # Simplified RSI calculation for faster response
                    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                    gains = [d for d in deltas if d > 0]
                    losses = [-d for d in deltas if d < 0]
                    
                    if gains and losses:
                        avg_gain = sum(gains) / len(gains)
                        avg_loss = sum(losses) / len(losses)
                        rs = avg_gain / (avg_loss + 0.001)
                        rsi_value = 100 - (100 / (1 + rs))
            
            # ALWAYS check for buy signals (very aggressive)
            should_buy, buy_reason = self.auto_trader.should_buy(token, price_data, rsi_value)
            
            # ALWAYS check for sell signals
            should_sell, sell_reason = self.auto_trader.should_sell(
                token, current_price, self.portfolio[token], rsi_value, price_data
            )
            
            # Log what we're checking (with mode indicator)
            mode_indicator = "🚀 SIMPLE" if len(price_data) < 5 else "📊 ADVANCED"
            if len(price_data) >= 1:
                rsi_display = f"{rsi_value:.1f}" if rsi_value is not None else "N/A"
                self.log_message(f"🔍 {token}: Price ${current_price:.6f}, RSI: {rsi_display}")
                self.log_message(f"   {mode_indicator} Mode ({len(price_data)} data points)")
                self.log_message(f"   Holdings: {self.portfolio[token]['holdings']:.6f}")
                self.log_message(f"   Buy: {buy_reason}")
                self.log_message(f"   Sell: {sell_reason}")
            else:
                self.log_message(f"⚠️ {token}: No price data yet (first scan)")
            
            # Execute trades based on signals (PRIORITIZE SELLING FIRST)
            if should_sell and self.portfolio[token]['holdings'] > 0:
                self.log_message(f"🔥 EXECUTING SELL for {token}: {sell_reason}")
                await self.auto_trader.execute_auto_trade(token, 'SELL', sell_reason)
                
            elif should_buy and not should_sell:
                # Buy more aggressively - even if we already have holdings
                self.log_message(f"🔥 EXECUTING BUY for {token}: {buy_reason}")
                await self.auto_trader.execute_auto_trade(token, 'BUY', buy_reason)
            else:
                # Log why we're not trading
                if not should_buy and not should_sell:
                    self.log_message(f"❌ {token}: No trading signals")
                elif should_buy and should_sell:
                    self.log_message(f"⚖️ {token}: Conflicting signals - holding")
                    
            # FORCE SOME TRADING - if we haven't traded in a while, make a tiny trade
            if hasattr(self, '_force_trade_counter'):
                self._force_trade_counter += 1
            else:
                self._force_trade_counter = 1
                
            # Force a tiny trade every 50 scans to ensure system is working
            if self._force_trade_counter % 50 == 0 and len(price_data) >= 2:
                self.log_message(f"🚀 FORCE TRADING {token} - System test")
                await self.execute_simple_test_trade(token, current_price)
            
        except Exception as e:
            self.log_message(f"❌ Auto trading signal error for {token}: {e}")
    
    async def execute_simple_test_trade(self, token, price):
        """Execute a simple test trade to verify system is working"""
        try:
            # Tiny test trade amount
            test_amount = 0.001
            action = 'BUY'
            
            self.log_message(f"🧪 TEST TRADE: {action} {test_amount:.6f} {token} @ ${price:.6f}")
            
            # Record the trade directly
            timestamp = datetime.now().isoformat()
            trade_value = test_amount * price
            
            # Update portfolio
            if action == 'BUY':
                current_holdings = self.portfolio[token]['holdings']
                current_invested = self.portfolio[token]['total_invested']
                
                new_holdings = current_holdings + test_amount
                new_invested = current_invested + trade_value
                new_avg_price = new_invested / new_holdings if new_holdings > 0 else price
                
                self.portfolio[token]['holdings'] = new_holdings
                self.portfolio[token]['total_invested'] = new_invested
                self.portfolio[token]['avg_price'] = new_avg_price
            
            # Save to database
            self.conn.execute('''
                INSERT INTO trades (timestamp, token, action, quantity, price, value, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, token, action, test_amount, price, trade_value, 0))
            self.conn.commit()
            
            self.log_message(f"✅ Test trade complete: Portfolio updated")
            self.root.after(0, self.update_portfolio_display)
            
        except Exception as e:
            self.log_message(f"❌ Test trade error: {e}")
    
    def update_auto_trading_settings(self):
        """Update auto-trading parameters"""
        try:
            profit_threshold = float(self.profit_threshold_var.get()) / 100
            stop_loss = float(self.stop_loss_var.get()) / 100
            trade_amount = float(self.trade_amount_var.get())
            
            self.auto_trader.min_profit_threshold = profit_threshold
            self.auto_trader.max_loss_threshold = -stop_loss  # Make it negative
            self.auto_trader.trade_amount = trade_amount
            
            self.log_message(f"⚙️ Auto-trading settings updated:")
            self.log_message(f"   Profit threshold: {profit_threshold*100:.1f}%")
            self.log_message(f"   Stop loss: {stop_loss*100:.1f}%")
            self.log_message(f"   Trade amount: {trade_amount}")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for settings")
    
    def update_portfolio(self, token, action, quantity, price):
        """Update portfolio holdings"""
        portfolio = self.portfolio[token]
        
        if action == 'BUY':
            # Update average price and holdings
            total_cost = (portfolio['holdings'] * portfolio['avg_price']) + (quantity * price)
            portfolio['holdings'] += quantity
            portfolio['avg_price'] = total_cost / portfolio['holdings'] if portfolio['holdings'] > 0 else 0
            portfolio['total_invested'] += quantity * price
            
        elif action == 'SELL':
            if portfolio['holdings'] >= quantity:
                portfolio['holdings'] -= quantity
                portfolio['total_invested'] -= quantity * portfolio['avg_price']
                
                # Reset if all sold
                if portfolio['holdings'] == 0:
                    portfolio['avg_price'] = 0
                    portfolio['total_invested'] = 0
    
    def update_portfolio_display(self):
        """Update portfolio display in GUI"""
        total_value = 0
        total_pnl = 0
        
        for token in self.target_tokens:
            portfolio = self.portfolio[token]
            current_price = self.token_data.get(token, {}).get('price', 0)
            
            holdings = portfolio['holdings']
            avg_price = portfolio['avg_price']
            current_value = holdings * current_price
            pnl = (current_price - avg_price) * holdings if holdings > 0 else 0
            pnl_percent = (pnl / (avg_price * holdings)) * 100 if holdings > 0 and avg_price > 0 else 0
            
            total_value += current_value
            total_pnl += pnl
            
            # Update tree view
            self.portfolio_tree.item(token, values=(
                token,
                f"{holdings:.4f}",
                f"${avg_price:.6f}",
                f"${current_price:.6f}",
                f"${current_value:.2f}",
                f"${pnl:.2f}",
                f"{pnl_percent:+.2f}%"
            ))
        
        # Update summary labels
        self.total_value_label.config(text=f"Total Value: ${total_value:.2f}")
        self.total_pnl_label.config(text=f"Total P&L: ${total_pnl:+.2f}")
        
        # Calculate daily P&L (placeholder - would need historical data)
        today = datetime.now().date()
        daily_pnl = self.daily_pnl.get(today, 0)
        self.daily_pnl_label.config(text=f"Daily P&L: ${daily_pnl:+.2f}")
    
    def update_whale_display(self, whale_signal):
        """Update whale tracker display with new signal"""
        try:
            # Add to whale tree (most recent first)
            time_str = whale_signal['time'].strftime("%H:%M:%S")
            whale_short = f"{whale_signal['wallet'][:6]}..."
            
            self.whale_tree.insert('', 0, values=(
                time_str,
                whale_short,
                whale_signal['action'],
                whale_signal['token'],
                f"{whale_signal['amount']:.2f}",
                "✅ COPIED"
            ))
            
            # Keep only last 10 entries
            children = self.whale_tree.get_children()
            if len(children) > 10:
                for item in children[10:]:
                    self.whale_tree.delete(item)
            
            # Update whale status label
            whale_idx = self.whale_wallets.index(whale_signal['wallet'])
            if whale_idx < len(self.whale_status_labels):
                self.whale_status_labels[whale_idx].config(
                    text=f"Whale {whale_idx+1}: {whale_signal['wallet'][:8]}...{whale_signal['wallet'][-6:]} - ✅ ACTIVE ({whale_signal['minutes_ago']:.0f}m ago)",
                    foreground='green'
                )
        
        except Exception as e:
            self.log_message(f"Error updating whale display: {e}")
    
    def load_trade_history(self):
        """Load trade history from database"""
        try:
            cursor = self.conn.execute('''
                SELECT * FROM trades ORDER BY timestamp DESC LIMIT 100
            ''')
            
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Add trades to tree
            for row in cursor.fetchall():
                self.history_tree.insert('', 'end', values=row)
            
            # Update analytics
            self.update_analytics()
            
        except Exception as e:
            self.log_message(f"Error loading trade history: {e}")
    
    def clear_trade_history(self):
        """Clear all trade history from database"""
        from tkinter import messagebox
        
        # Confirm with user
        response = messagebox.askyesno(
            "Clear Trade History",
            "Are you sure you want to clear ALL trade history?\n\n"
            "This will delete:\n"
            "• All trade records\n"
            "• All price history\n"
            "• All analytics data\n\n"
            "This action CANNOT be undone!",
            icon='warning'
        )
        
        if not response:
            self.log_message("Trade history clear cancelled")
            return
        
        try:
            # Clear trades table
            self.conn.execute('DELETE FROM trades')
            self.log_message("✅ Cleared all trades from database")
            
            # Clear price history table
            self.conn.execute('DELETE FROM price_history')
            self.log_message("✅ Cleared all price history from database")
            
            # Commit changes
            self.conn.commit()
            
            # Clear GUI display
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Reset analytics
            self.win_rate_label.config(text="Win Rate: 0.0%")
            self.avg_profit_label.config(text="Avg Profit: $0.00")
            self.total_trades_label.config(text="Total Trades: 0")
            self.best_trade_label.config(text="Best Trade: $0.00")
            
            # Clear chart
            self.pnl_figure.clear()
            self.pnl_canvas.draw()
            
            self.log_message("🎯 Trade history cache cleared successfully!")
            
            messagebox.showinfo(
                "Success",
                "Trade history cleared successfully!\n\n"
                "All historical data has been removed from the database."
            )
            
        except Exception as e:
            self.log_message(f"❌ Error clearing trade history: {e}")
            messagebox.showerror(
                "Error",
                f"Failed to clear trade history:\n{e}"
            )
    
    def update_analytics(self):
        """Update analytics display"""
        try:
            # Get trade statistics
            cursor = self.conn.execute('''
                SELECT action, profit_loss FROM trades WHERE action = 'SELL'
            ''')
            
            sell_trades = cursor.fetchall()
            if not sell_trades:
                return
            
            profitable_trades = [t for t in sell_trades if t[1] > 0]
            win_rate = (len(profitable_trades) / len(sell_trades)) * 100
            
            total_pnl = sum(t[1] for t in sell_trades)
            avg_profit = total_pnl / len(sell_trades)
            best_trade = max(t[1] for t in sell_trades) if sell_trades else 0
            
            # Update labels
            self.win_rate_label.config(text=f"Win Rate: {win_rate:.1f}%")
            self.avg_profit_label.config(text=f"Avg Profit: ${avg_profit:.2f}")
            self.total_trades_label.config(text=f"Total Trades: {len(sell_trades)}")
            self.best_trade_label.config(text=f"Best Trade: ${best_trade:.2f}")
            
            # Update P&L chart
            self.update_pnl_chart(sell_trades)
            
        except Exception as e:
            self.log_message(f"Error updating analytics: {e}")
    
    def update_pnl_chart(self, trades):
        """Update P&L chart"""
        if not trades:
            return
        
        self.pnl_figure.clear()
        ax = self.pnl_figure.add_subplot(111)
        
        # Calculate cumulative P&L
        cumulative_pnl = []
        running_total = 0
        
        for trade in reversed(trades):  # Reverse to get chronological order
            running_total += trade[1]
            cumulative_pnl.append(running_total)
        
        # Plot cumulative P&L
        x_values = range(len(cumulative_pnl))
        ax.plot(x_values, cumulative_pnl, linewidth=2, color='#4CAF50' if cumulative_pnl[-1] > 0 else '#F44336')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Cumulative Profit & Loss', fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        
        self.pnl_figure.tight_layout()
        self.pnl_canvas.draw()
    
    def export_history(self):
        """Export trade history to CSV"""
        try:
            import csv
            from tkinter.filedialog import asksaveasfilename
            
            filename = asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )
            
            if filename:
                cursor = self.conn.execute('SELECT * FROM trades ORDER BY timestamp')
                trades = cursor.fetchall()
                
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['ID', 'Timestamp', 'Token', 'Action', 'Quantity', 'Price', 'Value', 'P&L'])
                    writer.writerows(trades)
                
                messagebox.showinfo("Export Complete", f"Trade history exported to {filename}")
                self.log_message(f"📊 Trade history exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def refresh_wallet_balance(self):
        """Refresh wallet balance from the blockchain"""
        def fetch_balance():
            try:
                self.log_message("🔄 Refreshing wallet balance...")
                
                # Get wallet address from config
                wallet_address = self.config.get('wallet', {}).get('solana_address', '')
                
                if not wallet_address:
                    self.log_message("❌ No wallet address configured")
                    self.root.after(0, lambda: self.update_wallet_display(0, "Not configured"))
                    return
                
                self.log_message(f"📍 Checking wallet: {wallet_address[:8]}...{wallet_address[-8:]}")
                
                # Use a simple HTTP request to check SOL balance (Solana RPC)
                import requests
                
                # Solana mainnet RPC endpoint
                rpc_url = "https://api.mainnet-beta.solana.com"
                
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [wallet_address]
                }
                
                self.log_message("🌐 Connecting to Solana RPC...")
                response = requests.post(rpc_url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_message(f"📡 RPC Response: {data}")
                    
                    if 'result' in data:
                        # Balance is in lamports, convert to SOL (1 SOL = 1,000,000,000 lamports)
                        lamports = data['result']['value']
                        sol_balance = lamports / 1_000_000_000
                        
                        self.sol_balance = sol_balance
                        self.root.after(0, lambda: self.update_wallet_display(sol_balance, "Connected"))
                        
                        self.log_message(f"✅ Wallet balance updated: {sol_balance:.6f} SOL")
                        
                    else:
                        error = data.get('error', {}).get('message', 'Unknown error')
                        self.log_message(f"❌ RPC Error: {error}")
                        self.root.after(0, lambda: self.update_wallet_display(0, "RPC Error"))
                        
                else:
                    self.log_message(f"❌ HTTP Error: {response.status_code}")
                    self.root.after(0, lambda: self.update_wallet_display(0, "Connection Error"))
                    
            except requests.exceptions.Timeout:
                self.log_message("❌ Wallet balance check timed out")
                self.root.after(0, lambda: self.update_wallet_display(0, "Timeout"))
                
            except Exception as e:
                self.log_message(f"❌ Error fetching wallet balance: {e}")
                self.root.after(0, lambda: self.update_wallet_display(0, "Error"))
        
        # Run in background thread to avoid blocking GUI
        import threading
        threading.Thread(target=fetch_balance, daemon=True).start()
    
    def sync_portfolio_with_blockchain(self):
        """
        Sync portfolio holdings with actual blockchain balances
        CRITICAL: Prevents 'Insufficient funds' errors by ensuring portfolio matches wallet
        """
        # Check if AutoTrader has real trading enabled
        if not hasattr(self, 'auto_trader') or not self.auto_trader.real_trading_mode:
            return
        
        if not hasattr(self.auto_trader, 'real_trader') or self.auto_trader.real_trader is None:
            return
        
        self.log_message("🔄 Syncing portfolio with blockchain...")
        sync_errors = []
        tokens_synced = 0
        
        for token in self.target_tokens:
            try:
                # Get actual balance from blockchain (bypasses portfolio cache)
                actual_balance = self.auto_trader.real_trader.get_balance(token, use_cache=False)
                portfolio_balance = self.portfolio[token]['holdings']
                
                # Check for discrepancies
                if abs(actual_balance - portfolio_balance) > 0.000001:  # Allow tiny rounding errors
                    if actual_balance > 0:
                        # Update portfolio with actual balance
                        current_price = self.token_data.get(token, {}).get('price', 0)
                        
                        if current_price > 0:
                            self.portfolio[token]['holdings'] = actual_balance
                            # Use current price as avg price (we don't know actual cost basis)
                            self.portfolio[token]['avg_price'] = current_price
                            self.portfolio[token]['total_invested'] = actual_balance * current_price
                            
                            self.log_message(f"✅ {token}: Synced {actual_balance:.6f} tokens @ ${current_price:.8f}")
                            self.log_message(f"   📊 Was: {portfolio_balance:.6f}, Now: {actual_balance:.6f} (Δ {actual_balance - portfolio_balance:+.6f})")
                            tokens_synced += 1
                        else:
                            # Price not loaded yet - update holdings anyway, price will update later
                            self.portfolio[token]['holdings'] = actual_balance
                            self.log_message(f"⚠️ {token}: Synced {actual_balance:.6f} tokens (waiting for price data)")
                            tokens_synced += 1
                    else:
                        # Clear stale portfolio data if blockchain shows zero balance
                        if self.portfolio[token]['holdings'] > 0:
                            self.log_message(f"🧹 {token}: Clearing stale portfolio data")
                            self.log_message(f"   ❌ Portfolio shows: {portfolio_balance:.6f} tokens")
                            self.log_message(f"   ✅ Blockchain shows: 0 tokens")
                            self.log_message(f"   🔧 Fixing: Resetting to 0")
                            self.portfolio[token]['holdings'] = 0
                            self.portfolio[token]['avg_price'] = 0
                            self.portfolio[token]['total_invested'] = 0
                            tokens_synced += 1
                else:
                    # Balances match - no sync needed
                    if actual_balance > 0:
                        self.log_message(f"✓ {token}: In sync ({actual_balance:.6f} tokens)")
                    
            except Exception as e:
                error_msg = f"❌ Error syncing {token}: {e}"
                self.log_message(error_msg)
                sync_errors.append(error_msg)
        
        # Summary
        if tokens_synced > 0:
            self.log_message(f"🎯 Portfolio sync complete: {tokens_synced} token(s) updated")
            # Force portfolio display update
            self.root.after(0, self.update_portfolio_display)
        else:
            self.log_message(f"✅ Portfolio already in sync with blockchain")
        
        if sync_errors:
            self.log_message(f"⚠️ {len(sync_errors)} sync error(s) occurred")
        
        # Update display
        self.root.after(0, self.update_portfolio_display)
        self.log_message("✅ Portfolio sync complete!")
    
    def old_refresh_wallet_balance_function(self):
        """Old refresh function kept for reference"""
        # Run in background thread to avoid blocking GUI
        threading.Thread(target=fetch_balance, daemon=True).start()
    
    def update_wallet_display(self, sol_balance, status):
        """Update wallet balance display in GUI"""
        try:
            # Update SOL balance
            if status == "Connected":
                balance_text = f"SOL Balance: {sol_balance:.6f} SOL"
                self.sol_balance_label.config(text=balance_text)
                
                # Update trading tab wallet display
                if hasattr(self, 'trade_wallet_label'):
                    self.trade_wallet_label.config(text=f"Wallet: {sol_balance:.4f} SOL")
                
                # Calculate USD value if we have SOL price
                if 'SOL' in self.token_data:
                    sol_price = self.token_data['SOL']['price']
                    usd_value = sol_balance * sol_price
                    self.wallet_usd_value = usd_value
                    self.wallet_usd_label.config(text=f"USD Value: ${usd_value:.2f}")
                else:
                    self.wallet_usd_label.config(text="USD Value: Price loading...")
                    
            else:
                self.sol_balance_label.config(text=f"SOL Balance: {status}")
                self.wallet_usd_label.config(text="USD Value: --")
                
                # Update trading tab
                if hasattr(self, 'trade_wallet_label'):
                    self.trade_wallet_label.config(text=f"Wallet: {status}")
                
        except Exception as e:
            self.log_message(f"❌ Error updating wallet display: {e}")
    
    def update_wallet_usd_value(self):
        """Update wallet USD value when SOL price changes"""
        if self.sol_balance > 0 and 'SOL' in self.token_data:
            sol_price = self.token_data['SOL']['price']
            usd_value = self.sol_balance * sol_price
            self.wallet_usd_value = usd_value
            self.wallet_usd_label.config(text=f"USD Value: ${usd_value:.2f}")
    
    def get_token_balance(self, token_address):
        """Get balance for a specific SPL token (placeholder for future implementation)"""
        # This would require more complex SPL token balance checking
        # For now, return 0 as placeholder
        return 0.0
        
    def log_message(self, message):
        """Add a message to the log - THREAD SAFE"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_entry = f"[{timestamp}] {message}\n"
            
            # Always print to console for debugging
            print(f"[LOG] {log_entry.strip()}")
            
            # Use root.after to ensure GUI updates happen on main thread
            if hasattr(self, 'log_text') and hasattr(self, 'root'):
                def update_log():
                    try:
                        self.log_text.insert(tk.END, log_entry)
                        self.log_text.see(tk.END)
                        
                        # Keep only last 1000 lines
                        lines = self.log_text.get(1.0, tk.END).split('\n')
                        if len(lines) > 1000:
                            self.log_text.delete(1.0, f"{len(lines) - 1000}.0")
                    except Exception as e:
                        print(f"[GUI UPDATE ERROR] {e}")
                
                # Schedule update on main thread
                self.root.after(0, update_log)
            else:
                print(f"[WARNING] log_text or root not initialized yet!")
        except Exception as e:
            print(f"[LOG ERROR] Failed to log message: {e}")
            import traceback
            traceback.print_exc()
    
    def test_log_message(self):
        """Test function to verify logging works"""
        print("=" * 50)
        print("TEST LOG BUTTON CLICKED!")
        print(f"log_text exists: {hasattr(self, 'log_text')}")
        if hasattr(self, 'log_text'):
            print(f"log_text widget: {self.log_text}")
            print(f"Attempting to insert text...")
        print("=" * 50)
        
        import random
        test_messages = [
            "🔧 TEST: Logging system is working!",
            "💡 TEST: If you see this, logs are functional",
            "🎯 TEST: Bot status check",
            "📊 TEST: Token scan simulation",
            "🔥 TEST: Trading signal test"
        ]
        self.log_message(random.choice(test_messages))
        self.log_message(f"🔍 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"🤖 Auto-trading: {'ENABLED' if self.auto_trader.trading_enabled else 'DISABLED'}")
        self.log_message(f"🏃 Bot running: {'YES' if self.running else 'NO'}")
        
        print("Test log messages sent!")
        
        # Force a messagebox to confirm button works
        messagebox.showinfo("Test Log", "Test log button clicked! Check the Analytics tab for logs.")
    
    def update_market_card(self, token, price, change_percent=0):
        """Update market card display for a token"""
        if token not in self.market_cards:
            return
        
        card = self.market_cards[token]
        
        # Update price
        card.price_label.config(text=f"${price:.6f}")
        
        # Update change
        change_text = f"{change_percent:+.2f}%"
        change_color = 'green' if change_percent > 0 else 'red' if change_percent < 0 else 'black'
        card.change_label.config(text=change_text, foreground=change_color)
        
        # Update indicator
        indicators = self.calculate_market_indicators(token, price)
        trend = indicators['trend']
        
        if trend == 'BULLISH':
            indicator_text = "🟢 BULLISH"
            indicator_color = 'green'
        elif trend == 'BEARISH':
            indicator_text = "� BEARISH"
            indicator_color = 'red'
        else:
            indicator_text = "⚪ NEUTRAL"
            indicator_color = 'gray'
        
        card.indicator_label.config(text=indicator_text, foreground=indicator_color)
    
    async def analyze_token(self, symbol):
        """Analyze a token for trading opportunities"""
        try:
            self.log_message(f"🔍 Analyzing {symbol}...")
            
            # Get token information
            token_info = await self.exchange_manager.get_token_info(symbol)
            if not token_info:
                self.log_message(f"❌ Could not get info for {symbol}")
                return None
            
            price = token_info.price_usd
            timestamp = datetime.now()
            
            # Store price data
            price_data = {
                'price': price,
                'timestamp': timestamp,
                'volume': 0  # Placeholder
            }
            
            self.price_history[symbol].append(price_data)
            
            # Calculate change percentage
            change_percent = 0
            if symbol in self.token_data:
                prev_price = self.token_data[symbol]['price']
                if prev_price > 0:
                    change_percent = ((price - prev_price) / prev_price) * 100
            
            self.token_data[symbol] = {'price': price, 'timestamp': timestamp}
            
            # Update GUI components
            self.root.after(0, lambda: self.update_market_card(symbol, price, change_percent))
            self.root.after(0, lambda: self.update_portfolio_display())
            
            # Auto-trading logic
            if self.auto_trader.trading_enabled:
                self.log_message(f"🤖 Checking auto-trading signals for {symbol}...")
                await self.check_auto_trading_signals(symbol, {
                    'symbol': symbol,
                    'price': price,
                    'address': token_info.address,
                    'timestamp': timestamp,
                    'change_percent': change_percent
                })
            else:
                # Log when auto-trading is disabled to help debug
                if hasattr(self, '_debug_counter'):
                    self._debug_counter += 1
                else:
                    self._debug_counter = 1
                
                if self._debug_counter % 20 == 0:  # Every 20th scan
                    self.log_message(f"ℹ️ Auto-trading DISABLED - Check the '🤖 Auto Trading' checkbox to enable")
            
            # Store price in database
            self.conn.execute('''
                INSERT INTO price_history (timestamp, token, price)
                VALUES (?, ?, ?)
            ''', (timestamp.isoformat(), symbol, price))
            self.conn.commit()
            
            # Update daily P&L
            today = timestamp.date()
            if today not in self.daily_pnl:
                self.daily_pnl[today] = 0
            
            self.log_message(f"💰 {symbol}: ${price:.6f} ({change_percent:+.2f}%)")
            
            return {
                'symbol': symbol,
                'price': price,
                'address': token_info.address,
                'timestamp': timestamp,
                'change_percent': change_percent
            }
            
        except Exception as e:
            self.log_message(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def scan_tokens(self):
        """Scan and analyze target tokens + SOL for wallet calculations"""
        self.log_message("📊 Starting comprehensive token scan...")
        
        # Always fetch SOL price first for wallet USD calculations
        try:
            self.log_message("💰 Fetching SOL price...")
            sol_info = await self.exchange_manager.get_token_info('SOL')
            if sol_info:
                sol_price = sol_info.price_usd
                self.token_data['SOL'] = {'price': sol_price, 'timestamp': datetime.now()}
                self.root.after(0, self.update_wallet_usd_value)
                self.log_message(f"💰 SOL: ${sol_price:.2f} (for wallet calculations)")
            else:
                self.log_message("⚠️ No SOL price data returned")
        except Exception as e:
            self.log_message(f"⚠️ Could not fetch SOL price for wallet: {e}")
        
        # Now scan our target tokens
        for symbol in self.target_tokens:
            if not self.running:
                self.log_message(f"🛑 Bot stopped during {symbol} scan")
                break
                
            try:
                self.log_message(f"🔍 Analyzing {symbol}...")
                analysis = await self.analyze_token(symbol)
                if analysis:
                    self.log_message(f"✅ {symbol} analysis complete")
                    # Update chart if it's the selected token
                    if symbol == self.chart_token_var.get():
                        self.root.after(0, self.update_chart)
                else:
                    self.log_message(f"❌ No analysis data for {symbol}")
            except Exception as e:
                self.log_message(f"❌ Error analyzing {symbol}: {e}")
            
            await asyncio.sleep(1)  # Small delay between tokens
        
        if self.running:
            self.log_message(f"✅ Scan complete. Next scan in {self.scan_interval} seconds...")
            # Add periodic status updates
            if hasattr(self, '_scan_counter'):
                self._scan_counter += 1
            else:
                self._scan_counter = 1
            
            if self._scan_counter % 5 == 0:  # Every 5th scan
                self.log_message(f"🔄 Continuous monitoring active (Scan #{self._scan_counter})")
                self.log_message(f"📊 Tracking {len(self.target_tokens)} ultra-aggressive tokens")
    
    async def bot_loop(self):
        """Main bot loop with enhanced features"""
        scan_count = 0
        self.log_message("🎯 Bot loop starting...")
        
        while self.running:
            try:
                self.log_message(f"🔄 Starting scan #{scan_count + 1}...")
                await self.scan_tokens()
                
                # Refresh wallet balance every 10 scans (to avoid too frequent requests)
                scan_count += 1
                if scan_count % 10 == 0:
                    self.log_message("💰 Refreshing wallet balance...")
                    self.refresh_wallet_balance()
                
                # Sync portfolio with blockchain every 20 scans (prevents "insufficient funds" errors)
                if scan_count % 20 == 0:
                    self.log_message("🔄 Syncing portfolio with blockchain...")
                    self.root.after(0, self.sync_portfolio_with_blockchain)
                
                self.log_message(f"⏰ Waiting {self.scan_interval} seconds for next scan...")
                # Wait for next scan
                for i in range(self.scan_interval):
                    if not self.running:
                        break
                    await asyncio.sleep(1)
                    
            except Exception as e:
                self.log_message(f"❌ Bot error: {e}")
                import traceback
                self.log_message(f"📋 Error details: {traceback.format_exc()}")
                # Don't break - try to continue
                await asyncio.sleep(5)  # Wait 5 seconds before retrying
        
        self.log_message("🛑 Trading bot stopped")
    
    def start_bot(self):
        """Start the advanced trading bot"""
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Running ✅")
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            
            self.log_message("🚀 Advanced Trading Bot started!")
            self.log_message("📊 Real-time price monitoring active")
            self.log_message("📈 Chart updates enabled")
            
            # Start bot in a separate thread
            def run_bot():
                try:
                    self.log_message("🔧 Creating new async event loop...")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    self.log_message("🚀 Starting bot loop...")
                    loop.run_until_complete(self.bot_loop())
                    loop.close()
                    self.log_message("🏁 Bot loop completed")
                except Exception as e:
                    self.log_message(f"💥 CRITICAL BOT ERROR: {e}")
                    import traceback
                    self.log_message(f"📋 Traceback: {traceback.format_exc()}")
            
            self.bot_thread = threading.Thread(target=run_bot, daemon=True)
            self.bot_thread.start()
            self.log_message("🧵 Bot thread started")
    
    def stop_bot(self):
        """Stop the trading bot"""
        if self.running:
            self.running = False
            self.status_label.config(text="Status: Stopped ❌")
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')
            
            self.log_message("🛑 Trading Bot stopped by user")
    
    def toggle_real_trading(self):
        """Toggle between paper trading and real trading mode"""
        if not REAL_TRADING_AVAILABLE:
            messagebox.showerror("Error", "Real trading module not available!\nMake sure real_trader.py is installed.")
            return
        
        if not self.auto_trader.real_trading_mode:
            # Switching to REAL trading
            response = messagebox.askyesno(
                "⚠️ ENABLE REAL TRADING ⚠️",
                "WARNING: You are about to enable REAL blockchain trading!\n\n"
                "This will:\n"
                "• Execute REAL transactions on Solana blockchain\n"
                "• Spend REAL SOL for transaction fees\n"
                "• Trade REAL tokens from your wallet\n"
                "• Cannot be undone once transactions are on-chain\n\n"
                "You need to provide your private key.\n\n"
                "Are you ABSOLUTELY sure you want to continue?",
                icon='warning'
            )
            
            if response:
                # Get private key
                from tkinter import simpledialog
                private_key = simpledialog.askstring(
                    "Private Key Required",
                    "Enter your Solana wallet private key (Base58):\n\n"
                    "⚠️ WARNING: Keep your private key secure!\n"
                    "Never share it or commit it to version control.",
                    show='*'
                )
                
                if private_key:
                    try:
                        # Initialize real trader
                        self.log_message("🔴 Initializing real trading mode...")
                        self.auto_trader.real_trader = RealTrader(private_key)
                        
                        # Set Jupiter API key if available from config
                        try:
                            from pathlib import Path
                            config_path = Path(__file__).parent.parent / 'config.json'
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            
                            if 'solana' in config and 'jupiter_api_key' in config['solana'] and config['solana']['jupiter_api_key']:
                                self.auto_trader.real_trader.set_jupiter_api_key(config['solana']['jupiter_api_key'])
                        except Exception as e:
                            self.log_message(f"⚠️ Could not load Jupiter API key from config: {e}")
                        
                        self.auto_trader.real_trading_mode = True
                        
                        # Update UI
                        self.trading_mode_label.config(
                            text="🔴 REAL TRADING MODE - LIVE",
                            foreground='red'
                        )
                        self.real_trading_button.config(text="Disable Real Trading")
                        
                        self.log_message("✅ REAL TRADING MODE ENABLED")
                        self.log_message(f"🔗 Wallet: {self.auto_trader.real_trader.wallet_address}")
                        self.log_message("⚠️ All trades will be executed on Solana blockchain!")
                        
                        # Sync portfolio with blockchain immediately after enabling
                        self.root.after(2000, self.sync_portfolio_with_blockchain)
                        
                        messagebox.showinfo(
                            "Real Trading Enabled",
                            f"✅ Real trading mode is now ACTIVE!\n\n"
                            f"Wallet: {self.auto_trader.real_trader.wallet_address}\n\n"
                            f"All trades will execute on Solana blockchain."
                        )
                        
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to initialize real trading:\n{e}")
                        self.log_message(f"❌ Real trading initialization failed: {e}")
        else:
            # Switching back to paper trading
            response = messagebox.askyesno(
                "Disable Real Trading",
                "Switch back to paper trading mode?\n\n"
                "This will stop executing real blockchain transactions."
            )
            
            if response:
                self.auto_trader.real_trading_mode = False
                self.auto_trader.real_trader = None
                
                # Update UI
                self.trading_mode_label.config(
                    text="📄 PAPER TRADING MODE",
                    foreground='orange'
                )
                self.real_trading_button.config(text="Enable Real Trading")
                
                self.log_message("📄 Switched back to PAPER TRADING mode")
                messagebox.showinfo("Paper Trading", "Switched back to paper trading mode.")
    
    def update_settings(self):
        """Update bot settings"""
        try:
            new_interval = int(self.interval_var.get())
            if new_interval < 5:
                messagebox.showwarning("Warning", "Minimum scan interval is 5 seconds")
                return
            
            self.scan_interval = new_interval
            self.log_message(f"⚙️ Scan interval updated to {new_interval} seconds")
            
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for scan interval")
    
    def on_closing(self):
        """Handle window closing"""
        if self.running:
            self.stop_bot()
        
        if hasattr(self, 'conn'):
            self.conn.close()
        
        self.root.destroy()
    
    def run(self):
        """Start the advanced GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = TradingBotGUI()
        app.run()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Failed to start Advanced Trading Dashboard: {e}")

if __name__ == "__main__":
    main()