"""
Modern Automated Trading Dashboard
Beautiful, clean interface focused on monitoring and visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
import json
import sqlite3
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import queue
import random

# Import core trading components
from src.strategies.duplex_strategy import DuplexTradingStrategy, TradeStrategy
from src.strategies.profit_optimizer import ProfitOptimizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Container for trading performance metrics"""
    total_balance: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    active_positions: int = 0
    available_capital: float = 0.0

@dataclass
class MarketSentiment:
    """Container for market sentiment data"""
    overall_sentiment: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    confidence: float = 0.5  # 0.0 to 1.0
    trending_tokens: List[str] = None
    volume_surge: bool = False
    volatility_level: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    
    def __post_init__(self):
        if self.trending_tokens is None:
            self.trending_tokens = []

class ModernTradingDashboard:
    """
    Beautiful automated trading dashboard with real-time monitoring
    """
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🚀 AI Trading Bot Dashboard")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')  # Dark theme
        
        # Trading state
        self.is_trading = False
        self.trading_thread = None
        self.update_queue = queue.Queue()
        
        # Data storage
        self.metrics = TradingMetrics()
        self.sentiment = MarketSentiment()
        self.trade_history = []
        self.balance_history = []
        self.pnl_history = []
        
        # Initialize trading components
        self.config = self.load_config()
        self.duplex_strategy = DuplexTradingStrategy(self.config)
        self.profit_optimizer = ProfitOptimizer(self.config)
        
        # Setup GUI
        self.setup_styles()
        self.create_header()
        self.create_main_dashboard()
        self.create_footer()
        
        # Start update loop
        self.update_display()
        
        logger.info("🎨 Modern Trading Dashboard initialized")
    
    def load_config(self) -> Dict:
        """Load trading configuration"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'trading': {
                    'initial_capital': 1000.0,
                    'max_position_size': 100.0,
                    'risk_per_trade': 0.02
                }
            }
    
    def setup_styles(self):
        """Setup modern dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#1a1a1a', 
            'bg_tertiary': '#2a2a2a',
            'accent_green': '#00ff88',
            'accent_red': '#ff4444',
            'accent_blue': '#4488ff',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'text_muted': '#888888'
        }
        
        # Configure ttk styles
        style.configure('Dashboard.TFrame', background=colors['bg_primary'])
        style.configure('Card.TFrame', background=colors['bg_secondary'], relief='solid', borderwidth=1)
        style.configure('Header.TLabel', background=colors['bg_primary'], foreground=colors['text_primary'], 
                       font=('Helvetica', 24, 'bold'))
        style.configure('Title.TLabel', background=colors['bg_secondary'], foreground=colors['text_primary'],
                       font=('Helvetica', 14, 'bold'))
        style.configure('Value.TLabel', background=colors['bg_secondary'], foreground=colors['accent_green'],
                       font=('Helvetica', 16, 'bold'))
        style.configure('Status.TLabel', background=colors['bg_secondary'], foreground=colors['text_secondary'],
                       font=('Helvetica', 12))
        
        # Button styles
        style.configure('Start.TButton', font=('Helvetica', 12, 'bold'))
        style.configure('Stop.TButton', font=('Helvetica', 12, 'bold'))
        
        self.colors = colors
    
    def create_header(self):
        """Create the header section"""
        header_frame = ttk.Frame(self.root, style='Dashboard.TFrame')
        header_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        # Title and status
        title_frame = ttk.Frame(header_frame, style='Dashboard.TFrame')
        title_frame.pack(fill=tk.X)
        
        ttk.Label(title_frame, text="🚀 AI Trading Bot", style='Header.TLabel').pack(side=tk.LEFT)
        
        # Status indicator
        self.status_frame = ttk.Frame(title_frame, style='Dashboard.TFrame')
        self.status_frame.pack(side=tk.RIGHT)
        
        self.status_label = ttk.Label(self.status_frame, text="● OFFLINE", 
                                     foreground=self.colors['accent_red'],
                                     background=self.colors['bg_primary'],
                                     font=('Helvetica', 14, 'bold'))
        self.status_label.pack(side=tk.RIGHT, padx=(0, 20))
        
        # Control buttons
        control_frame = ttk.Frame(self.status_frame)
        control_frame.pack(side=tk.RIGHT, padx=(0, 20))
        
        self.start_button = ttk.Button(control_frame, text="▶ START", 
                                      command=self.start_trading, style='Start.TButton')
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="⏹ STOP", 
                                     command=self.stop_trading, style='Stop.TButton',
                                     state='disabled')
        self.stop_button.pack(side=tk.LEFT)
    
    def create_main_dashboard(self):
        """Create the main dashboard with charts and metrics"""
        main_frame = ttk.Frame(self.root, style='Dashboard.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)
        
        # Top row - Key metrics
        self.create_metrics_panel(main_frame)
        
        # Middle row - Charts and sentiment
        self.create_charts_panel(main_frame)
        
        # Bottom row - Active trades and recent activity
        self.create_activity_panel(main_frame)
    
    def create_metrics_panel(self, parent):
        """Create the key metrics panel"""
        metrics_frame = ttk.Frame(parent, style='Dashboard.TFrame')
        metrics_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Create metric cards
        metrics = [
            ("💰 Total Balance", "total_balance", "${:.2f}"),
            ("📈 Daily P&L", "daily_pnl", "${:+.2f}"),
            ("🎯 Win Rate", "win_rate", "{:.1f}%"),
            ("🔄 Active Trades", "active_positions", "{}"),
            ("💵 Available", "available_capital", "${:.2f}"),
            ("📊 Total Trades", "total_trades", "{}")
        ]
        
        self.metric_labels = {}
        
        for i, (title, key, format_str) in enumerate(metrics):
            card = self.create_metric_card(metrics_frame, title, key, format_str)
            card.grid(row=0, column=i, padx=10, sticky='ew')
            metrics_frame.grid_columnconfigure(i, weight=1)
    
    def create_metric_card(self, parent, title, key, format_str):
        """Create an individual metric card"""
        card = ttk.Frame(parent, style='Card.TFrame')
        card.configure(padding=15)
        
        ttk.Label(card, text=title, style='Status.TLabel').pack()
        
        value_label = ttk.Label(card, text=format_str.format(0), style='Value.TLabel')
        value_label.pack(pady=(5, 0))
        
        self.metric_labels[key] = (value_label, format_str)
        
        return card
    
    def create_charts_panel(self, parent):
        """Create the charts and sentiment panel"""
        charts_frame = ttk.Frame(parent, style='Dashboard.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Left side - Performance chart
        chart_card = ttk.Frame(charts_frame, style='Card.TFrame')
        chart_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(chart_card, text="📈 Performance Chart", style='Title.TLabel').pack(pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), facecolor='#1a1a1a')
        self.ax = self.fig.add_subplot(111, facecolor='#1a1a1a')
        
        # Style the chart
        self.ax.tick_params(colors='white')
        self.ax.spines['bottom'].set_color('white')
        self.ax.spines['top'].set_color('white')
        self.ax.spines['right'].set_color('white')
        self.ax.spines['left'].set_color('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, chart_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right side - Market sentiment
        sentiment_card = ttk.Frame(charts_frame, style='Card.TFrame')
        sentiment_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        sentiment_card.configure(width=300)
        
        self.create_sentiment_panel(sentiment_card)
    
    def create_sentiment_panel(self, parent):
        """Create the market sentiment panel"""
        ttk.Label(parent, text="🎯 Market Sentiment", style='Title.TLabel').pack(pady=10)
        
        # Sentiment indicator
        sentiment_frame = ttk.Frame(parent, style='Card.TFrame')
        sentiment_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.sentiment_label = ttk.Label(sentiment_frame, text="NEUTRAL", 
                                        style='Value.TLabel')
        self.sentiment_label.pack(pady=10)
        
        # Confidence meter
        confidence_frame = ttk.Frame(parent, style='Card.TFrame')
        confidence_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(confidence_frame, text="Confidence", style='Status.TLabel').pack()
        
        self.confidence_var = tk.DoubleVar()
        self.confidence_bar = ttk.Progressbar(confidence_frame, variable=self.confidence_var,
                                            maximum=100, style='TProgressbar')
        self.confidence_bar.pack(fill=tk.X, padx=10, pady=10)
        
        self.confidence_label = ttk.Label(confidence_frame, text="50%", style='Status.TLabel')
        self.confidence_label.pack()
        
        # Trending tokens
        trending_frame = ttk.Frame(parent, style='Card.TFrame')
        trending_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(trending_frame, text="🔥 Hot Tokens", style='Title.TLabel').pack(pady=(10, 5))
        
        self.trending_text = tk.Text(trending_frame, height=8, bg='#2a2a2a', fg='white',
                                    font=('Courier', 10), relief='flat')
        self.trending_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    
    def create_activity_panel(self, parent):
        """Create the activity panel with trades and logs"""
        activity_frame = ttk.Frame(parent, style='Dashboard.TFrame')
        activity_frame.pack(fill=tk.X)
        
        # Active positions
        positions_card = ttk.Frame(activity_frame, style='Card.TFrame')
        positions_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        ttk.Label(positions_card, text="🔄 Active Positions", style='Title.TLabel').pack(pady=10)
        
        # Positions treeview
        positions_frame = ttk.Frame(positions_card)
        positions_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        columns = ('Symbol', 'Side', 'Size', 'Entry', 'Current', 'P&L%')
        self.positions_tree = ttk.Treeview(positions_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            self.positions_tree.heading(col, text=col)
            self.positions_tree.column(col, width=80)
        
        positions_scroll = ttk.Scrollbar(positions_frame, orient=tk.VERTICAL, command=self.positions_tree.yview)
        self.positions_tree.configure(yscrollcommand=positions_scroll.set)
        
        self.positions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        positions_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Recent activity
        activity_card = ttk.Frame(activity_frame, style='Card.TFrame')
        activity_card.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        ttk.Label(activity_card, text="📋 Recent Activity", style='Title.TLabel').pack(pady=10)
        
        # Activity log
        log_frame = ttk.Frame(activity_card)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.activity_text = tk.Text(log_frame, height=6, bg='#2a2a2a', fg='white',
                                   font=('Courier', 9), relief='flat')
        
        activity_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.activity_text.yview)
        self.activity_text.configure(yscrollcommand=activity_scroll.set)
        
        self.activity_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        activity_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_footer(self):
        """Create the footer with additional info"""
        footer_frame = ttk.Frame(self.root, style='Dashboard.TFrame')
        footer_frame.pack(fill=tk.X, padx=20, pady=(10, 20))
        
        self.footer_label = ttk.Label(footer_frame, 
                                     text="Ready to start automated trading...",
                                     style='Status.TLabel')
        self.footer_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(footer_frame, 
                                   text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                   style='Status.TLabel')
        self.time_label.pack(side=tk.RIGHT)
    
    def start_trading(self):
        """Start automated trading"""
        if not self.is_trading:
            self.is_trading = True
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            
            self.status_label.configure(text="● ONLINE", foreground=self.colors['accent_green'])
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.log_activity("🚀 Automated trading started")
            logger.info("🚀 Automated trading started")
    
    def stop_trading(self):
        """Stop automated trading"""
        if self.is_trading:
            self.is_trading = False
            self.start_button.configure(state='normal')
            self.stop_button.configure(state='disabled')
            
            self.status_label.configure(text="● OFFLINE", foreground=self.colors['accent_red'])
            
            self.log_activity("⏹ Automated trading stopped")
            logger.info("⏹ Automated trading stopped")
    
    def trading_loop(self):
        """Main trading loop running in separate thread"""
        while self.is_trading:
            try:
                # Simulate trading activity
                self.simulate_trading_activity()
                
                # Update metrics
                self.update_trading_metrics()
                
                # Sleep for a bit
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"❌ Trading loop error: {e}")
                self.log_activity(f"❌ Error: {str(e)}")
    
    def simulate_trading_activity(self):
        """Simulate trading activity for demonstration"""
        # Simulate finding and executing trades
        if random.random() < 0.1:  # 10% chance of new trade
            symbols = ['BTC', 'ETH', 'SOL', 'TOKEN1', 'TOKEN2', 'MATIC']
            symbol = random.choice(symbols)
            side = 'LONG'
            size = random.uniform(50, 200)
            
            self.metrics.total_trades += 1
            self.metrics.active_positions += 1
            
            self.log_activity(f"🎯 NEW TRADE: {symbol} {side} ${size:.2f}")
        
        # Simulate position updates
        if random.random() < 0.2:  # 20% chance of position update
            if self.metrics.active_positions > 0:
                symbols = ['BTC', 'ETH', 'SOL', 'TOKEN1', 'TOKEN2']
                symbol = random.choice(symbols)
                pnl = random.uniform(-20, 50)
                
                if pnl > 0:
                    self.metrics.winning_trades += 1
                    self.log_activity(f"✅ PROFIT: {symbol} +${pnl:.2f}")
                else:
                    self.metrics.losing_trades += 1
                    self.log_activity(f"❌ LOSS: {symbol} ${pnl:.2f}")
                
                self.metrics.daily_pnl += pnl
                self.metrics.active_positions = max(0, self.metrics.active_positions - 1)
    
    def update_trading_metrics(self):
        """Update trading metrics"""
        # Calculate derived metrics
        if self.metrics.total_trades > 0:
            self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.total_trades) * 100
        
        self.metrics.total_balance = 1000.0 + self.metrics.daily_pnl
        self.metrics.available_capital = self.metrics.total_balance * 0.8  # 80% available
        
        if self.metrics.total_balance > 0:
            self.metrics.daily_pnl_pct = (self.metrics.daily_pnl / 1000.0) * 100
        
        # Update sentiment
        if self.metrics.daily_pnl > 20:
            self.sentiment.overall_sentiment = "BULLISH"
            self.sentiment.confidence = min(0.9, 0.5 + (self.metrics.daily_pnl / 100))
        elif self.metrics.daily_pnl < -10:
            self.sentiment.overall_sentiment = "BEARISH"
            self.sentiment.confidence = min(0.9, 0.5 + (abs(self.metrics.daily_pnl) / 100))
        else:
            self.sentiment.overall_sentiment = "NEUTRAL"
            self.sentiment.confidence = 0.5
        
        # Add to history
        current_time = datetime.now()
        self.balance_history.append((current_time, self.metrics.total_balance))
        self.pnl_history.append((current_time, self.metrics.daily_pnl))
        
        # Keep only last 100 points
        if len(self.balance_history) > 100:
            self.balance_history = self.balance_history[-100:]
            self.pnl_history = self.pnl_history[-100:]
    
    def update_display(self):
        """Update the display with current data"""
        try:
            # Update metrics
            for key, (label, format_str) in self.metric_labels.items():
                value = getattr(self.metrics, key, 0)
                if key == 'win_rate':
                    label.configure(text=format_str.format(value))
                elif 'pnl' in key and value >= 0:
                    label.configure(text=format_str.format(value), foreground=self.colors['accent_green'])
                elif 'pnl' in key and value < 0:
                    label.configure(text=format_str.format(value), foreground=self.colors['accent_red'])
                else:
                    label.configure(text=format_str.format(value))
            
            # Update sentiment
            sentiment_color = {
                'BULLISH': self.colors['accent_green'],
                'BEARISH': self.colors['accent_red'],
                'NEUTRAL': self.colors['text_secondary']
            }
            
            self.sentiment_label.configure(
                text=self.sentiment.overall_sentiment,
                foreground=sentiment_color.get(self.sentiment.overall_sentiment, self.colors['text_secondary'])
            )
            
            confidence_pct = self.sentiment.confidence * 100
            self.confidence_var.set(confidence_pct)
            self.confidence_label.configure(text=f"{confidence_pct:.0f}%")
            
            # Update chart
            self.update_chart()
            
            # Update time
            self.time_label.configure(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        except Exception as e:
            logger.error(f"❌ Display update error: {e}")
        
        # Schedule next update
        self.root.after(1000, self.update_display)  # Update every second
    
    def update_chart(self):
        """Update the performance chart"""
        if len(self.balance_history) < 2:
            return
        
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        
        # Extract data
        times = [entry[0] for entry in self.balance_history]
        balances = [entry[1] for entry in self.balance_history]
        
        # Plot balance line
        self.ax.plot(times, balances, color=self.colors['accent_green'], linewidth=2, label='Balance')
        
        # Format chart
        self.ax.set_title('Portfolio Balance', color='white', fontsize=12)
        self.ax.set_ylabel('Balance ($)', color='white')
        self.ax.tick_params(colors='white')
        
        # Format x-axis
        if len(times) > 1:
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        
        # Style
        for spine in self.ax.spines.values():
            spine.set_color('white')
        
        self.ax.grid(True, alpha=0.3, color='white')
        
        self.canvas.draw()
    
    def log_activity(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        
        self.activity_text.insert(tk.END, formatted_message)
        self.activity_text.see(tk.END)
        
        # Keep only last 50 lines
        lines = self.activity_text.get("1.0", tk.END).split('\n')
        if len(lines) > 50:
            self.activity_text.delete("1.0", f"{len(lines)-50}.0")
    
    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            logger.info("🛑 Dashboard shutdown requested")
        finally:
            self.stop_trading()

def main():
    """Main entry point"""
    dashboard = ModernTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()