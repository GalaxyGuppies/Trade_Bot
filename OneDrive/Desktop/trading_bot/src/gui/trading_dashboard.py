"""
Advanced Trading GUI with Sentiment Analysis and Trade Ledger
Real-time dashboard with comprehensive trading analytics
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import threading
import json
from typing import Dict, List, Optional
import logging

# Import our custom modules
from src.data.trading_ledger import TradingLedger, TradeRecord
from src.data.unified_market_provider import UnifiedMarketDataProvider
from src.data.dappradar_provider import DappRadarProvider
from src.risk.adaptive_scaling import AdaptiveProfitScaling

logger = logging.getLogger(__name__)

class TradingDashboard:
    """
    Comprehensive trading dashboard with real-time updates
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.ledger = TradingLedger()
        self.market_provider = UnifiedMarketDataProvider(config)
        self.dappradar = DappRadarProvider(config.get('dappradar_api_key', ''))
        self.adaptive_scaler = AdaptiveProfitScaling()
        
        # GUI state
        self.root = None
        self.running = False
        self.update_interval = 5000  # 5 seconds
        
        # Data storage
        self.current_prices = {}
        self.sentiment_data = {}
        self.performance_metrics = {}
        
        self.setup_gui()
    
    def setup_gui(self):
        """Initialize the main GUI"""
        self.root = tk.Tk()
        self.root.title("Advanced Trading Dashboard - Sentiment & Analytics")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Dark.TFrame', background='#2d2d2d')
        style.configure('Dark.TLabel', background='#2d2d2d', foreground='white')
        style.configure('Dark.TButton', background='#404040', foreground='white')
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Real-time Dashboard
        self.create_dashboard_tab(notebook)
        
        # Tab 2: Trade Ledger
        self.create_ledger_tab(notebook)
        
        # Tab 3: Sentiment Analysis
        self.create_sentiment_tab(notebook)
        
        # Tab 4: Performance Analytics
        self.create_analytics_tab(notebook)
        
        # Tab 5: DeFi Analytics
        self.create_defi_tab(notebook)
        
        # Control panel
        self.create_control_panel(main_frame)
    
    def create_dashboard_tab(self, parent):
        """Create real-time dashboard tab"""
        dashboard_frame = ttk.Frame(parent, style='Dark.TFrame')
        parent.add(dashboard_frame, text='📊 Dashboard')
        
        # Market overview section
        market_frame = ttk.LabelFrame(dashboard_frame, text="Market Overview", style='Dark.TFrame')
        market_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Price display
        price_frame = ttk.Frame(market_frame, style='Dark.TFrame')
        price_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create price labels
        self.price_labels = {}
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        for i, symbol in enumerate(symbols):
            symbol_frame = ttk.Frame(price_frame, style='Dark.TFrame')
            symbol_frame.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
            
            ttk.Label(symbol_frame, text=symbol, font=('Arial', 12, 'bold'), 
                     style='Dark.TLabel').pack()
            
            self.price_labels[symbol] = ttk.Label(symbol_frame, text="$0.00", 
                                                 font=('Arial', 16), style='Dark.TLabel')
            self.price_labels[symbol].pack()
        
        # Configure grid weights
        for i in range(len(symbols)):
            price_frame.columnconfigure(i, weight=1)
        
        # Sentiment overview
        sentiment_frame = ttk.LabelFrame(dashboard_frame, text="Sentiment Overview", style='Dark.TFrame')
        sentiment_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sentiment_labels = {}
        sentiment_types = ['Market', 'Social', 'DeFi', 'Whale Activity']
        
        for i, sent_type in enumerate(sentiment_types):
            sent_frame = ttk.Frame(sentiment_frame, style='Dark.TFrame')
            sent_frame.grid(row=0, column=i, padx=10, pady=5, sticky='ew')
            
            ttk.Label(sent_frame, text=sent_type, font=('Arial', 10, 'bold'),
                     style='Dark.TLabel').pack()
            
            self.sentiment_labels[sent_type] = ttk.Label(sent_frame, text="Neutral",
                                                        font=('Arial', 12), style='Dark.TLabel')
            self.sentiment_labels[sent_type].pack()
        
        for i in range(len(sentiment_types)):
            sentiment_frame.columnconfigure(i, weight=1)
        
        # Position scaling info
        scaling_frame = ttk.LabelFrame(dashboard_frame, text="Adaptive Position Scaling", style='Dark.TFrame')
        scaling_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.scaling_info = ttk.Label(scaling_frame, text="Current Scale: 1.0x | Total Profit: $0.00",
                                     font=('Arial', 12), style='Dark.TLabel')
        self.scaling_info.pack(pady=5)
        
        # Chart area
        chart_frame = ttk.LabelFrame(dashboard_frame, text="Price Chart", style='Dark.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_price_chart(chart_frame)
    
    def create_ledger_tab(self, parent):
        """Create trade ledger tab"""
        ledger_frame = ttk.Frame(parent, style='Dark.TFrame')
        parent.add(ledger_frame, text='📋 Trade Ledger')
        
        # Controls
        controls_frame = ttk.Frame(ledger_frame, style='Dark.TFrame')
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="Refresh Trades", 
                  command=self.refresh_trade_ledger).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(controls_frame, text="Export CSV", 
                  command=self.export_trades).pack(side=tk.LEFT, padx=5)
        
        # Trade list
        list_frame = ttk.LabelFrame(ledger_frame, text="Recent Trades", style='Dark.TFrame')
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Treeview for trades
        self.trade_tree = ttk.Treeview(list_frame, columns=(
            'Time', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L', 'Sentiment', 'Strategy'
        ), show='headings', height=15)
        
        # Configure columns
        columns = ['Time', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L', 'Sentiment', 'Strategy']
        for col in columns:
            self.trade_tree.heading(col, text=col)
            self.trade_tree.column(col, width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.trade_tree.yview)
        self.trade_tree.configure(yscrollcommand=scrollbar.set)
        
        self.trade_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_sentiment_tab(self, parent):
        """Create sentiment analysis tab"""
        sentiment_frame = ttk.Frame(parent, style='Dark.TFrame')
        parent.add(sentiment_frame, text='💭 Sentiment Analysis')
        
        # Sentiment metrics
        metrics_frame = ttk.LabelFrame(sentiment_frame, text="Sentiment Metrics", style='Dark.TFrame')
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sentiment_metrics = ttk.Label(metrics_frame, 
                                          text="Loading sentiment data...",
                                          font=('Arial', 11), style='Dark.TLabel')
        self.sentiment_metrics.pack(pady=10)
        
        # Sentiment chart
        chart_frame = ttk.LabelFrame(sentiment_frame, text="Sentiment vs Performance", style='Dark.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_sentiment_chart(chart_frame)
    
    def create_analytics_tab(self, parent):
        """Create performance analytics tab"""
        analytics_frame = ttk.Frame(parent, style='Dark.TFrame')
        parent.add(analytics_frame, text='📈 Analytics')
        
        # Performance summary
        summary_frame = ttk.LabelFrame(analytics_frame, text="Performance Summary", style='Dark.TFrame')
        summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.performance_summary = ttk.Label(summary_frame,
                                           text="Calculating performance metrics...",
                                           font=('Arial', 11), style='Dark.TLabel')
        self.performance_summary.pack(pady=10)
        
        # Analytics charts
        charts_frame = ttk.LabelFrame(analytics_frame, text="Performance Charts", style='Dark.TFrame')
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_analytics_charts(charts_frame)
    
    def create_defi_tab(self, parent):
        """Create DeFi analytics tab"""
        defi_frame = ttk.Frame(parent, style='Dark.TFrame')
        parent.add(defi_frame, text='🔗 DeFi Analytics')
        
        # DeFi metrics
        metrics_frame = ttk.LabelFrame(defi_frame, text="DeFi Market Metrics", style='Dark.TFrame')
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.defi_metrics = ttk.Label(metrics_frame,
                                     text="Loading DeFi analytics...",
                                     font=('Arial', 11), style='Dark.TLabel')
        self.defi_metrics.pack(pady=10)
        
        # Protocol rankings
        rankings_frame = ttk.LabelFrame(defi_frame, text="Top DeFi Protocols", style='Dark.TFrame')
        rankings_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.defi_tree = ttk.Treeview(rankings_frame, columns=(
            'Protocol', 'Chain', 'Volume', 'Users', 'TVL'
        ), show='headings', height=10)
        
        for col in ['Protocol', 'Chain', 'Volume', 'Users', 'TVL']:
            self.defi_tree.heading(col, text=col)
            self.defi_tree.column(col, width=120)
        
        self.defi_tree.pack(fill=tk.BOTH, expand=True)
    
    def create_control_panel(self, parent):
        """Create control panel"""
        control_frame = ttk.Frame(parent, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="Start/Stop Bot", 
                  command=self.toggle_bot).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Manual Refresh", 
                  command=self.manual_refresh).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Status: Ready", 
                                     style='Dark.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=5)
    
    def create_price_chart(self, parent):
        """Create price chart"""
        self.price_figure = Figure(figsize=(12, 4), dpi=100, facecolor='#2d2d2d')
        self.price_ax = self.price_figure.add_subplot(111, facecolor='#2d2d2d')
        self.price_ax.tick_params(colors='white')
        self.price_ax.set_xlabel('Time', color='white')
        self.price_ax.set_ylabel('Price', color='white')
        
        self.price_canvas = FigureCanvasTkAgg(self.price_figure, parent)
        self.price_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_sentiment_chart(self, parent):
        """Create sentiment analysis chart"""
        self.sentiment_figure = Figure(figsize=(10, 6), dpi=100, facecolor='#2d2d2d')
        self.sentiment_ax = self.sentiment_figure.add_subplot(111, facecolor='#2d2d2d')
        self.sentiment_ax.tick_params(colors='white')
        
        self.sentiment_canvas = FigureCanvasTkAgg(self.sentiment_figure, parent)
        self.sentiment_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_analytics_charts(self, parent):
        """Create analytics charts"""
        self.analytics_figure = Figure(figsize=(12, 8), dpi=100, facecolor='#2d2d2d')
        self.analytics_canvas = FigureCanvasTkAgg(self.analytics_figure, parent)
        self.analytics_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    async def update_market_data(self):
        """Update market data"""
        try:
            symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
            for symbol in symbols:
                data = await self.market_provider.get_enhanced_data(symbol)
                if data:
                    display_symbol = symbol.replace('-', '/')
                    self.current_prices[display_symbol] = data
                    
                    # Update price display
                    if display_symbol in self.price_labels:
                        price = data.get('price', 0)
                        self.price_labels[display_symbol].config(text=f"${price:,.2f}")
            
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def update_sentiment_data(self):
        """Update sentiment analysis"""
        try:
            # Get DeFi sentiment
            defi_sentiment = await self.dappradar.get_defi_sentiment()
            
            # Update sentiment labels
            if 'Market' in self.sentiment_labels:
                self.sentiment_labels['Market'].config(text="Bullish")
            
            if 'DeFi' in self.sentiment_labels:
                sentiment_text = defi_sentiment.get('sentiment', 'neutral').title()
                self.sentiment_labels['DeFi'].config(text=sentiment_text)
            
        except Exception as e:
            logger.error(f"Error updating sentiment: {e}")
    
    def refresh_trade_ledger(self):
        """Refresh trade ledger display"""
        try:
            # Clear existing items
            for item in self.trade_tree.get_children():
                self.trade_tree.delete(item)
            
            # Get recent trades
            trades = self.ledger.get_trade_history(50)
            
            for trade in trades:
                timestamp = datetime.fromisoformat(trade['timestamp']).strftime('%H:%M:%S')
                self.trade_tree.insert('', 'end', values=(
                    timestamp,
                    trade['symbol'],
                    trade['side'].upper(),
                    f"{trade['quantity']:.4f}",
                    f"${trade['price']:.2f}",
                    f"${trade['pnl_realized']:.2f}",
                    trade['market_sentiment'],
                    trade['strategy_used']
                ))
                
        except Exception as e:
            logger.error(f"Error refreshing trade ledger: {e}")
    
    def export_trades(self):
        """Export trades to CSV"""
        filename = self.ledger.export_to_csv()
        if filename:
            messagebox.showinfo("Export Complete", f"Trades exported to {filename}")
        else:
            messagebox.showerror("Export Failed", "Failed to export trades")
    
    def update_performance_metrics(self):
        """Update performance metrics display"""
        try:
            metrics = self.ledger.get_performance_metrics()
            
            if 'error' not in metrics:
                summary_text = f"""
Performance Summary (Last 30 Days):
• Total Trades: {metrics['total_trades']}
• Win Rate: {metrics['win_rate']}%
• Net Profit: ${metrics['net_profit']:,.2f}
• Best Strategy: {metrics['best_strategy']} (${metrics['best_strategy_pnl']:,.2f})
• Average Holding Period: {metrics['avg_holding_period_hours']:.1f} hours
                """
                self.performance_summary.config(text=summary_text.strip())
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    def update_defi_analytics(self):
        """Update DeFi analytics display"""
        async def _update():
            try:
                # Clear existing items
                for item in self.defi_tree.get_children():
                    self.defi_tree.delete(item)
                
                # Get DeFi protocols
                protocols = await self.dappradar.get_defi_rankings("defi", 10)
                
                for protocol in protocols:
                    self.defi_tree.insert('', 'end', values=(
                        protocol['name'],
                        protocol['chain'],
                        f"${protocol['volume_24h']:,.0f}",
                        f"{protocol['users_24h']:,}",
                        f"${protocol['tvl']:,.0f}"
                    ))
                
            except Exception as e:
                logger.error(f"Error updating DeFi analytics: {e}")
        
        # Run async update
        asyncio.create_task(_update())
    
    def toggle_bot(self):
        """Toggle bot running state"""
        self.running = not self.running
        status = "Running" if self.running else "Stopped"
        self.status_label.config(text=f"Status: {status}")
        
        if self.running:
            self.start_updates()
    
    def manual_refresh(self):
        """Manual refresh of all data"""
        async def _refresh():
            await self.update_market_data()
            await self.update_sentiment_data()
        
        asyncio.create_task(_refresh())
        self.refresh_trade_ledger()
        self.update_performance_metrics()
        self.update_defi_analytics()
    
    def start_updates(self):
        """Start periodic updates"""
        if self.running:
            # Schedule async updates
            asyncio.create_task(self.update_market_data())
            asyncio.create_task(self.update_sentiment_data())
            
            # Update other components
            self.refresh_trade_ledger()
            self.update_performance_metrics()
            
            # Schedule next update
            self.root.after(self.update_interval, self.start_updates)
    
    def run(self):
        """Run the GUI"""
        logger.info("Starting trading dashboard GUI...")
        
        # Initial data load
        self.manual_refresh()
        
        # Start the GUI
        self.root.mainloop()

def create_trading_gui(config: Dict):
    """Create and run the trading GUI"""
    # Add DappRadar API key to config
    config['dappradar_api_key'] = 'xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA'
    
    dashboard = TradingDashboard(config)
    dashboard.run()

if __name__ == "__main__":
    # Sample configuration
    sample_config = {
        'market_data': {
            'coinmarketcap_api_key': '6cad35f36d7b4e069b8dcb0eb9d17d56',
            'coingecko_api_key': 'CG-uKph8trS6RiycsxwVQtxfxvF'
        },
        'dappradar_api_key': 'xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA'
    }
    
    create_trading_gui(sample_config)