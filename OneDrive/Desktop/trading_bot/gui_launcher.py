"""
GUI Launcher for Advanced Trading Dashboard
Launches the comprehensive trading interface with sentiment analysis
"""

import sys
import os
import asyncio
import tkinter as tk
from tkinter import messagebox
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.gui.trading_dashboard import create_trading_gui
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

def setup_logging():
    """Setup logging for the GUI"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('gui.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main GUI launcher"""
    setup_logging()
    
    # Trading bot configuration
    config = {
        'market_data': {
            'coinmarketcap_api_key': '6cad35f36d7b4e069b8dcb0eb9d17d56',
            'coingecko_api_key': 'CG-uKph8trS6RiycsxwVQtxfxvF'
        },
        'dappradar_api_key': 'xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA',
        'symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
        'max_position_size': 10000,
        'risk_tolerance': 0.02,
        'confidence_threshold': 0.7
    }
    
    try:
        print("🚀 Launching Advanced Trading Dashboard...")
        print("=" * 50)
        print("Features:")
        print("• Real-time market data with sentiment analysis")
        print("• Comprehensive trade ledger with statistics")
        print("• DeFi analytics and protocol rankings")
        print("• Adaptive position scaling visualization")
        print("• Performance analytics and charts")
        print("=" * 50)
        
        # Launch the GUI
        create_trading_gui(config)
        
    except Exception as e:
        error_msg = f"Failed to launch GUI: {e}"
        logging.error(error_msg)
        
        # Show error dialog
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Launch Error", error_msg)
        root.destroy()

if __name__ == "__main__":
    main()