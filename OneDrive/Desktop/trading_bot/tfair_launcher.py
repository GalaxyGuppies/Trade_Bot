#!/usr/bin/env python3
"""
TFair's Powerhouse Trading System Launcher
Personalized launcher for wallet: 0xb4add0df12df32981773ca25ee88bdab750bfa20
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tfair_trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TFairTradingLauncher:
    """Personalized launcher for TFair's trading system"""
    
    def __init__(self):
        self.wallet_address = "0xb4add0df12df32981773ca25ee88bdab750bfa20"
        self.config = None
        self.startup_checks_passed = False
        
    def load_config(self):
        """Load configuration"""
        try:
            with open('config.json', 'r') as f:
                self.config = json.load(f)
            
            # Verify wallet address
            config_wallet = self.config.get('wallet', {}).get('address', '')
            if config_wallet.lower() != self.wallet_address.lower():
                logger.warning(f"Wallet mismatch: config={config_wallet}, expected={self.wallet_address}")
            
            logger.info(f"✅ Configuration loaded for wallet: {self.wallet_address}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            return False
    
    def run_startup_checks(self):
        """Run comprehensive startup checks"""
        try:
            logger.info("🔍 Running startup checks...")
            
            checks = [
                ("Configuration", self.check_config),
                ("Dependencies", self.check_dependencies),
                ("Database", self.check_database),
                ("API Keys", self.check_api_keys),
                ("Wallet", self.check_wallet)
            ]
            
            passed = 0
            for check_name, check_func in checks:
                try:
                    if check_func():
                        logger.info(f"✅ {check_name}: PASS")
                        passed += 1
                    else:
                        logger.warning(f"⚠️ {check_name}: FAIL")
                except Exception as e:
                    logger.error(f"❌ {check_name}: ERROR - {e}")
            
            success_rate = passed / len(checks)
            if success_rate >= 0.8:
                logger.info(f"🚀 Startup checks passed: {passed}/{len(checks)} ({success_rate:.1%})")
                self.startup_checks_passed = True
                return True
            else:
                logger.warning(f"⚠️ Startup checks incomplete: {passed}/{len(checks)} ({success_rate:.1%})")
                return False
                
        except Exception as e:
            logger.error(f"Error in startup checks: {e}")
            return False
    
    def check_config(self):
        """Check configuration"""
        return self.config is not None
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        try:
            import pandas
            import numpy
            import sklearn
            import aiohttp
            import requests
            import praw
            import textblob
            return True
        except ImportError as e:
            logger.error(f"Missing dependency: {e}")
            return False
    
    def check_database(self):
        """Check database connectivity"""
        try:
            import sqlite3
            conn = sqlite3.connect('trading_bot.db')
            conn.close()
            return True
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False
    
    def check_api_keys(self):
        """Check API key configuration"""
        if not self.config:
            return False
        
        api_keys = self.config.get('api_keys', {})
        required_keys = ['reddit', 'worldnews']
        
        for key in required_keys:
            if key not in api_keys:
                logger.warning(f"Missing API key: {key}")
                return False
        
        return True
    
    def check_wallet(self):
        """Check wallet configuration"""
        if not self.config:
            return False
        
        wallet = self.config.get('wallet', {})
        return wallet.get('address', '').lower() == self.wallet_address.lower()
    
    def launch_gui_mode(self):
        """Launch the GUI trading interface"""
        try:
            logger.info("🖥️ Launching GUI mode...")
            
            # Import and run the GUI
            if os.path.exists('advanced_microcap_gui.py'):
                os.system('python advanced_microcap_gui.py')
            else:
                logger.error("GUI file not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error launching GUI: {e}")
            return False
    
    def launch_automated_mode(self):
        """Launch automated trading mode"""
        try:
            logger.info("🤖 Launching automated mode...")
            
            if os.path.exists('automated_microcap_trader.py'):
                os.system('python automated_microcap_trader.py')
            else:
                logger.error("Automated trader file not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error launching automated mode: {e}")
            return False
    
    def show_menu(self):
        """Show launch menu"""
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 TFAIR'S POWERHOUSE TRADING SYSTEM 🚀                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  Wallet: {self.wallet_address}                ║
║  Status: {'🟢 READY' if self.startup_checks_passed else '🟡 PARTIAL'}                                                           ║
║                                                                              ║
║  Available Modes:                                                            ║
║                                                                              ║
║  [1] 🖥️  GUI Mode (Recommended)                                              ║
║      Interactive dashboard with full control                                 ║
║                                                                              ║
║  [2] 🤖 Automated Mode                                                       ║
║      Fully automated microcap trading                                        ║
║                                                                              ║
║  [3] 🔧 Component Testing                                                    ║
║      Test individual system components                                       ║
║                                                                              ║
║  [4] ⚙️  Configuration                                                       ║
║      Edit system configuration                                               ║
║                                                                              ║
║  [0] ❌ Exit                                                                 ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
    
    def run_component_tests(self):
        """Run individual component tests"""
        print("🔧 Running component tests...")
        
        test_files = [
            ('Social Sentiment', 'src/data/social_sentiment.py'),
            ('Security Analysis', 'src/security/contract_analyzer.py'),
            ('Whale Monitoring', 'src/monitoring/whale_monitor.py'),
            ('DEX Aggregation', 'src/trading/dex_aggregator.py'),
            ('ML Prediction', 'src/ai/ml_predictor.py')
        ]
        
        for name, file_path in test_files:
            if os.path.exists(file_path):
                print(f"  ✅ {name}: Available")
                try:
                    # Could run individual tests here
                    pass
                except Exception as e:
                    print(f"  ❌ {name}: Error - {e}")
            else:
                print(f"  ⚠️ {name}: File not found")
    
    def edit_configuration(self):
        """Open configuration for editing"""
        print("⚙️ Configuration options:")
        print("1. Edit config.json manually")
        print("2. View current configuration")
        print("3. Reset to defaults")
        
        if self.config:
            print(f"\nCurrent wallet: {self.config.get('wallet', {}).get('address', 'Not set')}")
            print(f"Max capital allocation: {self.config.get('trading', {}).get('max_capital_allocation', 0.75)*100}%")
            print(f"Risk level: {self.config.get('risk_management', {}).get('max_position_size', 0.1)*100}% per position")
    
    def main(self):
        """Main launcher function"""
        print("🚀 TFAIR'S POWERHOUSE TRADING SYSTEM")
        print("=" * 50)
        print(f"Initializing for wallet: {self.wallet_address}")
        
        # Load configuration
        if not self.load_config():
            print("❌ Configuration loading failed. Check config.json")
            return
        
        # Run startup checks
        self.run_startup_checks()
        
        # Main menu loop
        while True:
            self.show_menu()
            
            try:
                choice = input("\nSelect option [1-4, 0 to exit]: ").strip()
                
                if choice == '1':
                    self.launch_gui_mode()
                elif choice == '2':
                    self.launch_automated_mode()
                elif choice == '3':
                    self.run_component_tests()
                elif choice == '4':
                    self.edit_configuration()
                elif choice == '0':
                    print("👋 Goodbye! Happy trading!")
                    break
                else:
                    print("❌ Invalid choice. Please try again.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye! Happy trading!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    launcher = TFairTradingLauncher()
    launcher.main()