#!/usr/bin/env python3
"""
Headless Auto Trader - Runs duplex trading strategy without GUI
This avoids all GUI threading issues while maintaining full trading functionality.
"""

import sys
import time
import logging
from datetime import datetime
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('auto_trader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HeadlessAutoTrader:
    """Headless version of the trading bot for automation"""
    
    def __init__(self):
        """Initialize the headless trader"""
        logger.info("🚀 Initializing Headless Auto Trader...")
        
        # Import the main trading class
        from advanced_microcap_gui import AdvancedTradingGUI
        
        # Create instance without GUI
        self.trader = AdvancedTradingGUI()
        self.trader.automation_enabled = True
        
        # Disable GUI-dependent operations
        self.trader.headless_mode = True
        
        logger.info("✅ Headless trader initialized")
        logger.info(f"💰 Available capital: ${self.trader.available_capital:.2f}")
        logger.info(f"📊 Active positions: {len(self.trader.active_positions)}")
        
    def run_trading_cycle(self):
        """Run one complete trading cycle"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            logger.info(f"🔄 Starting trading cycle at {current_time}")
            
            # 1. Scan for new trading opportunities
            logger.info("🔍 Scanning for trading opportunities...")
            self.trader.scan_microcap_candidates()
            
            # 2. Monitor existing positions
            logger.info("📊 Monitoring existing positions...")
            self.trader.monitor_positions()
            
            # 3. Log current status
            logger.info(f"💰 Current capital: ${self.trader.available_capital:.2f}")
            logger.info(f"📈 Active positions: {len(self.trader.active_positions)}")
            
            # 4. Evaluate trading opportunities if automation is enabled
            if self.trader.automation_enabled and hasattr(self.trader, 'microcap_candidates'):
                logger.info("🎯 Evaluating trading opportunities...")
                self.trader.evaluate_trading_opportunities()
            
            logger.info("✅ Trading cycle completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
            return False
    
    def run_continuous(self, scan_interval_minutes=1):
        """Run continuous trading with specified interval"""
        logger.info(f"🔄 Starting continuous trading (scan every {scan_interval_minutes} minutes)")
        
        cycle_count = 0
        while True:
            try:
                cycle_count += 1
                logger.info(f"📊 === Trading Cycle #{cycle_count} ===")
                
                success = self.run_trading_cycle()
                
                if success:
                    logger.info(f"✅ Cycle #{cycle_count} completed successfully")
                else:
                    logger.warning(f"⚠️ Cycle #{cycle_count} completed with errors")
                
                # Wait for next cycle
                logger.info(f"⏰ Waiting {scan_interval_minutes} minutes until next cycle...")
                time.sleep(scan_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping continuous trading (Ctrl+C pressed)")
                break
            except Exception as e:
                logger.error(f"❌ Critical error in continuous trading: {e}")
                logger.info("⏰ Waiting 60 seconds before retry...")
                time.sleep(60)

def main():
    """Main function"""
    try:
        # Create headless trader
        trader = HeadlessAutoTrader()
        
        # Check command line arguments
        if len(sys.argv) > 1:
            if sys.argv[1] == "--once":
                # Run single cycle
                logger.info("🎯 Running single trading cycle...")
                trader.run_trading_cycle()
                
            elif sys.argv[1] == "--continuous":
                # Run continuous trading
                interval = 1  # Default 1 minute
                if len(sys.argv) > 2:
                    try:
                        interval = float(sys.argv[2])
                    except ValueError:
                        logger.warning(f"Invalid interval '{sys.argv[2]}', using default 1 minute")
                
                trader.run_continuous(interval)
                
            else:
                logger.error(f"Unknown argument: {sys.argv[1]}")
                print("Usage: python auto_trader.py [--once|--continuous [interval_minutes]]")
                
        else:
            # Default: run single cycle
            logger.info("🎯 Running single trading cycle (use --continuous for continuous trading)...")
            trader.run_trading_cycle()
            
    except Exception as e:
        logger.error(f"❌ Auto trader failed to start: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())