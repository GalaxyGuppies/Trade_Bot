#!/usr/bin/env python3
"""
12-Hour Scalping Bot Test Runner
Runs the trading bot continuously for 12 hours with enhanced monitoring
"""

import subprocess
import time
import logging
import os
from datetime import datetime, timedelta

# Setup logging for the test run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('12_hour_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run 12-hour scalping test"""
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=12)
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING 12-HOUR SCALPING BOT TEST")
    logger.info(f"📅 Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🏁 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)
    
    bot_process = None
    restart_count = 0
    
    try:
        while datetime.now() < end_time:
            try:
                # Start the bot
                if bot_process is None or bot_process.poll() is not None:
                    if bot_process is not None:
                        restart_count += 1
                        logger.warning(f"🔄 Bot crashed, restarting... (restart #{restart_count})")
                    
                    logger.info("🤖 Starting scalping bot...")
                    bot_process = subprocess.Popen(
                        ['python', 'advanced_microcap_gui.py'],
                        cwd=os.path.dirname(os.path.abspath(__file__)),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                
                # Check bot status every 5 minutes
                time.sleep(300)  # 5 minutes
                
                # Log progress
                current_time = datetime.now()
                elapsed = current_time - start_time
                remaining = end_time - current_time
                
                logger.info(f"⏱️  PROGRESS: {elapsed} elapsed, {remaining} remaining")
                logger.info(f"🔄 Restarts: {restart_count}")
                
            except KeyboardInterrupt:
                logger.info("❌ Test interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Test error: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    finally:
        # Clean shutdown
        if bot_process and bot_process.poll() is None:
            logger.info("🛑 Stopping bot...")
            bot_process.terminate()
            bot_process.wait()
        
        # Final report
        actual_end_time = datetime.now()
        total_runtime = actual_end_time - start_time
        
        logger.info("=" * 60)
        logger.info("📊 12-HOUR TEST COMPLETED")
        logger.info(f"⏱️  Total Runtime: {total_runtime}")
        logger.info(f"🔄 Total Restarts: {restart_count}")
        logger.info(f"📈 Check trading_bot.db for trade results")
        logger.info(f"📋 Full log available in: 12_hour_test.log")
        logger.info("=" * 60)

if __name__ == "__main__":
    main()