#!/usr/bin/env python3
"""
Powerhouse Trading System Setup Script
Installs dependencies and configures the system
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def create_config():
    """Create configuration files"""
    print("Creating configuration files...")
    
    config_template = '''
{
    "api_keys": {
        "reddit_client_id": "YOUR_REDDIT_CLIENT_ID",
        "reddit_client_secret": "YOUR_REDDIT_CLIENT_SECRET", 
        "worldnews_api_key": "YOUR_WORLDNEWS_API_KEY",
        "twitter_bearer_token": "YOUR_TWITTER_BEARER_TOKEN"
    },
    "trading": {
        "max_capital_allocation": 0.75,
        "default_slippage": 0.01,
        "min_liquidity_usd": 50000,
        "max_market_cap": 1500000,
        "min_market_cap": 500000
    },
    "risk_management": {
        "max_position_size": 0.1,
        "stop_loss_percentage": 0.15,
        "take_profit_percentage": 0.3
    }
}
    '''
    
    with open("config.json", "w") as f:
        f.write(config_template.strip())
    
    print("Configuration template created")

def setup_database():
    """Initialize database"""
    print("Setting up database...")
    
    # Database will be auto-created by components
    print("Database setup complete")

def main():
    """Main setup function"""
    print("POWERHOUSE TRADING SYSTEM SETUP")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("Setup failed - dependency installation error")
        return False
    
    # Create config
    create_config()
    
    # Setup database
    setup_database()
    
    print("\nSETUP COMPLETE!")
    print("Next steps:")
    print("1. Edit config.json with your API keys")
    print("2. Run: python advanced_microcap_gui.py")
    print("3. Start with paper trading mode")
    
    return True

if __name__ == "__main__":
    main()
