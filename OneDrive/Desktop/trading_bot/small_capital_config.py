"""
Small Capital Trading Configuration

Optimized setup for $100-$150 starting capital
Focus on growth with minimal API costs
"""

# Small capital trading configuration
SMALL_CAPITAL_CONFIG = {
    # Capital allocation
    'starting_capital': 125.0,          # $125 starting amount
    'max_position_size_pct': 15.0,      # Max 15% per trade (safer for small capital)
    'reserve_for_gas': 25.0,            # Keep $25 for gas fees
    'trading_capital': 100.0,           # $100 for actual trading
    
    # Position sizing for small capital
    'min_trade_size': 10.0,             # Minimum $10 per trade
    'max_trade_size': 15.0,             # Maximum $15 per trade (15% of $100)
    'max_concurrent_positions': 3,       # Only 3 positions at once
    
    # Conservative risk management
    'stop_loss_pct': 8.0,               # 8% stop loss (tight for small capital)
    'take_profit_pct': 12.0,            # 12% take profit target
    'max_daily_loss_pct': 5.0,          # Stop trading if down 5% in a day
    
    # Optimized for FREE APIs
    'monitored_symbols': ['BTC', 'ETH', 'SOL'],  # Focus on major coins
    'sentiment_check_interval': 4,       # Check every 4 hours (6x per day)
    'max_daily_api_calls': 50,          # Conservative API usage
    
    # FREE API configuration
    'api_config': {
        'lunarcrush_tier': 'free',       # 200 requests/day
        'santiment_tier': 'free',        # 1000 requests/month  
        'messari_tier': 'free',          # Free tier
        'use_twitter': True,             # Your existing API
        'use_openai': False,             # Skip paid APIs initially
    },
    
    # Growth strategy
    'growth_targets': {
        'phase_1': 250.0,               # Grow to $250
        'phase_2': 500.0,               # Then to $500
        'phase_3': 1000.0,              # Then to $1000
        'upgrade_apis_at': 500.0,       # Consider paid APIs at $500+
    },
    
    # Small capital specific features
    'features': {
        'compound_profits': True,        # Reinvest all profits
        'fractional_trading': True,      # Allow fractional positions
        'gas_optimization': True,        # Optimize for gas costs
        'focus_on_majors': True,        # Stick to BTC, ETH, SOL
        'avoid_low_liquidity': True,    # Skip illiquid altcoins
    }
}

def get_small_capital_config():
    """Return the small capital configuration"""
    return SMALL_CAPITAL_CONFIG

def calculate_position_size(capital: float, confidence: float, max_pct: float = 15.0) -> float:
    """
    Calculate position size for small capital trading
    
    Args:
        capital: Available trading capital
        confidence: Signal confidence (0.0 to 1.0)
        max_pct: Maximum percentage per trade
    
    Returns:
        Position size in USD
    """
    # Scale position size by confidence
    position_pct = max_pct * confidence
    
    # Ensure minimum viable trade size
    min_trade = 10.0
    max_trade = capital * (position_pct / 100.0)
    
    return max(min_trade, min(max_trade, 25.0))  # Cap at $25 for safety

def should_trade_small_capital(
    current_capital: float, 
    daily_pnl: float, 
    max_daily_loss_pct: float = 5.0
) -> bool:
    """
    Determine if trading should continue based on small capital rules
    
    Args:
        current_capital: Current available capital
        daily_pnl: Today's P&L
        max_daily_loss_pct: Maximum daily loss percentage
    
    Returns:
        True if trading should continue
    """
    # Stop if daily loss exceeds threshold
    daily_loss_pct = abs(daily_pnl) / current_capital * 100
    if daily_pnl < 0 and daily_loss_pct > max_daily_loss_pct:
        return False
    
    # Stop if capital too low for minimum trade
    if current_capital < 25.0:  # Need at least $25 for gas + min trade
        return False
    
    return True

if __name__ == "__main__":
    config = get_small_capital_config()
    print("=== SMALL CAPITAL TRADING CONFIGURATION ===")
    print(f"Starting capital: ${config['starting_capital']}")
    print(f"Trading capital: ${config['trading_capital']}")
    print(f"Max position size: {config['max_position_size_pct']}%")
    print(f"Monitored symbols: {config['monitored_symbols']}")
    print(f"Growth target phase 1: ${config['growth_targets']['phase_1']}")
    print(f"API upgrade threshold: ${config['growth_targets']['upgrade_apis_at']}")