"""
Gas Fee Management for Trading Bot
Automatically reserves capital for gas fees and calculates available trading capital
"""

import json
import logging
from typing import Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class GasFeeManager:
    """
    Manages gas fee allocation and capital reservation for trading operations
    """
    
    def __init__(self, config_path: str = 'config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        
        # Gas fee settings
        self.gas_config = self.config.get('trading', {}).get('gas_fee_allocation', {})
        
        # Default gas reserves if not configured
        self.eth_gas_reserve = self.gas_config.get('ethereum_gas_reserve', 0.05)  # ETH
        self.sol_gas_reserve = self.gas_config.get('solana_gas_reserve', 0.1)    # SOL
        self.base_gas_per_trade = self.gas_config.get('base_gas_per_trade', 0.01)
        self.max_gas_per_day = self.gas_config.get('max_gas_per_day', 2.0)
        self.emergency_reserve = self.gas_config.get('emergency_gas_reserve', 1.0)
        
        # Track daily gas usage
        self.daily_gas_used = 0.0
        self.last_reset_date = datetime.now().date()
        
        logger.info(f"💰 Gas fee manager initialized - Reserved: ETH({self.eth_gas_reserve}) SOL({self.sol_gas_reserve})")
    
    def load_config(self) -> Dict:
        """Load trading configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def reset_daily_gas_if_needed(self):
        """Reset daily gas usage counter if new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_gas_used = 0.0
            self.last_reset_date = current_date
            logger.info("📅 Daily gas usage counter reset")
    
    def calculate_available_trading_capital(self, total_wallet_balance: float, chain: str = 'solana') -> Tuple[float, Dict]:
        """
        Calculate available trading capital after reserving gas fees
        
        Args:
            total_wallet_balance: Total wallet balance in USD
            chain: Blockchain being used ('ethereum' or 'solana')
            
        Returns:
            Tuple of (available_capital, gas_breakdown)
        """
        self.reset_daily_gas_if_needed()
        
        # For scalping with limited capital, use minimal gas reserves
        if total_wallet_balance < 100:  # Scalping mode with limited capital
            # Minimal reserves for scalping
            if chain.lower() == 'ethereum':
                base_gas_reserve = min(5.0, total_wallet_balance * 0.1)  # Max $5 or 10% of balance
            else:  # Solana
                base_gas_reserve = min(2.0, total_wallet_balance * 0.05)  # Max $2 or 5% of balance
            
            emergency_reserve_usd = min(1.0, total_wallet_balance * 0.02)  # Max $1 or 2% of balance
            remaining_daily_gas = min(1.0, total_wallet_balance * 0.02)   # Max $1 or 2% of balance
        else:
            # Normal mode gas reserves
            if chain.lower() == 'ethereum':
                base_gas_reserve = self.eth_gas_reserve * 3000  # Assume $3000 ETH
            else:  # Solana
                base_gas_reserve = self.sol_gas_reserve * 180   # Assume $180 SOL
            
            emergency_reserve_usd = self.emergency_reserve
            remaining_daily_gas = max(0, self.max_gas_per_day - self.daily_gas_used)
        
        # Total gas reserves needed
        total_gas_reserve = base_gas_reserve + emergency_reserve_usd + remaining_daily_gas
        
        # Available capital
        available_capital = max(0, total_wallet_balance - total_gas_reserve)
        
        # Gas breakdown for logging
        gas_breakdown = {
            'total_wallet_balance': total_wallet_balance,
            'base_gas_reserve': base_gas_reserve,
            'emergency_reserve': emergency_reserve_usd,
            'remaining_daily_gas_allowance': remaining_daily_gas,
            'total_reserved_for_gas': total_gas_reserve,
            'available_for_trading': available_capital,
            'daily_gas_used': self.daily_gas_used,
            'gas_utilization_pct': (self.daily_gas_used / self.max_gas_per_day * 100) if self.max_gas_per_day > 0 else 0
        }
        
        logger.info(f"💰 Capital Allocation: Total=${total_wallet_balance:.2f}, Reserved=${total_gas_reserve:.2f}, Available=${available_capital:.2f}")
        
        return available_capital, gas_breakdown
    
    def estimate_trade_gas_cost(self, trade_amount: float, chain: str = 'solana') -> float:
        """
        Estimate gas cost for a trade
        
        Args:
            trade_amount: USD amount of trade
            chain: Blockchain being used
            
        Returns:
            Estimated gas cost in USD
        """
        if chain.lower() == 'ethereum':
            # Ethereum gas estimation
            if trade_amount < 100:
                return 5.0    # Small trade
            elif trade_amount < 1000:
                return 15.0   # Medium trade
            else:
                return 30.0   # Large trade
        else:
            # Solana gas estimation (much cheaper)
            if trade_amount < 100:
                return 0.01   # Small trade
            elif trade_amount < 1000:
                return 0.02   # Medium trade
            else:
                return 0.05   # Large trade
    
    def can_afford_trade(self, trade_amount: float, available_capital: float, chain: str = 'solana') -> Tuple[bool, str]:
        """
        Check if we can afford a trade including gas costs
        
        Returns:
            Tuple of (can_afford, reason)
        """
        self.reset_daily_gas_if_needed()
        
        # Estimate gas cost
        estimated_gas = self.estimate_trade_gas_cost(trade_amount, chain)
        
        # Total cost including gas
        total_cost = trade_amount + estimated_gas
        
        # Check if we have enough capital
        if total_cost > available_capital:
            return False, f"Insufficient capital: need ${total_cost:.2f}, have ${available_capital:.2f}"
        
        # Check daily gas limit
        if self.daily_gas_used + estimated_gas > self.max_gas_per_day:
            remaining_gas = self.max_gas_per_day - self.daily_gas_used
            return False, f"Daily gas limit exceeded: need ${estimated_gas:.2f}, remaining ${remaining_gas:.2f}"
        
        return True, "Trade affordable"
    
    def record_gas_usage(self, gas_cost: float):
        """Record actual gas usage"""
        self.daily_gas_used += gas_cost
        logger.info(f"⛽ Gas used: ${gas_cost:.4f}, Daily total: ${self.daily_gas_used:.2f}/{self.max_gas_per_day:.2f}")
    
    def get_position_size_with_gas(self, available_capital: float, risk_percentage: float, chain: str = 'solana') -> float:
        """
        Calculate position size accounting for gas costs
        
        Args:
            available_capital: Available trading capital
            risk_percentage: Risk percentage (e.g., 0.25 for 25%)
            chain: Blockchain being used
            
        Returns:
            Recommended position size
        """
        # Base position size
        base_position = available_capital * risk_percentage
        
        # Estimate gas cost
        estimated_gas = self.estimate_trade_gas_cost(base_position, chain)
        
        # Reduce position size to account for gas
        adjusted_position = base_position - estimated_gas
        
        # Ensure minimum viable position
        min_position = 5.0 if chain.lower() == 'ethereum' else 1.0
        
        return max(min_position, adjusted_position)
    
    def get_gas_status(self) -> Dict:
        """Get current gas usage status"""
        self.reset_daily_gas_if_needed()
        
        return {
            'daily_gas_used': self.daily_gas_used,
            'daily_gas_limit': self.max_gas_per_day,
            'remaining_gas_budget': self.max_gas_per_day - self.daily_gas_used,
            'utilization_pct': (self.daily_gas_used / self.max_gas_per_day * 100) if self.max_gas_per_day > 0 else 0,
            'eth_reserve': self.eth_gas_reserve,
            'sol_reserve': self.sol_gas_reserve,
            'emergency_reserve': self.emergency_reserve
        }
    
    def update_gas_reserves(self, eth_reserve: float = None, sol_reserve: float = None, daily_limit: float = None):
        """Update gas reserve settings"""
        if eth_reserve is not None:
            self.eth_gas_reserve = eth_reserve
        if sol_reserve is not None:
            self.sol_gas_reserve = sol_reserve
        if daily_limit is not None:
            self.max_gas_per_day = daily_limit
        
        logger.info(f"💰 Gas reserves updated: ETH({self.eth_gas_reserve}) SOL({self.sol_gas_reserve}) Daily({self.max_gas_per_day})")


# Integration example
def integrate_with_trading_bot():
    """Example of how to integrate gas management with trading bot"""
    
    # Initialize gas manager
    gas_manager = GasFeeManager()
    
    # Example wallet balance (from your SOL wallet)
    total_balance = 47.66  # Your current balance
    
    # Calculate available capital
    available_capital, gas_breakdown = gas_manager.calculate_available_trading_capital(total_balance, 'solana')
    
    print(f"💰 CAPITAL ALLOCATION:")
    print(f"   Total Balance: ${total_balance:.2f}")
    print(f"   Reserved for Gas: ${gas_breakdown['total_reserved_for_gas']:.2f}")
    print(f"   Available for Trading: ${available_capital:.2f}")
    print(f"   Daily Gas Used: ${gas_breakdown['daily_gas_used']:.2f}/{gas_breakdown['remaining_daily_gas_allowance']:.2f}")
    
    # Example trade scenarios
    trade_amounts = [5.0, 10.0, 15.0, 20.0]
    
    print(f"\n📊 TRADE SCENARIOS:")
    for amount in trade_amounts:
        can_afford, reason = gas_manager.can_afford_trade(amount, available_capital, 'solana')
        position_size = gas_manager.get_position_size_with_gas(available_capital, amount/available_capital, 'solana')
        gas_cost = gas_manager.estimate_trade_gas_cost(amount, 'solana')
        
        status = "✅" if can_afford else "❌"
        print(f"   {status} ${amount:.0f} trade: Gas=${gas_cost:.3f}, Adjusted=${position_size:.2f} - {reason}")


if __name__ == "__main__":
    integrate_with_trading_bot()