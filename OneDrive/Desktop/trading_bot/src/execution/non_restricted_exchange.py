"""
Non-Restricted Exchange Client
Uses alternative exchanges and APIs that don't have geo-restrictions
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import time
import hashlib
import hmac

logger = logging.getLogger(__name__)

@dataclass
class OrderResult:
    """Order execution result"""
    success: bool
    order_id: Optional[str] = None
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None
    error_message: Optional[str] = None

class NonRestrictedExchange:
    """Exchange client using non-restricted APIs"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.connected_exchanges = []
        
        # Non-restricted exchange endpoints
        self.exchanges = {
            'kucoin': {
                'base_url': 'https://api.kucoin.com',
                'trading_fee': 0.001,
                'supports_trading': True
            },
            'mexc': {
                'base_url': 'https://api.mexc.com',
                'trading_fee': 0.001,
                'supports_trading': True
            },
            'gate': {
                'base_url': 'https://api.gateio.ws',
                'trading_fee': 0.002,
                'supports_trading': True
            },
            'bybit': {
                'base_url': 'https://api.bybit.com',
                'trading_fee': 0.001,
                'supports_trading': True
            }
        }
        
        logger.info("🌍 Non-restricted exchange client initialized")
    
    async def initialize(self):
        """Initialize connections to non-restricted exchanges"""
        try:
            # Test connections to all exchanges
            for exchange_name, exchange_config in self.exchanges.items():
                if await self._test_connection(exchange_name):
                    self.connected_exchanges.append(exchange_name)
                    logger.info(f"✅ Connected to {exchange_name}")
                else:
                    logger.warning(f"⚠️ Failed to connect to {exchange_name}")
            
            if self.connected_exchanges:
                logger.info(f"🌍 Non-restricted exchanges available: {', '.join(self.connected_exchanges)}")
                return True
            else:
                logger.error("❌ No non-restricted exchanges available")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing non-restricted exchanges: {e}")
            return False
    
    async def _test_connection(self, exchange_name: str) -> bool:
        """Test connection to an exchange"""
        try:
            exchange_config = self.exchanges[exchange_name]
            
            if exchange_name == 'kucoin':
                url = f"{exchange_config['base_url']}/api/v1/timestamp"
            elif exchange_name == 'mexc':
                url = f"{exchange_config['base_url']}/api/v3/time"
            elif exchange_name == 'gate':
                url = f"{exchange_config['base_url']}/api/v4/spot/time"
            elif exchange_name == 'bybit':
                url = f"{exchange_config['base_url']}/v5/market/time"
            else:
                return False
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(url) as response:
                    return response.status == 200
                    
        except Exception as e:
            logger.debug(f"Connection test failed for {exchange_name}: {e}")
            return False
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get current price from best available exchange"""
        # Normalize symbol (remove USDT suffix for some exchanges)
        base_symbol = symbol.replace('USDT', '').replace('USDC', '')
        
        for exchange_name in self.connected_exchanges:
            try:
                price = await self._get_exchange_price(exchange_name, symbol, base_symbol)
                if price and price > 0:
                    logger.info(f"💰 Real price from {exchange_name} for {symbol}: ${price}")
                    return price
            except Exception as e:
                logger.warning(f"Price fetch failed from {exchange_name}: {e}")
                continue
        
        return None
    
    async def _get_exchange_price(self, exchange_name: str, symbol: str, base_symbol: str) -> Optional[float]:
        """Get price from specific exchange"""
        exchange_config = self.exchanges[exchange_name]
        
        try:
            if exchange_name == 'kucoin':
                return await self._get_kucoin_price(exchange_config, symbol)
            elif exchange_name == 'mexc':
                return await self._get_mexc_price(exchange_config, symbol)
            elif exchange_name == 'gate':
                return await self._get_gate_price(exchange_config, symbol)
            elif exchange_name == 'bybit':
                return await self._get_bybit_price(exchange_config, symbol)
        except Exception as e:
            logger.warning(f"Error getting price from {exchange_name}: {e}")
            return None
    
    async def _get_kucoin_price(self, config: dict, symbol: str) -> Optional[float]:
        """Get price from KuCoin"""
        kucoin_symbol = f"{symbol.replace('USDT', '')}-USDT"
        url = f"{config['base_url']}/api/v1/market/orderbook/level1?symbol={kucoin_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        return float(data['data']['price'])
        return None
    
    async def _get_mexc_price(self, config: dict, symbol: str) -> Optional[float]:
        """Get price from MEXC"""
        mexc_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
        url = f"{config['base_url']}/api/v3/ticker/price?symbol={mexc_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'price' in data:
                        return float(data['price'])
        return None
    
    async def _get_gate_price(self, config: dict, symbol: str) -> Optional[float]:
        """Get price from Gate.io"""
        gate_symbol = f"{symbol.replace('USDT', '')}_USDT"
        url = f"{config['base_url']}/api/v4/spot/tickers?currency_pair={gate_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        return float(data[0]['last'])
        return None
    
    async def _get_bybit_price(self, config: dict, symbol: str) -> Optional[float]:
        """Get price from Bybit"""
        bybit_symbol = symbol if symbol.endswith('USDT') else f"{symbol}USDT"
        url = f"{config['base_url']}/v5/market/tickers?category=spot&symbol={bybit_symbol}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data and 'list' in data['result'] and data['result']['list']:
                        return float(data['result']['list'][0]['lastPrice'])
        return None
    
    async def execute_trade(self, symbol: str, side: str, amount: float, order_type: str = "market") -> OrderResult:
        """Execute trade on best available exchange"""
        # For now, simulate trades since this requires API keys
        # This can be extended to support real trading with proper API authentication
        
        price = await self.get_price(symbol)
        if not price:
            return OrderResult(
                success=False,
                error_message="Unable to get current price"
            )
        
        # Simulate trade execution
        slippage = 0.001  # 0.1% slippage
        fee = 0.001       # 0.1% trading fee
        
        if side.upper() == 'BUY':
            filled_price = price * (1 + slippage + fee)
        else:
            filled_price = price * (1 - slippage - fee)
        
        logger.info(f"🎭 SIMULATED trade on non-restricted exchange: {side} {amount} {symbol} @ ${filled_price}")
        
        return OrderResult(
            success=True,
            order_id=f"unrestricted_{int(time.time())}",
            filled_price=filled_price,
            filled_quantity=amount
        )
    
    def is_available(self) -> bool:
        """Check if any non-restricted exchanges are available"""
        return len(self.connected_exchanges) > 0


# Global instance
_non_restricted_exchange = None

async def get_non_restricted_exchange() -> NonRestrictedExchange:
    """Get or create the global non-restricted exchange client"""
    global _non_restricted_exchange
    
    if _non_restricted_exchange is None:
        _non_restricted_exchange = NonRestrictedExchange()
        await _non_restricted_exchange.initialize()
    
    return _non_restricted_exchange