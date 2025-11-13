"""
Alternative Price Data Sources
Bypasses geo-restrictions by using multiple unrestricted APIs
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional
import time

logger = logging.getLogger(__name__)

class AlternativePriceProvider:
    """Multiple unrestricted price data sources"""
    
    def __init__(self):
        self.price_cache = {}
        self.cache_duration = 30  # Cache prices for 30 seconds
        
        # Alternative APIs that don't have geo-restrictions
        self.apis = {
            'coinpaprika': 'https://api.coinpaprika.com/v1',
            'cryptocompare': 'https://min-api.cryptocompare.com/data',
            'coinlore': 'https://api.coinlore.net/api',
            'kraken': 'https://api.kraken.com/0/public',
            'kucoin': 'https://api.kucoin.com/api/v1',
            'mexc': 'https://api.mexc.com/api/v3'
        }
        
        # Symbol mappings for different exchanges
        self.symbol_mappings = {
            'coinpaprika': {
                'BTC': 'btc-bitcoin',
                'ETH': 'eth-ethereum', 
                'SOL': 'sol-solana',
                'ADA': 'ada-cardano',
                'MATIC': 'matic-polygon',
                'DOGE': 'doge-dogecoin',
                'LINK': 'link-chainlink',
                'UNI': 'uni-uniswap',
                'AVAX': 'avax-avalanche',
                'USDC': 'usdc-usd-coin',
                'USDT': 'usdt-tether'
            },
            'kraken': {
                'BTC': 'XXBTZUSD',
                'ETH': 'XETHZUSD',
                'SOL': 'SOLUSD',
                'ADA': 'ADAUSD',
                'MATIC': 'MATICUSD',
                'DOGE': 'XDGUSD',
                'LINK': 'LINKUSD',
                'UNI': 'UNIUSD',
                'AVAX': 'AVAXUSD'
            },
            'kucoin': {
                'BTC': 'BTC-USDT',
                'ETH': 'ETH-USDT',
                'SOL': 'SOL-USDT',
                'ADA': 'ADA-USDT',
                'MATIC': 'MATIC-USDT',
                'DOGE': 'DOGE-USDT',
                'LINK': 'LINK-USDT',
                'UNI': 'UNI-USDT',
                'AVAX': 'AVAX-USDT'
            }
        }
        
        logger.info("💰 Alternative price provider initialized")
    
    async def get_price(self, symbol: str) -> Optional[float]:
        """Get price from multiple sources with fallback"""
        symbol_clean = symbol.replace('USDT', '').replace('USDC', '').upper()
        
        # Check cache first
        cache_key = f"{symbol_clean}_price"
        if cache_key in self.price_cache:
            cached_data = self.price_cache[cache_key]
            if time.time() - cached_data['timestamp'] < self.cache_duration:
                return cached_data['price']
        
        # Try multiple APIs in order of reliability
        price_sources = [
            self._get_coinpaprika_price,
            self._get_kraken_price,
            self._get_kucoin_price,
            self._get_cryptocompare_price,
            self._get_mexc_price
        ]
        
        for source in price_sources:
            try:
                price = await source(symbol_clean)
                if price and price > 0:
                    # Cache the result
                    self.price_cache[cache_key] = {
                        'price': price,
                        'timestamp': time.time(),
                        'source': source.__name__
                    }
                    logger.info(f"💰 Real price from {source.__name__} for {symbol_clean}: ${price}")
                    return price
            except Exception as e:
                logger.warning(f"⚠️ {source.__name__} failed for {symbol_clean}: {e}")
                continue
        
        logger.error(f"❌ All price sources failed for {symbol_clean}")
        return None
    
    async def _get_coinpaprika_price(self, symbol: str) -> Optional[float]:
        """Get price from CoinPaprika (no restrictions)"""
        coin_id = self.symbol_mappings['coinpaprika'].get(symbol)
        if not coin_id:
            return None
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.apis['coinpaprika']}/tickers/{coin_id}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data['quotes']['USD']['price'])
        return None
    
    async def _get_kraken_price(self, symbol: str) -> Optional[float]:
        """Get price from Kraken (no restrictions)"""
        kraken_symbol = self.symbol_mappings['kraken'].get(symbol)
        if not kraken_symbol:
            return None
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.apis['kraken']}/Ticker?pair={kraken_symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'result' in data and kraken_symbol in data['result']:
                        return float(data['result'][kraken_symbol]['c'][0])
        return None
    
    async def _get_kucoin_price(self, symbol: str) -> Optional[float]:
        """Get price from KuCoin (no restrictions)"""
        kucoin_symbol = self.symbol_mappings['kucoin'].get(symbol)
        if not kucoin_symbol:
            return None
            
        async with aiohttp.ClientSession() as session:
            url = f"{self.apis['kucoin']}/market/orderbook/level1?symbol={kucoin_symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'data' in data and data['data']:
                        return float(data['data']['price'])
        return None
    
    async def _get_cryptocompare_price(self, symbol: str) -> Optional[float]:
        """Get price from CryptoCompare (no restrictions)"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.apis['cryptocompare']}/price?fsym={symbol}&tsyms=USD"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'USD' in data:
                        return float(data['USD'])
        return None
    
    async def _get_mexc_price(self, symbol: str) -> Optional[float]:
        """Get price from MEXC (no restrictions)"""
        mexc_symbol = f"{symbol}USDT"
        async with aiohttp.ClientSession() as session:
            url = f"{self.apis['mexc']}/ticker/price?symbol={mexc_symbol}"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'price' in data:
                        return float(data['price'])
        return None
    
    async def get_multiple_prices(self, symbols: list) -> Dict[str, float]:
        """Get prices for multiple symbols efficiently"""
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_price(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_dict = {}
        for i, symbol in enumerate(symbols):
            if not isinstance(results[i], Exception) and results[i]:
                price_dict[symbol] = results[i]
        
        return price_dict


# Global instance
_price_provider = None

async def get_alternative_price_provider() -> AlternativePriceProvider:
    """Get or create the global price provider"""
    global _price_provider
    if _price_provider is None:
        _price_provider = AlternativePriceProvider()
    return _price_provider

async def get_unrestricted_price(symbol: str) -> Optional[float]:
    """Get price from unrestricted sources"""
    provider = await get_alternative_price_provider()
    return await provider.get_price(symbol)