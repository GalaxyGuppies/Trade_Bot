"""
DexScreener Price Provider
Real-time Solana DEX prices without geo-restrictions
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional, List
import time

logger = logging.getLogger(__name__)

class DexScreenerProvider:
    """DexScreener API client for real Solana DEX prices"""
    
    def __init__(self):
        self.base_url = "https://api.dexscreener.com/latest/dex"
        self.price_cache = {}
        self.cache_duration = 30  # Cache for 30 seconds
        
        # Token contract addresses
        self.token_addresses = {
            'SOL': 'So11111111111111111111111111111111111111112',
            'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
            'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
            'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
            'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
            'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
            'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
            # 🐋 Whale-tracked tokens (Nov 2025)
            'BANGERS': '3wppuwUMAGgxnX75Aqr4W91xYWaN6RjxjCUFiPZUpump',
            'TRUMP': '6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN',
            'BASED': 'EMAGfmV5bMzYEtgda43ZmCYwmLL7SaMi2RVqaRPjpump',
            # Previous aggressive tokens
            'GXY': 'PKikg1HNZinFvMgqk76aBDY4fF1fgGYQ3tv9kKypump',
            'ACE': 'GEuuznWpn6iuQAJxLKQDVGXPtrqXHNWTk3gZqqvJpump',
            'ROI': 'vEHiuRmd8WvCkswH8Xy4VXTEMXA7JScik47XZkDbonk',
            'TROLL': '63LfDmNb3MQ8mw9MtZ2To9bEA2M71kZUUGq5tiJxcqj9', 
            'USELESS': 'Dz9mQ9NzkBcCsuGPFJ3r1bS4wgqKMHBPiVuniW8Mbonk'
        }
        
        logger.info("📊 DexScreener provider initialized")
    
    async def get_token_price(self, symbol: str) -> Optional[float]:
        """Get real-time price for a token"""
        try:
            symbol_upper = symbol.replace('USDT', '').replace('USDC', '').upper()
            
            # Check cache first
            cache_key = f"{symbol_upper}_price"
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Get token address
            token_address = self.token_addresses.get(symbol_upper)
            if not token_address:
                logger.warning(f"❌ Token {symbol_upper} not found in DexScreener mapping")
                return None
            
            # Fetch from DexScreener
            price = await self._fetch_token_price(token_address)
            
            if price and price > 0:
                # Cache the result
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': time.time(),
                    'source': 'dexscreener'
                }
                logger.info(f"📊 DexScreener price for {symbol_upper}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting DexScreener price for {symbol}: {e}")
            return None
    
    async def _fetch_token_price(self, token_address: str) -> Optional[float]:
        """Fetch price from DexScreener API"""
        try:
            url = f"{self.base_url}/tokens/{token_address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"API Response for {token_address}: {data}")
                        
                        # Check if we have pairs data
                        if not data or 'pairs' not in data:
                            logger.warning(f"No pairs data for token {token_address}")
                            return None
                            
                        pairs = data['pairs']
                        if not pairs or len(pairs) == 0:
                            logger.warning(f"No liquidity pairs found for token {token_address}")
                            return None
                        
                        # Sort by liquidity (USD) to get most liquid pair
                        valid_pairs = []
                        for pair in pairs:
                            if pair and pair.get('priceUsd'):
                                valid_pairs.append(pair)
                        
                        if not valid_pairs:
                            logger.warning(f"No valid pairs with price data for {token_address}")
                            return None
                        
                        # Sort by liquidity
                        valid_pairs.sort(key=lambda x: x.get('liquidity', {}).get('usd', 0), reverse=True)
                        
                        best_pair = valid_pairs[0]
                        price_usd = best_pair.get('priceUsd')
                        
                        if price_usd:
                            return float(price_usd)
                    else:
                        logger.warning(f"API returned status {response.status} for {token_address}")
                
                return None
                
        except Exception as e:
            logger.error(f"DexScreener API error for {token_address}: {e}")
            return None
    
    async def get_token_info(self, symbol: str) -> Optional[Dict]:
        """Get detailed token information"""
        try:
            symbol_upper = symbol.replace('USDT', '').replace('USDC', '').upper()
            token_address = self.token_addresses.get(symbol_upper)
            
            if not token_address:
                return None
            
            url = f"{self.base_url}/tokens/{token_address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'pairs' in data and len(data['pairs']) > 0:
                            # Get most liquid pair
                            pairs = sorted(data['pairs'], 
                                         key=lambda x: x.get('liquidity', {}).get('usd', 0), 
                                         reverse=True)
                            
                            best_pair = pairs[0]
                            
                            return {
                                'symbol': symbol_upper,
                                'address': token_address,
                                'price_usd': float(best_pair.get('priceUsd', 0)),
                                'volume_24h': best_pair.get('volume', {}).get('h24', 0),
                                'liquidity_usd': best_pair.get('liquidity', {}).get('usd', 0),
                                'price_change_24h': best_pair.get('priceChange', {}).get('h24', 0),
                                'dex': best_pair.get('dexId', 'unknown'),
                                'pair_address': best_pair.get('pairAddress', ''),
                                'base_token': best_pair.get('baseToken', {}),
                                'quote_token': best_pair.get('quoteToken', {})
                            }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting token info for {symbol}: {e}")
            return None
    
    async def get_multiple_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get prices for multiple tokens efficiently"""
        tasks = []
        for symbol in symbols:
            tasks.append(self.get_token_price(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_dict = {}
        for i, symbol in enumerate(symbols):
            if not isinstance(results[i], Exception) and results[i]:
                price_dict[symbol] = results[i]
        
        return price_dict
    
    def is_supported(self, symbol: str) -> bool:
        """Check if token is supported"""
        symbol_clean = symbol.replace('USDT', '').replace('USDC', '').upper()
        return symbol_clean in self.token_addresses
    
    # Specific methods for aggressive/volatile tokens
    async def get_jellyjelly_price(self) -> Optional[float]:
        """Get JELLYJELLY token price - Ultra risky low-cap token"""
        return await self.get_token_price('JELLYJELLY')
    
    async def get_troll_price(self) -> Optional[float]:
        """Get TROLL token price - High volatility experimental token"""
        return await self.get_token_price('TROLL')
    
    async def get_useless_price(self) -> Optional[float]:
        """Get USELESS token price - Aggressive trading target"""
        return await self.get_token_price('USELESS')
    
    async def get_bonk_price(self) -> Optional[float]:
        """Get BONK token price - Proven volatile memecoin"""
        return await self.get_token_price('BONK')
    
    async def get_lowcap_prices(self) -> Dict[str, float]:
        """Get all aggressive trading token prices at once"""
        symbols = ['JELLYJELLY', 'TROLL', 'USELESS', 'BONK']
        return await self.get_multiple_prices(symbols)


# Global instance
_dexscreener_provider = None

async def get_dexscreener_provider() -> DexScreenerProvider:
    """Get or create the global DexScreener provider"""
    global _dexscreener_provider
    if _dexscreener_provider is None:
        _dexscreener_provider = DexScreenerProvider()
    return _dexscreener_provider

async def get_dex_price(symbol: str) -> Optional[float]:
    """Get price from DexScreener"""
    provider = await get_dexscreener_provider()
    return await provider.get_token_price(symbol)

# Aggressive/Volatile token endpoint functions
async def get_jellyjelly_price() -> Optional[float]:
    """GET JELLYJELLY_PRICE - Ultra risky low-cap token"""
    provider = await get_dexscreener_provider()
    return await provider.get_jellyjelly_price()

async def get_troll_price() -> Optional[float]:
    """GET TROLL_PRICE - High volatility experimental token"""
    provider = await get_dexscreener_provider()
    return await provider.get_troll_price()

async def get_useless_price() -> Optional[float]:
    """GET USELESS_PRICE - Aggressive trading target"""
    provider = await get_dexscreener_provider()
    return await provider.get_useless_price()

async def get_bonk_price() -> Optional[float]:
    """GET BONK_PRICE - Proven volatile memecoin"""
    provider = await get_dexscreener_provider()
    return await provider.get_bonk_price()

async def get_all_lowcap_prices() -> Dict[str, float]:
    """Get all aggressive trading token prices at once"""
    provider = await get_dexscreener_provider()
    return await provider.get_lowcap_prices()

async def get_dex_token_info(symbol: str) -> Optional[Dict]:
    """Get token info from DexScreener"""
    provider = await get_dexscreener_provider()
    return await provider.get_token_info(symbol)