"""
Dextools API Provider for Low-Cap Token Trading
Real-time Solana token data with better coverage than DexScreener
"""

import asyncio
import aiohttp
import logging
from typing import Dict, Optional, List
import time
import json

logger = logging.getLogger(__name__)

class DextoolsProvider:
    """Dextools API client for comprehensive Solana token data"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.base_url = "https://open-api.dextools.io/free/v2"
        self.api_key = api_key  # Optional for free tier
        self.price_cache = {}
        self.cache_duration = 30  # Cache for 30 seconds
        
        # Headers for API requests
        self.headers = {
            'accept': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        }
        
        if self.api_key:
            self.headers['X-BLOBR-KEY'] = self.api_key
        
        logger.info("🛠️ Dextools provider initialized")
    
    async def get_token_price(self, token_address: str) -> Optional[float]:
        """Get real-time price for a Solana token"""
        try:
            # Check cache first
            cache_key = f"{token_address}_price"
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Fetch from Dextools
            price = await self._fetch_token_price(token_address)
            
            if price and price > 0:
                # Cache the result
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': time.time(),
                    'source': 'dextools'
                }
                logger.info(f"🛠️ Dextools price for {token_address[:8]}...: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Dextools price for {token_address}: {e}")
            return None
    
    async def _fetch_token_price(self, token_address: str) -> Optional[float]:
        """Fetch price from Dextools API"""
        try:
            # Dextools uses solana chain identifier
            url = f"{self.base_url}/token/solana/{token_address}/price"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Dextools API response: {data}")
                        
                        # Dextools response format
                        if data and 'data' in data:
                            price_data = data['data']
                            price_usd = price_data.get('price')
                            
                            if price_usd:
                                return float(price_usd)
                    
                    elif response.status == 429:
                        logger.warning("Dextools rate limit hit")
                        return None
                    else:
                        text = await response.text()
                        logger.warning(f"Dextools API returned {response.status}: {text[:200]}")
                
                return None
                
        except Exception as e:
            logger.error(f"Dextools API error for {token_address}: {e}")
            return None
    
    async def get_token_info(self, token_address: str) -> Optional[Dict]:
        """Get detailed token information including liquidity and volume"""
        try:
            url = f"{self.base_url}/token/solana/{token_address}/info"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'data' in data:
                            return data['data']
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting token info from Dextools: {e}")
            return None
    
    async def search_tokens(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for tokens by name or symbol"""
        try:
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'page': 1,
                'pageSize': limit
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'data' in data:
                            return data['data']
                    
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching tokens on Dextools: {e}")
            return []
    
    async def get_trending_tokens(self, chain: str = 'solana', limit: int = 20) -> List[Dict]:
        """Get trending tokens for aggressive trading opportunities"""
        try:
            url = f"{self.base_url}/rankings/solana/trending"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'data' in data:
                            return data['data'][:limit]
                    
                    return []
                    
        except Exception as e:
            logger.error(f"Error getting trending tokens: {e}")
            return []
    
    async def get_pool_info(self, pool_address: str) -> Optional[Dict]:
        """Get liquidity pool information"""
        try:
            url = f"{self.base_url}/pool/solana/{pool_address}"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data and 'data' in data:
                            return data['data']
                    
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting pool info: {e}")
            return None


# Global instance
_dextools_provider = None

async def get_dextools_provider(api_key: Optional[str] = None) -> DextoolsProvider:
    """Get or create the global Dextools provider"""
    global _dextools_provider
    if _dextools_provider is None:
        _dextools_provider = DextoolsProvider(api_key)
    return _dextools_provider

async def get_dextools_price(token_address: str) -> Optional[float]:
    """Get price from Dextools"""
    provider = await get_dextools_provider()
    return await provider.get_token_price(token_address)

async def search_dextools_tokens(query: str) -> List[Dict]:
    """Search for tokens on Dextools"""
    provider = await get_dextools_provider()
    return await provider.search_tokens(query)

async def get_dextools_trending() -> List[Dict]:
    """Get trending tokens from Dextools"""
    provider = await get_dextools_provider()
    return await provider.get_trending_tokens()