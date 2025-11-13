"""
Official Dextools API Provider for Aggressive Token Trading
Using the official dextools-python library for reliable data
"""

import asyncio
import logging
from typing import Dict, Optional, List
import time

try:
    from dextools_python import DextoolsAPIV2, AsyncDextoolsAPIV2
    DEXTOOLS_AVAILABLE = True
except ImportError:
    DEXTOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)

class OfficialDextoolsProvider:
    """Official Dextools API client using dextools-python library"""
    
    def __init__(self, api_key: Optional[str] = None, plan: str = "free"):
        if not DEXTOOLS_AVAILABLE:
            raise ImportError("dextools-python library not installed. Run: pip install dextools-python")
        
        self.api_key = api_key
        self.plan = plan
        self.price_cache = {}
        self.cache_duration = 30  # Cache for 30 seconds
        
        # Initialize sync client for quick operations
        if api_key:
            self.dextools = DextoolsAPIV2(api_key, plan=plan)
        else:
            # Try free tier without key
            try:
                self.dextools = DextoolsAPIV2("", plan="free")
            except:
                self.dextools = None
        
        logger.info(f"🛠️ Official Dextools provider initialized (plan: {plan})")
    
    async def get_token_price(self, token_address: str, chain: str = "solana") -> Optional[float]:
        """Get real-time price for a token using official Dextools API"""
        try:
            # Check cache first
            cache_key = f"{chain}_{token_address}_price"
            if cache_key in self.price_cache:
                cached_data = self.price_cache[cache_key]
                if time.time() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['price']
            
            # Use async client for better performance
            price = await self._fetch_token_price_async(token_address, chain)
            
            if price and price > 0:
                # Cache the result
                self.price_cache[cache_key] = {
                    'price': price,
                    'timestamp': time.time(),
                    'source': 'dextools_official'
                }
                logger.info(f"🛠️ Dextools price for {token_address[:8]}... on {chain}: ${price}")
                return price
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Dextools price for {token_address}: {e}")
            return None
    
    async def _fetch_token_price_async(self, token_address: str, chain: str) -> Optional[float]:
        """Fetch price using async client"""
        try:
            if not self.api_key:
                # Try without API key first (free tier)
                return await self._try_free_tier_price(token_address, chain)
            
            async with AsyncDextoolsAPIV2(self.api_key, plan=self.plan) as dextools:
                response = await dextools.get_token_price(chain, token_address)
                
                if response and 'data' in response:
                    price_data = response['data']
                    price = price_data.get('price')
                    
                    if price:
                        return float(price)
                
                return None
                
        except Exception as e:
            logger.error(f"Async Dextools API error: {e}")
            return None
    
    async def _try_free_tier_price(self, token_address: str, chain: str) -> Optional[float]:
        """Try to get price without API key (free tier limitations)"""
        try:
            # For free tier, we might need to use sync API in a thread
            import asyncio
            
            def get_price_sync():
                try:
                    if self.dextools:
                        response = self.dextools.get_token_price(chain, token_address)
                        if response and 'data' in response:
                            price_data = response['data']
                            return price_data.get('price')
                except Exception as e:
                    logger.debug(f"Free tier price fetch failed: {e}")
                return None
            
            # Run sync operation in thread pool
            loop = asyncio.get_event_loop()
            price = await loop.run_in_executor(None, get_price_sync)
            
            if price:
                return float(price)
            
            return None
            
        except Exception as e:
            logger.error(f"Free tier price fetch error: {e}")
            return None
    
    async def get_token_info(self, token_address: str, chain: str = "solana") -> Optional[Dict]:
        """Get detailed token information"""
        try:
            if not self.api_key:
                return await self._try_free_tier_info(token_address, chain)
            
            async with AsyncDextoolsAPIV2(self.api_key, plan=self.plan) as dextools:
                response = await dextools.get_token(chain, token_address)
                
                if response and 'data' in response:
                    return response['data']
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return None
    
    async def _try_free_tier_info(self, token_address: str, chain: str) -> Optional[Dict]:
        """Try to get token info without API key"""
        try:
            import asyncio
            
            def get_info_sync():
                try:
                    if self.dextools:
                        response = self.dextools.get_token(chain, token_address)
                        if response and 'data' in response:
                            return response['data']
                except Exception as e:
                    logger.debug(f"Free tier info fetch failed: {e}")
                return None
            
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, get_info_sync)
            return info
            
        except Exception as e:
            logger.error(f"Free tier info fetch error: {e}")
            return None
    
    async def search_tokens(self, query: str, chain: str = "solana") -> List[Dict]:
        """Search for tokens (requires API key for most functionality)"""
        try:
            if not self.api_key:
                logger.warning("Token search requires API key")
                return []
            
            async with AsyncDextoolsAPIV2(self.api_key, plan=self.plan) as dextools:
                # Note: Search might not be available in all plans
                # This is a placeholder - actual search API might be different
                response = await dextools.get_tokens(chain)
                
                if response and 'data' in response:
                    # Filter results by query
                    tokens = response['data']
                    filtered = []
                    
                    for token in tokens:
                        if query.lower() in token.get('name', '').lower() or \
                           query.lower() in token.get('symbol', '').lower():
                            filtered.append(token)
                    
                    return filtered[:10]  # Limit to 10 results
                
                return []
                
        except Exception as e:
            logger.error(f"Error searching tokens: {e}")
            return []
    
    async def get_trending_tokens(self, chain: str = "solana") -> List[Dict]:
        """Get trending tokens (requires API key)"""
        try:
            if not self.api_key:
                logger.warning("Trending tokens requires API key")
                return []
            
            async with AsyncDextoolsAPIV2(self.api_key, plan=self.plan) as dextools:
                # Try to get hot pools or gainers as trending indicators
                response = await dextools.get_ranking_hotpools(chain)
                
                if response and 'data' in response:
                    return response['data'][:20]  # Top 20
                
                return []
                
        except Exception as e:
            logger.error(f"Error getting trending tokens: {e}")
            return []


# Global instance
_official_dextools_provider = None

async def get_official_dextools_provider(api_key: Optional[str] = None) -> OfficialDextoolsProvider:
    """Get or create the global official Dextools provider"""
    global _official_dextools_provider
    if _official_dextools_provider is None:
        _official_dextools_provider = OfficialDextoolsProvider(api_key)
    return _official_dextools_provider

async def get_dextools_token_price(token_address: str, chain: str = "solana") -> Optional[float]:
    """Get token price using official Dextools API"""
    provider = await get_official_dextools_provider()
    return await provider.get_token_price(token_address, chain)

async def get_dextools_token_info(token_address: str, chain: str = "solana") -> Optional[Dict]:
    """Get token info using official Dextools API"""
    provider = await get_official_dextools_provider()
    return await provider.get_token_info(token_address, chain)