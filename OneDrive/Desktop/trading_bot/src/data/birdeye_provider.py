"""
Birdeye API Provider for Token Discovery
Advanced token discovery using Birdeye's comprehensive API endpoints
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BirdeyeToken:
    """Birdeye token data structure"""
    address: str
    symbol: str
    name: str
    decimals: int
    price_usd: float
    volume_24h: float
    market_cap: float
    liquidity_usd: float
    price_change_24h: float
    security_score: float
    creation_time: datetime
    holders: int

class BirdeyeProvider:
    """
    Birdeye API provider for comprehensive token discovery and analysis
    """
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.base_url = "https://public-api.birdeye.so"
        
        # Rate limiting (Birdeye is generally more generous than other APIs)
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.last_request = 0
        
        # Cache for API responses
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Authentication headers
        self.headers = {
            'accept': 'application/json',
            'x-chain': 'solana'  # Default to Solana chain
        }
        
        if self.api_key:
            self.headers['X-API-KEY'] = self.api_key
        else:
            logger.warning("No Birdeye API key provided. Some endpoints may be limited or unavailable.")
        
        # Supported networks
        self.networks = {
            'ethereum': 'ethereum',
            'solana': 'solana',
            'bsc': 'bsc',
            'polygon': 'polygon',
            'arbitrum': 'arbitrum',
            'avalanche': 'avalanche',
            'base': 'base'
        }
        
        logger.info("🦅 Birdeye API provider initialized")
    
    async def _make_request(self, endpoint: str, params: Dict = None, method: str = 'GET') -> Optional[Dict]:
        """Make rate-limited request to Birdeye API"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        try:
            # Use the configured headers with API key if available
            headers = self.headers.copy()
            
            url = f"{self.base_url}{endpoint}"
            
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers, params=params) as response:
                        self.last_request = time.time()
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429:
                            logger.warning("⚠️ Birdeye rate limit hit - waiting")
                            await asyncio.sleep(2.0)
                            return None
                        elif response.status == 401:
                            logger.error("🔑 Birdeye API authentication failed - check API key")
                            return None
                        else:
                            logger.warning(f"Birdeye API error {response.status}: {endpoint}")
                            return None
                else:
                    async with session.post(url, headers=headers, json=params) as response:
                        self.last_request = time.time()
                        
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 401:
                            logger.error("🔑 Birdeye API authentication failed - check API key")
                            return None
                        else:
                            logger.warning(f"Birdeye API error {response.status}: {endpoint}")
                            return None
                        
        except Exception as e:
            logger.error(f"Birdeye API request failed: {e}")
            return None
    
    async def get_trending_tokens(self, network: str = 'ethereum', limit: int = 50) -> List[Dict]:
        """Get trending tokens from Birdeye"""
        try:
            endpoint = f"/defi/tokenlist"
            params = {
                'sort_by': 'volume24hUSD',
                'sort_type': 'desc',
                'limit': limit,
                'offset': 0
            }
            
            if network in self.networks:
                # For multi-chain support, we might need different endpoints
                # For now, focus on general trending
                pass
            
            result = await self._make_request(endpoint, params)
            
            if result and result.get('success') and result.get('data'):
                tokens = []
                for token_data in result['data']['tokens'][:limit]:
                    try:
                        tokens.append({
                            'address': token_data.get('address', ''),
                            'symbol': token_data.get('symbol', '').upper(),
                            'name': token_data.get('name', ''),
                            'decimals': token_data.get('decimals', 18),
                            'price_usd': float(token_data.get('price', 0)),
                            'volume_24h': float(token_data.get('volume24hUSD', 0)),
                            'market_cap': float(token_data.get('mc', 0)),
                            'liquidity_usd': float(token_data.get('liquidity', 0)),
                            'price_change_24h': float(token_data.get('priceChange24h', 0)),
                            'discovery_source': 'birdeye_trending'
                        })
                    except Exception as e:
                        logger.warning(f"Error processing Birdeye token: {e}")
                        continue
                
                return tokens
                
        except Exception as e:
            logger.error(f"❌ Birdeye trending tokens failed: {e}")
            
        return []
    
    async def get_new_listings(self, network: str = 'ethereum', limit: int = 50) -> List[Dict]:
        """Get newly listed tokens from Birdeye"""
        try:
            endpoint = f"/defi/tokenlist"
            params = {
                'sort_by': 'createdTime',
                'sort_type': 'desc',
                'limit': limit,
                'offset': 0
            }
            
            result = await self._make_request(endpoint, params)
            
            if result and result.get('success') and result.get('data'):
                tokens = []
                for token_data in result['data']['tokens'][:limit]:
                    try:
                        # Filter for recently created tokens with some activity
                        volume_24h = float(token_data.get('volume24hUSD', 0))
                        if volume_24h > 1000:  # Minimum activity threshold
                            tokens.append({
                                'address': token_data.get('address', ''),
                                'symbol': token_data.get('symbol', '').upper(),
                                'name': token_data.get('name', ''),
                                'decimals': token_data.get('decimals', 18),
                                'price_usd': float(token_data.get('price', 0)),
                                'volume_24h': volume_24h,
                                'market_cap': float(token_data.get('mc', 0)),
                                'liquidity_usd': float(token_data.get('liquidity', 0)),
                                'price_change_24h': float(token_data.get('priceChange24h', 0)),
                                'discovery_source': 'birdeye_new_listings'
                            })
                    except Exception as e:
                        logger.warning(f"Error processing Birdeye new listing: {e}")
                        continue
                
                return tokens
                
        except Exception as e:
            logger.error(f"❌ Birdeye new listings failed: {e}")
            
        return []
    
    async def get_token_security(self, token_address: str) -> Optional[Dict]:
        """Get token security information from Birdeye"""
        try:
            endpoint = f"/defi/token_security"
            params = {'address': token_address}
            
            result = await self._make_request(endpoint, params)
            
            if result and result.get('success'):
                return result.get('data', {})
                
        except Exception as e:
            logger.error(f"❌ Birdeye token security failed: {e}")
            
        return None
    
    async def get_multiple_token_data(self, token_addresses: List[str]) -> Dict[str, Dict]:
        """Get market data for multiple tokens"""
        try:
            endpoint = f"/defi/multi_price"
            params = {
                'list_address': ','.join(token_addresses)
            }
            
            result = await self._make_request(endpoint, params, method='POST')
            
            if result and result.get('success') and result.get('data'):
                token_data = {}
                for address, data in result['data'].items():
                    token_data[address.lower()] = {
                        'price_usd': float(data.get('value', 0)),
                        'price_change_24h': float(data.get('priceChange24h', 0)),
                        'volume_24h': float(data.get('volume24hUSD', 0)),
                        'market_cap': float(data.get('mc', 0)),
                        'liquidity': float(data.get('liquidity', 0)),
                        'discovery_source': 'birdeye_multi'
                    }
                
                return token_data
                
        except Exception as e:
            logger.error(f"❌ Birdeye multiple token data failed: {e}")
            
        return {}
    
    async def search_tokens(self, query: str, limit: int = 20) -> List[Dict]:
        """Search for tokens by name or symbol"""
        try:
            endpoint = f"/defi/search"
            params = {
                'keyword': query,
                'limit': limit
            }
            
            result = await self._make_request(endpoint, params)
            
            if result and result.get('success') and result.get('data'):
                tokens = []
                for token_data in result['data']['tokens'][:limit]:
                    try:
                        tokens.append({
                            'address': token_data.get('address', ''),
                            'symbol': token_data.get('symbol', '').upper(),
                            'name': token_data.get('name', ''),
                            'decimals': token_data.get('decimals', 18),
                            'price_usd': float(token_data.get('price', 0)),
                            'volume_24h': float(token_data.get('volume24hUSD', 0)),
                            'market_cap': float(token_data.get('mc', 0)),
                            'discovery_source': 'birdeye_search'
                        })
                    except Exception as e:
                        logger.warning(f"Error processing Birdeye search result: {e}")
                        continue
                
                return tokens
                
        except Exception as e:
            logger.error(f"❌ Birdeye token search failed: {e}")
            
        return []
    
    async def get_filtered_tokens(self, min_volume: float = 50000, min_market_cap: float = 500000, 
                                 max_market_cap: float = 2000000, limit: int = 50) -> List[Dict]:
        """Get filtered tokens based on criteria using multiple Birdeye endpoints"""
        try:
            logger.info(f"🦅 Searching Birdeye for tokens: vol≥${min_volume:,.0f}, mc=${min_market_cap:,.0f}-${max_market_cap:,.0f}")
            
            all_tokens = []
            
            # Get trending tokens
            trending = await self.get_trending_tokens(limit=limit//2)
            all_tokens.extend(trending)
            
            # Get new listings
            new_listings = await self.get_new_listings(limit=limit//2)
            all_tokens.extend(new_listings)
            
            # Filter based on criteria
            filtered_tokens = []
            for token in all_tokens:
                try:
                    volume = token.get('volume_24h', 0)
                    market_cap = token.get('market_cap', 0)
                    
                    if (volume >= min_volume and 
                        min_market_cap <= market_cap <= max_market_cap):
                        filtered_tokens.append(token)
                        
                except Exception as e:
                    logger.warning(f"Error filtering Birdeye token: {e}")
                    continue
            
            # Remove duplicates by address
            seen_addresses = set()
            unique_tokens = []
            for token in filtered_tokens:
                address = token.get('address', '').lower()
                if address and address not in seen_addresses:
                    seen_addresses.add(address)
                    unique_tokens.append(token)
            
            logger.info(f"🦅 Birdeye found {len(unique_tokens)} tokens matching criteria")
            return unique_tokens[:limit]
            
        except Exception as e:
            logger.error(f"❌ Birdeye filtered tokens failed: {e}")
            return []


# Example usage and testing
async def test_birdeye_provider():
    """Test the Birdeye provider"""
    provider = BirdeyeProvider()
    
    print("🧪 Testing Birdeye API Provider...")
    
    # Test trending tokens
    print("\n1️⃣ Testing trending tokens...")
    trending = await provider.get_trending_tokens(limit=10)
    print(f"   Found {len(trending)} trending tokens")
    for i, token in enumerate(trending[:3]):
        print(f"   {i+1}. {token['symbol']}: ${token['price_usd']:.6f} | Vol: ${token['volume_24h']:,.0f}")
    
    # Test new listings
    print("\n2️⃣ Testing new listings...")
    new_listings = await provider.get_new_listings(limit=10)
    print(f"   Found {len(new_listings)} new listings")
    for i, token in enumerate(new_listings[:3]):
        print(f"   {i+1}. {token['symbol']}: ${token['price_usd']:.6f} | Vol: ${token['volume_24h']:,.0f}")
    
    # Test filtered search
    print("\n3️⃣ Testing filtered search...")
    filtered = await provider.get_filtered_tokens(
        min_volume=10000,
        min_market_cap=500000,
        max_market_cap=1500000,
        limit=10
    )
    print(f"   Found {len(filtered)} filtered tokens")
    for i, token in enumerate(filtered[:3]):
        print(f"   {i+1}. {token['symbol']}: ${token['price_usd']:.6f} | MC: ${token['market_cap']:,.0f}")
    
    print("\n✅ Birdeye provider test completed!")


if __name__ == "__main__":
    asyncio.run(test_birdeye_provider())