"""
CoinMarketCap Data Provider
Real-time market data using CoinMarketCap API
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class CoinMarketCapProvider:
    """
    CoinMarketCap API integration for real-time market data
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pro-api.coinmarketcap.com/v1"
        self.headers = {
            'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': api_key,
        }
        
        # Cache for API rate limiting (free tier: 333 calls/day, 10 calls/minute)
        self.cache = {}
        self.cache_duration = 60  # Cache for 1 minute
        self.last_request_time = 0
        self.min_request_interval = 6  # 6 seconds between requests for free tier
        
        logger.info("CoinMarketCap provider initialized")
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Check if we should use cached data"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: sleeping {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    async def get_latest_quotes(self, symbols: List[str]) -> Dict:
        """Get latest price quotes for symbols"""
        cache_key = f"quotes_{','.join(sorted(symbols))}"
        
        if self._should_use_cache(cache_key):
            logger.info(f"Using cached data for {symbols}")
            return self.cache[cache_key]['data']
        
        # Rate limiting
        self._rate_limit()
        
        # Convert symbols to CMC format (BTC-USD -> BTC)
        cmc_symbols = [symbol.split('-')[0] for symbol in symbols]
        symbol_string = ','.join(cmc_symbols)
        
        url = f"{self.base_url}/cryptocurrency/quotes/latest"
        params = {
            'symbol': symbol_string,
            'convert': 'USD'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process and cache the data
                        processed_data = self._process_quotes_data(data, symbols)
                        self.cache[cache_key] = {
                            'data': processed_data,
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"Retrieved quotes for {len(symbols)} symbols")
                        return processed_data
                    else:
                        error_text = await response.text()
                        logger.error(f"CMC API error: {response.status} - {error_text}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching quotes: {e}")
            return {}
    
    def _process_quotes_data(self, data: Dict, original_symbols: List[str]) -> Dict:
        """Process CMC API response into our format"""
        processed = {}
        
        if 'data' not in data:
            return processed
        
        for symbol in original_symbols:
            base_symbol = symbol.split('-')[0]  # BTC-USD -> BTC
            
            if base_symbol in data['data']:
                quote_data = data['data'][base_symbol]
                usd_quote = quote_data['quote']['USD']
                
                processed[symbol] = {
                    'symbol': symbol,
                    'price': usd_quote['price'],
                    'price_change_24h': usd_quote['percent_change_24h'],
                    'price_change_1h': usd_quote['percent_change_1h'],
                    'volume_24h': usd_quote['volume_24h'],
                    'market_cap': usd_quote['market_cap'],
                    'last_updated': quote_data['last_updated'],
                    'circulating_supply': quote_data['circulating_supply'],
                    'total_supply': quote_data['total_supply'],
                    'max_supply': quote_data.get('max_supply'),
                    'cmc_rank': quote_data['cmc_rank']
                }
        
        return processed
    
    async def get_market_metrics(self, symbol: str) -> Dict:
        """Get comprehensive market metrics for a symbol"""
        cache_key = f"metrics_{symbol}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        # Get latest quote first
        quotes = await self.get_latest_quotes([symbol])
        if symbol not in quotes:
            return {}
        
        quote = quotes[symbol]
        
        # Calculate additional metrics
        metrics = {
            'price': quote['price'],
            'volume_24h': quote['volume_24h'],
            'market_cap': quote['market_cap'],
            'volatility_24h': abs(quote['price_change_24h']) / 100,
            'volatility_1h': abs(quote['price_change_1h']) / 100,
            'liquidity_score': self._calculate_liquidity_score(quote),
            'momentum_score': self._calculate_momentum_score(quote),
            'market_strength': self._calculate_market_strength(quote),
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache the metrics
        self.cache[cache_key] = {
            'data': metrics,
            'timestamp': time.time()
        }
        
        return metrics
    
    def _calculate_liquidity_score(self, quote: Dict) -> float:
        """Calculate liquidity score based on volume and market cap"""
        try:
            volume = quote['volume_24h']
            market_cap = quote['market_cap']
            
            if market_cap <= 0:
                return 0.0
            
            # Volume to market cap ratio
            volume_ratio = volume / market_cap
            
            # Normalize to 0-1 scale (0.1 = high liquidity)
            liquidity_score = min(volume_ratio / 0.1, 1.0)
            
            return liquidity_score
            
        except (KeyError, ZeroDivisionError, TypeError):
            return 0.5  # Default moderate liquidity
    
    def _calculate_momentum_score(self, quote: Dict) -> float:
        """Calculate momentum score based on price changes"""
        try:
            change_1h = quote['price_change_1h']
            change_24h = quote['price_change_24h']
            
            # Weight recent changes more heavily
            momentum = (change_1h * 0.7) + (change_24h * 0.3)
            
            # Normalize to -1 to 1 scale
            momentum_score = max(-1.0, min(1.0, momentum / 10))
            
            return momentum_score
            
        except (KeyError, TypeError):
            return 0.0  # Neutral momentum
    
    def _calculate_market_strength(self, quote: Dict) -> float:
        """Calculate overall market strength"""
        try:
            # Factors: rank, volume, market cap
            rank = quote['cmc_rank']
            volume = quote['volume_24h']
            market_cap = quote['market_cap']
            
            # Rank score (lower rank = higher score)
            rank_score = max(0, (100 - rank) / 100) if rank <= 100 else 0
            
            # Volume score (logarithmic scale)
            volume_score = min(1.0, (volume / 1e9)) if volume > 0 else 0
            
            # Market cap score (logarithmic scale)
            market_cap_score = min(1.0, (market_cap / 1e11)) if market_cap > 0 else 0
            
            # Weighted average
            strength = (rank_score * 0.4) + (volume_score * 0.3) + (market_cap_score * 0.3)
            
            return strength
            
        except (KeyError, TypeError):
            return 0.5  # Default moderate strength
    
    async def get_top_movers(self, limit: int = 10) -> List[Dict]:
        """Get top price movers in the last 24 hours"""
        cache_key = f"top_movers_{limit}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        url = f"{self.base_url}/cryptocurrency/listings/latest"
        params = {
            'start': '1',
            'limit': str(limit * 2),  # Get more to filter
            'convert': 'USD',
            'sort': 'percent_change_24h',
            'sort_dir': 'desc'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        movers = []
                        for crypto in data.get('data', [])[:limit]:
                            if crypto['quote']['USD']['volume_24h'] > 1000000:  # Min $1M volume
                                movers.append({
                                    'symbol': crypto['symbol'],
                                    'name': crypto['name'],
                                    'price': crypto['quote']['USD']['price'],
                                    'change_24h': crypto['quote']['USD']['percent_change_24h'],
                                    'volume_24h': crypto['quote']['USD']['volume_24h'],
                                    'market_cap': crypto['quote']['USD']['market_cap'],
                                    'rank': crypto['cmc_rank']
                                })
                        
                        # Cache the results
                        self.cache[cache_key] = {
                            'data': movers,
                            'timestamp': time.time()
                        }
                        
                        return movers
                    else:
                        logger.error(f"Failed to get top movers: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching top movers: {e}")
            return []
    
    async def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        cache_key = "market_overview"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        url = f"{self.base_url}/global-metrics/quotes/latest"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        global_data = data['data']['quote']['USD']
                        
                        overview = {
                            'total_market_cap': global_data['total_market_cap'],
                            'total_volume_24h': global_data['total_volume_24h'],
                            'bitcoin_dominance': data['data']['btc_dominance'],
                            'ethereum_dominance': data['data']['eth_dominance'],
                            'active_cryptocurrencies': data['data']['active_cryptocurrencies'],
                            'market_cap_change_24h': global_data['total_market_cap_yesterday_percentage_change'],
                            'volume_change_24h': global_data['total_volume_24h_yesterday_percentage_change'],
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Cache the overview
                        self.cache[cache_key] = {
                            'data': overview,
                            'timestamp': time.time()
                        }
                        
                        return overview
                    else:
                        logger.error(f"Failed to get market overview: {response.status}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {}
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.cache)
        current_time = time.time()
        
        fresh_entries = sum(1 for entry in self.cache.values() 
                          if (current_time - entry['timestamp']) < self.cache_duration)
        
        return {
            'total_entries': total_entries,
            'fresh_entries': fresh_entries,
            'cache_hit_ratio': fresh_entries / total_entries if total_entries > 0 else 0,
            'last_request_time': self.last_request_time
        }

# Example usage and testing
async def test_coinmarketcap_provider():
    """Test the CoinMarketCap provider"""
    
    # Load API key from config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['coinmarketcap']
    except (FileNotFoundError, KeyError):
        print("❌ CoinMarketCap API key not found in config.json")
        return
    
    provider = CoinMarketCapProvider(api_key)
    
    print("🪙 Testing CoinMarketCap Provider...")
    print("=" * 50)
    
    # Test 1: Get quotes for major cryptocurrencies
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    print(f"\n📊 Getting quotes for: {', '.join(symbols)}")
    
    quotes = await provider.get_latest_quotes(symbols)
    
    for symbol, data in quotes.items():
        print(f"  {symbol}: ${data['price']:,.2f} "
              f"({data['price_change_24h']:+.2f}% 24h)")
    
    # Test 2: Get detailed metrics for BTC
    print(f"\n🔍 Getting detailed metrics for BTC-USD...")
    metrics = await provider.get_market_metrics('BTC-USD')
    
    if metrics:
        print(f"  Price: ${metrics['price']:,.2f}")
        print(f"  Volume 24h: ${metrics['volume_24h']:,.0f}")
        print(f"  Market Cap: ${metrics['market_cap']:,.0f}")
        print(f"  Volatility 24h: {metrics['volatility_24h']:.2%}")
        print(f"  Liquidity Score: {metrics['liquidity_score']:.2f}")
        print(f"  Momentum Score: {metrics['momentum_score']:+.2f}")
        print(f"  Market Strength: {metrics['market_strength']:.2f}")
    
    # Test 3: Get top movers
    print(f"\n🚀 Getting top 5 movers...")
    movers = await provider.get_top_movers(5)
    
    for i, mover in enumerate(movers, 1):
        print(f"  {i}. {mover['symbol']} ({mover['name']}): "
              f"{mover['change_24h']:+.2f}% - ${mover['price']:,.4f}")
    
    # Test 4: Market overview
    print(f"\n🌍 Market Overview...")
    overview = await provider.get_market_overview()
    
    if overview:
        print(f"  Total Market Cap: ${overview['total_market_cap']:,.0f}")
        print(f"  24h Volume: ${overview['total_volume_24h']:,.0f}")
        print(f"  Bitcoin Dominance: {overview['bitcoin_dominance']:.1f}%")
        print(f"  Market Cap Change 24h: {overview['market_cap_change_24h']:+.2f}%")
    
    # Test 5: Cache statistics
    print(f"\n📈 Cache Statistics...")
    cache_stats = provider.get_cache_stats()
    print(f"  Total Entries: {cache_stats['total_entries']}")
    print(f"  Fresh Entries: {cache_stats['fresh_entries']}")
    print(f"  Cache Hit Ratio: {cache_stats['cache_hit_ratio']:.1%}")
    
    print(f"\n✅ CoinMarketCap provider test completed!")

if __name__ == "__main__":
    asyncio.run(test_coinmarketcap_provider())