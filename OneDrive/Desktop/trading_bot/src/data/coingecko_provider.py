"""
CoinGecko Data Provider
Advanced market data and analytics using CoinGecko API
"""

import requests
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import numpy as np

logger = logging.getLogger(__name__)

class CoinGeckoProvider:
    """
    CoinGecko API integration for comprehensive market data and analytics
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.coingecko.com/api/v3"
        self.pro_url = "https://api.coingecko.com/api/v3"  # Use standard API for demo key
        self.headers = {
            'accept': 'application/json',
            'x-cg-demo-api-key': api_key,  # Use demo API key header
        }
        
        # Cache for API rate limiting (Demo: 30 calls/min)
        self.cache = {}
        self.cache_duration = 120  # Cache for 2 minutes for demo tier
        self.last_request_time = 0
        self.min_request_interval = 2.0  # 2 seconds between requests for demo tier
        
        # Coin ID mapping (CoinGecko uses different IDs)
        self.coin_mapping = {
            'BTC-USD': 'bitcoin',
            'ETH-USD': 'ethereum', 
            'SOL-USD': 'solana',
            'ADA-USD': 'cardano',
            'DOT-USD': 'polkadot',
            'MATIC-USD': 'polygon',
            'AVAX-USD': 'avalanche-2',
            'LINK-USD': 'chainlink',
            'UNI-USD': 'uniswap',
            'ATOM-USD': 'cosmos'
        }
        
        logger.info("CoinGecko provider initialized with Demo API key")
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Check if we should use cached data"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (time.time() - cache_time) < self.cache_duration
    
    def _rate_limit(self):
        """Enforce rate limiting for Pro tier"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_coin_id(self, symbol: str) -> str:
        """Convert symbol to CoinGecko coin ID"""
        return self.coin_mapping.get(symbol, symbol.lower().replace('-usd', ''))
    
    async def get_coin_data(self, symbols: List[str]) -> Dict:
        """Get comprehensive coin data"""
        cache_key = f"coin_data_{','.join(sorted(symbols))}"
        
        if self._should_use_cache(cache_key):
            logger.info(f"Using cached coin data for {symbols}")
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        # Convert symbols to coin IDs
        coin_ids = [self._get_coin_id(symbol) for symbol in symbols]
        ids_string = ','.join(coin_ids)
        
        url = f"{self.base_url}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ids_string,
            'order': 'market_cap_desc',
            'per_page': len(coin_ids),
            'page': 1,
            'sparkline': 'true',
            'price_change_percentage': '1h,24h,7d,14d,30d,200d,1y'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process and cache the data
                        processed_data = self._process_coin_data(data, symbols)
                        self.cache[cache_key] = {
                            'data': processed_data,
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"Retrieved coin data for {len(symbols)} symbols")
                        return processed_data
                    else:
                        error_text = await response.text()
                        logger.error(f"CoinGecko API error: {response.status} - {error_text}")
                        return {}
                        
        except Exception as e:
            logger.error(f"Error fetching coin data: {e}")
            return {}
    
    def _process_coin_data(self, data: List[Dict], original_symbols: List[str]) -> Dict:
        """Process CoinGecko API response into our format"""
        processed = {}
        
        # Create reverse mapping
        id_to_symbol = {}
        for symbol in original_symbols:
            coin_id = self._get_coin_id(symbol)
            id_to_symbol[coin_id] = symbol
        
        for coin in data:
            coin_id = coin['id']
            if coin_id in id_to_symbol:
                symbol = id_to_symbol[coin_id]
                
                processed[symbol] = {
                    'symbol': symbol,
                    'name': coin['name'],
                    'price': coin['current_price'],
                    'market_cap': coin['market_cap'],
                    'market_cap_rank': coin['market_cap_rank'],
                    'volume_24h': coin['total_volume'],
                    'price_change_24h': coin['price_change_percentage_24h'] or 0,
                    'price_change_1h': coin.get('price_change_percentage_1h_in_currency', 0) or 0,
                    'price_change_7d': coin.get('price_change_percentage_7d_in_currency', 0) or 0,
                    'price_change_30d': coin.get('price_change_percentage_30d_in_currency', 0) or 0,
                    'price_change_1y': coin.get('price_change_percentage_1y_in_currency', 0) or 0,
                    'circulating_supply': coin['circulating_supply'],
                    'total_supply': coin['total_supply'],
                    'max_supply': coin['max_supply'],
                    'ath': coin['ath'],
                    'ath_change_percentage': coin['ath_change_percentage'],
                    'ath_date': coin['ath_date'],
                    'atl': coin['atl'],
                    'atl_change_percentage': coin['atl_change_percentage'],
                    'atl_date': coin['atl_date'],
                    'last_updated': coin['last_updated'],
                    'sparkline_7d': coin.get('sparkline_in_7d', {}).get('price', [])
                }
        
        return processed
    
    async def get_advanced_metrics(self, symbol: str) -> Dict:
        """Get advanced metrics and analytics for a coin"""
        cache_key = f"advanced_metrics_{symbol}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        coin_id = self._get_coin_id(symbol)
        
        # Get basic coin data first
        coin_data = await self.get_coin_data([symbol])
        if symbol not in coin_data:
            return {}
        
        base_data = coin_data[symbol]
        
        # Calculate advanced metrics
        metrics = {
            'basic_data': base_data,
            'volatility_analysis': self._calculate_volatility_metrics(base_data),
            'momentum_indicators': self._calculate_momentum_indicators(base_data),
            'market_position': self._calculate_market_position(base_data),
            'risk_metrics': self._calculate_risk_metrics(base_data),
            'opportunity_score': self._calculate_opportunity_score(base_data),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get additional data if needed
        try:
            # Get detailed coin info
            detailed_info = await self._get_coin_detailed_info(coin_id)
            if detailed_info:
                metrics['detailed_info'] = detailed_info
        except Exception as e:
            logger.warning(f"Could not fetch detailed info for {symbol}: {e}")
        
        # Cache the metrics
        self.cache[cache_key] = {
            'data': metrics,
            'timestamp': time.time()
        }
        
        return metrics
    
    async def _get_coin_detailed_info(self, coin_id: str) -> Dict:
        """Get detailed coin information"""
        self._rate_limit()
        
        url = f"{self.base_url}/coins/{coin_id}"
        params = {
            'localization': 'false',
            'tickers': 'false',
            'market_data': 'true',
            'community_data': 'true',
            'developer_data': 'true'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'market_data': data.get('market_data', {}),
                            'community_data': data.get('community_data', {}),
                            'developer_data': data.get('developer_data', {}),
                            'sentiment_votes_up_percentage': data.get('sentiment_votes_up_percentage'),
                            'sentiment_votes_down_percentage': data.get('sentiment_votes_down_percentage')
                        }
                    else:
                        return {}
        except Exception as e:
            logger.error(f"Error fetching detailed info: {e}")
            return {}
    
    def _calculate_volatility_metrics(self, data: Dict) -> Dict:
        """Calculate volatility-based metrics"""
        try:
            price_changes = [
                data['price_change_1h'],
                data['price_change_24h'], 
                data['price_change_7d'],
                data['price_change_30d']
            ]
            
            # Remove None values
            price_changes = [pc for pc in price_changes if pc is not None]
            
            if not price_changes:
                return {'volatility_score': 0.5, 'stability_rating': 'unknown'}
            
            volatility = np.std(price_changes)
            short_term_vol = abs(data['price_change_24h'] or 0)
            
            # Volatility score (0-1, where 1 is very volatile)
            volatility_score = min(volatility / 50, 1.0)  # Normalize to 50% as max
            
            # Stability rating
            if volatility_score < 0.1:
                stability = 'very_stable'
            elif volatility_score < 0.2:
                stability = 'stable'
            elif volatility_score < 0.4:
                stability = 'moderate'
            elif volatility_score < 0.6:
                stability = 'volatile'
            else:
                stability = 'very_volatile'
            
            return {
                'volatility_score': volatility_score,
                'short_term_volatility': short_term_vol / 100,
                'stability_rating': stability,
                'price_consistency': 1 - volatility_score
            }
            
        except Exception as e:
            logger.error(f"Error calculating volatility metrics: {e}")
            return {'volatility_score': 0.5, 'stability_rating': 'unknown'}
    
    def _calculate_momentum_indicators(self, data: Dict) -> Dict:
        """Calculate momentum-based indicators"""
        try:
            # Short, medium, long term momentum
            momentum_1h = data['price_change_1h'] or 0
            momentum_24h = data['price_change_24h'] or 0
            momentum_7d = data['price_change_7d'] or 0
            momentum_30d = data['price_change_30d'] or 0
            
            # Weighted momentum score
            momentum_score = (
                momentum_1h * 0.4 +
                momentum_24h * 0.3 +
                momentum_7d * 0.2 +
                momentum_30d * 0.1
            ) / 100
            
            # Trend strength (consistency across timeframes)
            trends = [momentum_1h, momentum_24h, momentum_7d, momentum_30d]
            trends = [t for t in trends if t is not None]
            
            if trends:
                positive_trends = sum(1 for t in trends if t > 0)
                trend_strength = positive_trends / len(trends)
                
                if trend_strength >= 0.75:
                    trend_direction = 'strong_bullish'
                elif trend_strength >= 0.6:
                    trend_direction = 'bullish'
                elif trend_strength >= 0.4:
                    trend_direction = 'neutral'
                elif trend_strength >= 0.25:
                    trend_direction = 'bearish'
                else:
                    trend_direction = 'strong_bearish'
            else:
                trend_direction = 'neutral'
                trend_strength = 0.5
            
            return {
                'momentum_score': momentum_score,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'short_term_momentum': momentum_1h / 100,
                'medium_term_momentum': momentum_24h / 100,
                'long_term_momentum': momentum_7d / 100
            }
            
        except Exception as e:
            logger.error(f"Error calculating momentum indicators: {e}")
            return {'momentum_score': 0, 'trend_direction': 'neutral'}
    
    def _calculate_market_position(self, data: Dict) -> Dict:
        """Calculate market position metrics"""
        try:
            market_cap = data['market_cap'] or 0
            volume_24h = data['volume_24h'] or 0
            rank = data['market_cap_rank'] or 1000
            
            # Market cap categories
            if market_cap > 100e9:  # >$100B
                cap_category = 'large_cap'
                cap_score = 1.0
            elif market_cap > 10e9:  # >$10B
                cap_category = 'mid_cap'
                cap_score = 0.8
            elif market_cap > 1e9:  # >$1B
                cap_category = 'small_cap' 
                cap_score = 0.6
            else:
                cap_category = 'micro_cap'
                cap_score = 0.4
            
            # Liquidity score based on volume/market cap ratio
            if market_cap > 0:
                liquidity_ratio = volume_24h / market_cap
                liquidity_score = min(liquidity_ratio / 0.1, 1.0)  # 10% daily volume = max liquidity
            else:
                liquidity_score = 0
            
            # Rank score (inverse - lower rank is better)
            rank_score = max(0, (1000 - rank) / 1000) if rank <= 1000 else 0
            
            return {
                'market_cap_category': cap_category,
                'market_cap_score': cap_score,
                'liquidity_score': liquidity_score,
                'rank_score': rank_score,
                'overall_market_strength': (cap_score + liquidity_score + rank_score) / 3
            }
            
        except Exception as e:
            logger.error(f"Error calculating market position: {e}")
            return {'market_cap_category': 'unknown', 'overall_market_strength': 0.5}
    
    def _calculate_risk_metrics(self, data: Dict) -> Dict:
        """Calculate risk assessment metrics"""
        try:
            # ATH/ATL analysis
            current_price = data['price'] or 0
            ath = data['ath'] or current_price
            atl = data['atl'] or current_price
            
            if ath > 0:
                distance_from_ath = (ath - current_price) / ath
            else:
                distance_from_ath = 0
            
            if atl > 0 and ath > atl:
                price_position = (current_price - atl) / (ath - atl)
            else:
                price_position = 0.5
            
            # Risk score based on multiple factors
            volatility_risk = min(abs(data['price_change_24h'] or 0) / 20, 1.0)  # 20% = max risk
            position_risk = 1 - price_position if price_position > 0.8 else 0  # Risk when near ATH
            
            overall_risk = (volatility_risk + position_risk) / 2
            
            # Risk category
            if overall_risk < 0.2:
                risk_category = 'low'
            elif overall_risk < 0.4:
                risk_category = 'moderate'
            elif overall_risk < 0.6:
                risk_category = 'high'
            else:
                risk_category = 'very_high'
            
            return {
                'overall_risk_score': overall_risk,
                'risk_category': risk_category,
                'distance_from_ath': distance_from_ath,
                'price_position': price_position,
                'volatility_risk': volatility_risk
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {'overall_risk_score': 0.5, 'risk_category': 'moderate'}
    
    def _calculate_opportunity_score(self, data: Dict) -> float:
        """Calculate overall opportunity score (0-1)"""
        try:
            # Factors that contribute to opportunity
            momentum = abs(data['price_change_24h'] or 0) / 100  # Absolute momentum
            volume_score = min((data['volume_24h'] or 0) / 1e9, 1.0)  # $1B volume = max score
            rank_bonus = max(0, (100 - (data['market_cap_rank'] or 100)) / 100)  # Top 100 bonus
            
            # Volatility provides opportunity but also risk
            volatility = abs(data['price_change_24h'] or 0) / 100
            volatility_opportunity = volatility * 0.5  # Moderate weight
            
            # Combine factors
            opportunity_score = (
                momentum * 0.3 +
                volume_score * 0.2 +
                rank_bonus * 0.2 +
                volatility_opportunity * 0.3
            )
            
            return min(opportunity_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating opportunity score: {e}")
            return 0.5
    
    async def get_trending_coins(self, limit: int = 10) -> List[Dict]:
        """Get trending coins"""
        cache_key = f"trending_{limit}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        url = f"{self.base_url}/search/trending"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        trending = []
                        coins = data.get('coins', [])[:limit]
                        
                        for coin_data in coins:
                            coin = coin_data.get('item', {})
                            trending.append({
                                'id': coin.get('id'),
                                'name': coin.get('name'),
                                'symbol': coin.get('symbol'),
                                'market_cap_rank': coin.get('market_cap_rank'),
                                'score': coin.get('score', 0)
                            })
                        
                        # Cache the results
                        self.cache[cache_key] = {
                            'data': trending,
                            'timestamp': time.time()
                        }
                        
                        return trending
                    else:
                        logger.error(f"Failed to get trending coins: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching trending coins: {e}")
            return []
    
    async def get_market_overview(self) -> Dict:
        """Get global market overview"""
        cache_key = "global_market"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key]['data']
        
        self._rate_limit()
        
        url = f"{self.base_url}/global"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        global_data = data.get('data', {})
                        
                        overview = {
                            'total_market_cap': global_data.get('total_market_cap', {}).get('usd', 0),
                            'total_volume_24h': global_data.get('total_volume', {}).get('usd', 0),
                            'bitcoin_dominance': global_data.get('market_cap_percentage', {}).get('btc', 0),
                            'ethereum_dominance': global_data.get('market_cap_percentage', {}).get('eth', 0),
                            'active_cryptocurrencies': global_data.get('active_cryptocurrencies', 0),
                            'markets': global_data.get('markets', 0),
                            'market_cap_change_24h': global_data.get('market_cap_change_percentage_24h_usd', 0),
                            'defi_dominance': global_data.get('defi_market_cap', 0),
                            'defi_volume_24h': global_data.get('defi_24h_vol', 0),
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
            'last_request_time': self.last_request_time,
            'api_tier': 'demo'
        }

# Example usage and testing
async def test_coingecko_provider():
    """Test the CoinGecko provider"""
    
    # Load API key from config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['coingecko']
    except (FileNotFoundError, KeyError):
        print("❌ CoinGecko API key not found in config.json")
        return
    
    provider = CoinGeckoProvider(api_key)
    
    print("🦎 Testing CoinGecko Provider...")
    print("=" * 50)
    
    # Test 1: Get coin data for major cryptocurrencies
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']
    print(f"\n📊 Getting coin data for: {', '.join(symbols)}")
    
    coin_data = await provider.get_coin_data(symbols)
    
    for symbol, data in coin_data.items():
        print(f"  {symbol}: ${data['price']:,.2f} "
              f"({data['price_change_24h']:+.2f}% 24h) "
              f"Rank #{data['market_cap_rank']}")
    
    # Test 2: Get advanced metrics for BTC
    print(f"\n🔍 Getting advanced metrics for BTC-USD...")
    metrics = await provider.get_advanced_metrics('BTC-USD')
    
    if metrics:
        print(f"  Price: ${metrics['basic_data']['price']:,.2f}")
        print(f"  Volatility Score: {metrics['volatility_analysis']['volatility_score']:.2f}")
        print(f"  Momentum Score: {metrics['momentum_indicators']['momentum_score']:+.3f}")
        print(f"  Trend: {metrics['momentum_indicators']['trend_direction']}")
        print(f"  Market Strength: {metrics['market_position']['overall_market_strength']:.2f}")
        print(f"  Risk Category: {metrics['risk_metrics']['risk_category']}")
        print(f"  Opportunity Score: {metrics['opportunity_score']:.2f}")
    
    # Test 3: Get trending coins
    print(f"\n🔥 Getting trending coins...")
    trending = await provider.get_trending_coins(5)
    
    for i, coin in enumerate(trending, 1):
        print(f"  {i}. {coin['name']} ({coin['symbol']}) - "
              f"Rank #{coin['market_cap_rank']} - Score: {coin['score']}")
    
    # Test 4: Market overview
    print(f"\n🌍 Global Market Overview...")
    overview = await provider.get_market_overview()
    
    if overview:
        print(f"  Total Market Cap: ${overview['total_market_cap']:,.0f}")
        print(f"  24h Volume: ${overview['total_volume_24h']:,.0f}")
        print(f"  Bitcoin Dominance: {overview['bitcoin_dominance']:.1f}%")
        print(f"  Ethereum Dominance: {overview['ethereum_dominance']:.1f}%")
        print(f"  Active Cryptocurrencies: {overview['active_cryptocurrencies']:,}")
        print(f"  Market Cap Change 24h: {overview['market_cap_change_24h']:+.2f}%")
    
    # Test 5: Cache statistics
    print(f"\n📈 Cache Statistics...")
    cache_stats = provider.get_cache_stats()
    print(f"  Total Entries: {cache_stats['total_entries']}")
    print(f"  Fresh Entries: {cache_stats['fresh_entries']}")
    print(f"  Cache Hit Ratio: {cache_stats['cache_hit_ratio']:.1%}")
    print(f"  API Tier: {cache_stats['api_tier']}")
    
    print(f"\n✅ CoinGecko provider test completed!")

if __name__ == "__main__":
    asyncio.run(test_coingecko_provider())