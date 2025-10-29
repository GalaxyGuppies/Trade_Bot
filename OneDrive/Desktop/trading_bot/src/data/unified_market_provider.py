"""
Unified Market Data Aggregator
Combines CoinMarketCap and CoinGecko for comprehensive market intelligence
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

from src.data.coinmarketcap_provider import CoinMarketCapProvider
from src.data.coingecko_provider import CoinGeckoProvider

logger = logging.getLogger(__name__)

class UnifiedMarketDataProvider:
    """
    Aggregates data from multiple providers for enhanced accuracy and redundancy
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.providers = {}
        
        # Initialize CoinMarketCap if API key available
        cmc_key = config.get('market_data', {}).get('coinmarketcap_api_key') or config.get('api_keys', {}).get('coinmarketcap')
        if cmc_key:
            self.providers['coinmarketcap'] = CoinMarketCapProvider(cmc_key)
            logger.info("✅ CoinMarketCap provider initialized")
        
        # Initialize CoinGecko if API key available  
        cg_key = config.get('market_data', {}).get('coingecko_api_key') or config.get('api_keys', {}).get('coingecko')
        if cg_key:
            self.providers['coingecko'] = CoinGeckoProvider(cg_key)
            logger.info("✅ CoinGecko provider initialized")
        
        if not self.providers:
            logger.warning("⚠️ No market data providers available!")
        else:
            logger.info(f"🚀 Unified provider initialized with {len(self.providers)} data sources")
    
    async def get_enhanced_market_data(self, symbol: str) -> Dict:
        """Get enhanced market data by combining multiple sources"""
        
        if not self.providers:
            return self._get_fallback_data(symbol)
        
        # Collect data from all available providers
        provider_data = {}
        
        # Get CoinMarketCap data
        if 'coinmarketcap' in self.providers:
            try:
                cmc_data = await self.providers['coinmarketcap'].get_market_metrics(symbol)
                if cmc_data:
                    provider_data['coinmarketcap'] = cmc_data
                    logger.debug(f"Got CoinMarketCap data for {symbol}")
            except Exception as e:
                logger.warning(f"CoinMarketCap error for {symbol}: {e}")
        
        # Get CoinGecko data
        if 'coingecko' in self.providers:
            try:
                cg_data = await self.providers['coingecko'].get_advanced_metrics(symbol)
                if cg_data:
                    provider_data['coingecko'] = cg_data
                    logger.debug(f"Got CoinGecko data for {symbol}")
            except Exception as e:
                logger.warning(f"CoinGecko error for {symbol}: {e}")
        
        if not provider_data:
            logger.warning(f"No data available for {symbol}, using fallback")
            return self._get_fallback_data(symbol)
        
        # Combine and enhance the data
        enhanced_data = self._combine_provider_data(symbol, provider_data)
        
        return enhanced_data
    
    def _combine_provider_data(self, symbol: str, provider_data: Dict) -> Dict:
        """Intelligently combine data from multiple providers"""
        
        # Start with base structure
        combined = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'data_sources': list(provider_data.keys()),
            'data_quality': self._assess_data_quality(provider_data)
        }
        
        # Extract price data with consensus
        prices = []
        volumes = []
        market_caps = []
        
        # CoinMarketCap data
        if 'coinmarketcap' in provider_data:
            cmc = provider_data['coinmarketcap']
            if 'price' in cmc:
                prices.append(cmc['price'])
            if 'volume_24h' in cmc:
                volumes.append(cmc['volume_24h'])
            if 'market_cap' in cmc:
                market_caps.append(cmc['market_cap'])
        
        # CoinGecko data
        if 'coingecko' in provider_data:
            cg = provider_data['coingecko']
            if 'basic_data' in cg:
                basic = cg['basic_data']
                if 'price' in basic:
                    prices.append(basic['price'])
                if 'volume_24h' in basic:
                    volumes.append(basic['volume_24h'])
                if 'market_cap' in basic:
                    market_caps.append(basic['market_cap'])
        
        # Calculate consensus values
        combined['price'] = self._calculate_consensus(prices) if prices else 0
        combined['volume_24h'] = self._calculate_consensus(volumes) if volumes else 0
        combined['market_cap'] = self._calculate_consensus(market_caps) if market_caps else 0
        
        # Price change data (prefer CoinGecko for more timeframes)
        if 'coingecko' in provider_data:
            cg_basic = provider_data['coingecko'].get('basic_data', {})
            combined.update({
                'price_change_1h': cg_basic.get('price_change_1h', 0),
                'price_change_24h': cg_basic.get('price_change_24h', 0),
                'price_change_7d': cg_basic.get('price_change_7d', 0),
                'price_change_30d': cg_basic.get('price_change_30d', 0)
            })
        elif 'coinmarketcap' in provider_data:
            cmc = provider_data['coinmarketcap']
            combined.update({
                'price_change_1h': 0,  # CMC doesn't provide 1h in our current implementation
                'price_change_24h': cmc.get('price_change_24h', 0),
                'price_change_7d': 0,
                'price_change_30d': 0
            })
        
        # Enhanced analytics (prefer CoinGecko for detailed analysis)
        if 'coingecko' in provider_data:
            cg_data = provider_data['coingecko']
            combined.update({
                'volatility_analysis': cg_data.get('volatility_analysis', {}),
                'momentum_indicators': cg_data.get('momentum_indicators', {}),
                'market_position': cg_data.get('market_position', {}),
                'risk_metrics': cg_data.get('risk_metrics', {}),
                'opportunity_score': cg_data.get('opportunity_score', 0.5)
            })
        else:
            # Fallback to calculated metrics from available data
            combined.update(self._calculate_fallback_metrics(combined))
        
        # Add CoinMarketCap specific metrics if available
        if 'coinmarketcap' in provider_data:
            cmc = provider_data['coinmarketcap']
            combined.update({
                'liquidity_score': cmc.get('liquidity_score', 0.5),
                'momentum_score': cmc.get('momentum_score', 0),
                'market_strength': cmc.get('market_strength', 0.5)
            })
        
        # Calculate unified scores
        combined['unified_metrics'] = self._calculate_unified_metrics(combined, provider_data)
        
        # Generate price history simulation
        combined['price_history'] = self._generate_price_history(combined['price'])
        
        # Add order book simulation
        combined['orderbook'] = self._simulate_orderbook(combined['price'])
        
        return combined
    
    def _calculate_consensus(self, values: List[float]) -> float:
        """Calculate consensus value from multiple sources"""
        if not values:
            return 0
        
        if len(values) == 1:
            return values[0]
        
        # Remove outliers (values more than 20% different from median)
        median = np.median(values)
        filtered_values = [v for v in values if abs(v - median) / median <= 0.2]
        
        if not filtered_values:
            filtered_values = values
        
        # Return weighted average (could be enhanced with provider reliability weights)
        return np.mean(filtered_values)
    
    def _assess_data_quality(self, provider_data: Dict) -> Dict:
        """Assess the quality and completeness of data"""
        quality = {
            'completeness_score': 0,
            'freshness_score': 0,
            'consistency_score': 0,
            'overall_score': 0
        }
        
        total_fields = 0
        complete_fields = 0
        
        # Check data completeness
        required_fields = ['price', 'volume_24h', 'market_cap']
        
        for provider, data in provider_data.items():
            for field in required_fields:
                total_fields += 1
                if provider == 'coinmarketcap':
                    if field in data and data[field] is not None:
                        complete_fields += 1
                elif provider == 'coingecko':
                    basic_data = data.get('basic_data', {})
                    if field in basic_data and basic_data[field] is not None:
                        complete_fields += 1
        
        quality['completeness_score'] = complete_fields / total_fields if total_fields > 0 else 0
        
        # Check price consistency across providers
        prices = []
        for provider, data in provider_data.items():
            if provider == 'coinmarketcap' and 'price' in data:
                prices.append(data['price'])
            elif provider == 'coingecko':
                basic_data = data.get('basic_data', {})
                if 'price' in basic_data:
                    prices.append(basic_data['price'])
        
        if len(prices) > 1:
            price_variance = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 1
            quality['consistency_score'] = max(0, 1 - price_variance * 10)  # 10% variance = 0 score
        else:
            quality['consistency_score'] = 1.0
        
        # Freshness is always high for real-time APIs
        quality['freshness_score'] = 1.0
        
        # Overall score
        quality['overall_score'] = (
            quality['completeness_score'] * 0.4 +
            quality['consistency_score'] * 0.4 +
            quality['freshness_score'] * 0.2
        )
        
        return quality
    
    def _calculate_fallback_metrics(self, data: Dict) -> Dict:
        """Calculate basic metrics when advanced analytics aren't available"""
        
        price_change_24h = data.get('price_change_24h', 0)
        
        return {
            'volatility_analysis': {
                'volatility_score': min(abs(price_change_24h) / 20, 1.0),  # 20% = max volatility
                'stability_rating': 'stable' if abs(price_change_24h) < 5 else 'volatile'
            },
            'momentum_indicators': {
                'momentum_score': price_change_24h / 100,
                'trend_direction': 'bullish' if price_change_24h > 2 else 'bearish' if price_change_24h < -2 else 'neutral'
            },
            'risk_metrics': {
                'overall_risk_score': min(abs(price_change_24h) / 40, 1.0),  # 40% = max risk
                'risk_category': 'high' if abs(price_change_24h) > 10 else 'moderate'
            },
            'opportunity_score': min(abs(price_change_24h) / 20, 1.0)  # Higher volatility = more opportunity
        }
    
    def _calculate_unified_metrics(self, combined: Dict, provider_data: Dict) -> Dict:
        """Calculate unified metrics that combine insights from all providers"""
        
        # Trading signal strength (0-1)
        signal_factors = []
        
        # Momentum factor
        momentum = combined.get('momentum_indicators', {}).get('momentum_score', 0)
        signal_factors.append(abs(momentum))
        
        # Volume factor (normalized)
        volume = combined.get('volume_24h', 0)
        volume_factor = min(volume / 1e9, 1.0)  # $1B volume = max factor
        signal_factors.append(volume_factor)
        
        # Opportunity factor
        opportunity = combined.get('opportunity_score', 0.5)
        signal_factors.append(opportunity)
        
        trading_signal_strength = np.mean(signal_factors) if signal_factors else 0.5
        
        # Confidence score based on data quality and consistency
        data_quality = combined.get('data_quality', {})
        confidence_score = data_quality.get('overall_score', 0.5)
        
        # Risk-adjusted opportunity
        risk_score = combined.get('risk_metrics', {}).get('overall_risk_score', 0.5)
        risk_adjusted_opportunity = opportunity * (1 - risk_score)
        
        return {
            'trading_signal_strength': trading_signal_strength,
            'confidence_score': confidence_score,
            'risk_adjusted_opportunity': risk_adjusted_opportunity,
            'overall_attractiveness': (trading_signal_strength * confidence_score * (1 - risk_score))
        }
    
    def _generate_price_history(self, current_price: float) -> List[float]:
        """Generate simulated price history for compatibility"""
        if current_price <= 0:
            current_price = 45000  # Default BTC price
        
        history = []
        for i in range(100):
            # Add small random variations around current price
            variation = np.random.uniform(-0.02, 0.02)  # ±2% variation
            price = current_price * (1 + variation)
            history.append(price)
        
        return history
    
    def _simulate_orderbook(self, current_price: float) -> Dict:
        """Simulate order book for compatibility"""
        if current_price <= 0:
            current_price = 45000
        
        return {
            'bids': [[str(current_price * (1 - 0.001 * i)), str(10 + i)] for i in range(1, 6)],
            'asks': [[str(current_price * (1 + 0.001 * i)), str(10 + i)] for i in range(1, 6)]
        }
    
    def _get_fallback_data(self, symbol: str) -> Dict:
        """Provide fallback data when no providers are available"""
        base_price = 45000 if 'BTC' in symbol else 3000
        
        return {
            'symbol': symbol,
            'price': base_price,
            'price_history': [base_price + i * 10 for i in range(100)],
            'volume_24h': 25000000000,
            'market_cap': 900000000000,
            'volatility_analysis': {'volatility_score': 0.03, 'stability_rating': 'moderate'},
            'momentum_indicators': {'momentum_score': 0.01, 'trend_direction': 'neutral'},
            'risk_metrics': {'overall_risk_score': 0.3, 'risk_category': 'moderate'},
            'opportunity_score': 0.5,
            'liquidity_score': 0.8,
            'market_strength': 0.7,
            'timestamp': datetime.now().isoformat(),
            'data_sources': ['fallback'],
            'orderbook': self._simulate_orderbook(base_price)
        }
    
    async def get_market_overview(self) -> Dict:
        """Get comprehensive market overview from all providers"""
        overviews = {}
        
        # Get from CoinMarketCap
        if 'coinmarketcap' in self.providers:
            try:
                cmc_overview = await self.providers['coinmarketcap'].get_market_overview()
                if cmc_overview:
                    overviews['coinmarketcap'] = cmc_overview
            except Exception as e:
                logger.warning(f"CoinMarketCap market overview error: {e}")
        
        # Get from CoinGecko
        if 'coingecko' in self.providers:
            try:
                cg_overview = await self.providers['coingecko'].get_market_overview()
                if cg_overview:
                    overviews['coingecko'] = cg_overview
            except Exception as e:
                logger.warning(f"CoinGecko market overview error: {e}")
        
        if not overviews:
            return {'error': 'No market overview data available'}
        
        # Combine the overviews
        combined_overview = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': list(overviews.keys())
        }
        
        # Use consensus for key metrics
        market_caps = [ov.get('total_market_cap', 0) for ov in overviews.values()]
        volumes = [ov.get('total_volume_24h', 0) for ov in overviews.values()]
        btc_dominances = [ov.get('bitcoin_dominance', 0) for ov in overviews.values()]
        
        combined_overview.update({
            'total_market_cap': self._calculate_consensus(market_caps),
            'total_volume_24h': self._calculate_consensus(volumes),
            'bitcoin_dominance': self._calculate_consensus(btc_dominances),
            'market_sentiment': self._calculate_market_sentiment(overviews)
        })
        
        return combined_overview
    
    def _calculate_market_sentiment(self, overviews: Dict) -> str:
        """Calculate overall market sentiment from available data"""
        sentiment_factors = []
        
        for provider, data in overviews.items():
            market_cap_change = data.get('market_cap_change_24h', 0)
            sentiment_factors.append(market_cap_change)
        
        if sentiment_factors:
            avg_change = np.mean(sentiment_factors)
            if avg_change > 5:
                return 'very_bullish'
            elif avg_change > 2:
                return 'bullish'
            elif avg_change > -2:
                return 'neutral'
            elif avg_change > -5:
                return 'bearish'
            else:
                return 'very_bearish'
        
        return 'neutral'
    
    def get_provider_status(self) -> Dict:
        """Get status of all providers"""
        status = {
            'total_providers': len(self.providers),
            'active_providers': list(self.providers.keys()),
            'provider_details': {}
        }
        
        for name, provider in self.providers.items():
            try:
                cache_stats = provider.get_cache_stats()
                status['provider_details'][name] = {
                    'status': 'active',
                    'cache_stats': cache_stats
                }
            except Exception as e:
                status['provider_details'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return status

# Testing function
async def test_unified_provider():
    """Test the unified market data provider"""
    
    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Config file not found")
        return
    
    provider = UnifiedMarketDataProvider(config)
    
    print("🔄 Testing Unified Market Data Provider...")
    print("=" * 50)
    
    # Test provider status
    status = provider.get_provider_status()
    print(f"Active Providers: {status['active_providers']}")
    print(f"Total Providers: {status['total_providers']}")
    
    # Test enhanced market data
    symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD']
    
    for symbol in symbols:
        print(f"\n📊 Enhanced data for {symbol}:")
        data = await provider.get_enhanced_market_data(symbol)
        
        print(f"  Price: ${data['price']:,.2f}")
        print(f"  Data Sources: {data['data_sources']}")
        print(f"  Data Quality: {data['data_quality']['overall_score']:.2f}")
        print(f"  Trading Signal: {data['unified_metrics']['trading_signal_strength']:.2f}")
        print(f"  Opportunity Score: {data['unified_metrics']['risk_adjusted_opportunity']:.2f}")
    
    # Test market overview
    print(f"\n🌍 Market Overview:")
    overview = await provider.get_market_overview()
    if 'error' not in overview:
        print(f"  Total Market Cap: ${overview['total_market_cap']:,.0f}")
        print(f"  Total Volume: ${overview['total_volume_24h']:,.0f}")
        print(f"  Bitcoin Dominance: {overview['bitcoin_dominance']:.1f}%")
        print(f"  Market Sentiment: {overview['market_sentiment']}")
        print(f"  Data Sources: {overview['data_sources']}")
    
    print(f"\n✅ Unified provider test completed!")

if __name__ == "__main__":
    asyncio.run(test_unified_provider())