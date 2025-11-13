"""
Moralis Blockchain Analytics Provider
Advanced on-chain analysis for token research, whale tracking, and DEX analysis
"""

import asyncio
import aiohttp
import json
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TokenMetrics:
    """Token on-chain metrics from Moralis"""
    address: str
    symbol: str
    name: str
    decimals: int
    total_supply: float
    holder_count: int
    market_cap: float
    price_usd: float
    volume_24h: float
    liquidity_usd: float
    created_at: datetime
    creator_address: str
    verified: bool
    audit_status: str

@dataclass 
class WalletAnalysis:
    """Whale wallet analysis"""
    address: str
    balance_usd: float
    token_holdings: Dict[str, float]
    transaction_count: int
    first_seen: datetime
    last_active: datetime
    profit_loss: float
    risk_score: float
    labels: List[str]

@dataclass
class DexTradeAnalysis:
    """DEX trading analysis"""
    token_address: str
    dex_name: str
    volume_24h: float
    trades_24h: int
    buyers_24h: int
    sellers_24h: int
    avg_trade_size: float
    price_impact: float
    liquidity_score: float
    whale_activity: bool

class MoralisProvider:
    """
    Moralis Web3 API provider for comprehensive blockchain analysis
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://deep-index.moralis.io/api/v2"  # Updated to v2
        self.headers = {
            "X-API-Key": api_key,
            "accept": "application/json"
        }
        
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        self.last_request = 0
        self.rate_limit_delay = 0.2  # 200ms between requests
        
        # Supported chains
        self.chains = {
            'ethereum': '0x1',
            'bsc': '0x38', 
            'polygon': '0x89',
            'avalanche': '0xa86a',
            'arbitrum': '0xa4b1',
            'base': '0x2105',
            'optimism': '0xa',
            'fantom': '0xfa'
        }
    
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make rate-limited request to Moralis API"""
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    self.last_request = time.time()
                    
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 429:
                        logger.warning("⏰ Moralis API rate limit hit, waiting...")
                        await asyncio.sleep(2)
                        return None
                    else:
                        logger.error(f"❌ Moralis API error {response.status}: {await response.text()}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Moralis request failed: {e}")
            return None
    
    def _should_use_cache(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key in self.cache:
            cached_time, _ = self.cache[cache_key]
            return time.time() - cached_time < self.cache_ttl
        return False
    
    async def get_token_metadata(self, token_address: str, chain: str = 'ethereum') -> Optional[TokenMetrics]:
        """Get comprehensive token metadata"""
        cache_key = f"token_metadata_{chain}_{token_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/erc20/{token_address}/metadata"  # Updated endpoint
        params = {"chain": chain_id}
        
        result = await self._make_request(endpoint, params)
        
        if result:
            # Get additional token stats if available
            price_data = await self.get_token_price(token_address, chain)
            
            token_metrics = TokenMetrics(
                address=token_address,
                symbol=result.get('symbol', ''),
                name=result.get('name', ''),
                decimals=int(result.get('decimals', 18)),
                total_supply=float(result.get('total_supply', 0)) / (10 ** int(result.get('decimals', 18))),
                holder_count=0,  # Not available in metadata endpoint
                market_cap=0,    # Would need price calculation
                price_usd=price_data.get('usdPrice', 0) if price_data else 0,
                volume_24h=0,    # Not available in this endpoint
                liquidity_usd=0, # Would need DEX data
                created_at=datetime.now(),
                creator_address='',
                verified=result.get('verified_contract', False),
                audit_status='unknown'
            )
            
            self.cache[cache_key] = (time.time(), token_metrics)
            return token_metrics
        
        return None
    
    async def get_token_stats(self, token_address: str, chain: str = 'ethereum') -> Optional[Dict]:
        """Get token statistics and analytics"""
        cache_key = f"token_stats_{chain}_{token_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/erc20/{token_address}/stats"
        params = {"chain": chain_id}
        
        result = await self._make_request(endpoint, params)
        
        if result:
            self.cache[cache_key] = (time.time(), result)
            return result
        
        return None
    
    async def get_token_price(self, token_address: str, chain: str = 'ethereum') -> Optional[Dict]:
        """Get current token price data"""
        cache_key = f"token_price_{chain}_{token_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/erc20/{token_address}/price"
        params = {"chain": chain_id}
        
        result = await self._make_request(endpoint, params)
        
        if result:
            self.cache[cache_key] = (time.time(), result)
            return result
        
        return None
    
    async def get_wallet_token_balances(self, wallet_address: str, chain: str = 'ethereum') -> List[Dict]:
        """Get all token balances for a wallet"""
        cache_key = f"wallet_balances_{chain}_{wallet_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/{wallet_address}/erc20"
        params = {"chain": chain_id}
        
        result = await self._make_request(endpoint, params)
        
        if result:
            balances = result.get('result', [])
            self.cache[cache_key] = (time.time(), balances)
            return balances
        
        return []
    
    async def analyze_whale_wallet(self, wallet_address: str, chain: str = 'ethereum') -> Optional[WalletAnalysis]:
        """Comprehensive whale wallet analysis"""
        cache_key = f"whale_analysis_{chain}_{wallet_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        # Get wallet balances
        balances = await self.get_wallet_token_balances(wallet_address, chain)
        
        # Get transaction history
        transactions = await self.get_wallet_transactions(wallet_address, chain)
        
        # Calculate metrics
        total_value_usd = 0
        token_holdings = {}
        
        for balance in balances:
            symbol = balance.get('symbol', 'UNKNOWN')
            balance_formatted = float(balance.get('balance_formatted', 0))
            usd_value = float(balance.get('usd_value', 0))
            
            token_holdings[symbol] = balance_formatted
            total_value_usd += usd_value
        
        # Analyze transaction patterns
        if transactions:
            first_tx = min(transactions, key=lambda x: x.get('block_timestamp', ''))
            last_tx = max(transactions, key=lambda x: x.get('block_timestamp', ''))
            
            first_seen = datetime.fromisoformat(first_tx.get('block_timestamp', '').replace('Z', '+00:00'))
            last_active = datetime.fromisoformat(last_tx.get('block_timestamp', '').replace('Z', '+00:00'))
        else:
            first_seen = datetime.now()
            last_active = datetime.now()
        
        # Calculate risk score (simple heuristic)
        risk_score = self._calculate_wallet_risk_score(
            total_value_usd, len(token_holdings), len(transactions or [])
        )
        
        analysis = WalletAnalysis(
            address=wallet_address,
            balance_usd=total_value_usd,
            token_holdings=token_holdings,
            transaction_count=len(transactions or []),
            first_seen=first_seen,
            last_active=last_active,
            profit_loss=0.0,  # Would need complex calculation
            risk_score=risk_score,
            labels=self._get_wallet_labels(total_value_usd, len(token_holdings))
        )
        
        self.cache[cache_key] = (time.time(), analysis)
        return analysis
    
    async def get_wallet_transactions(self, wallet_address: str, chain: str = 'ethereum', limit: int = 50) -> List[Dict]:
        """Get wallet transaction history with reduced limit"""
        cache_key = f"wallet_txs_{chain}_{wallet_address}_{limit}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/{wallet_address}"
        params = {
            "chain": chain_id,
            "limit": min(limit, 100)  # Respect API limits
        }
        
        result = await self._make_request(endpoint, params)
        
        if result:
            transactions = result.get('result', [])
            self.cache[cache_key] = (time.time(), transactions)
            return transactions
        
        return []
    
    async def get_dex_trades(self, token_address: str, chain: str = 'ethereum', hours: int = 24) -> Optional[DexTradeAnalysis]:
        """Get DEX trading analysis for token"""
        cache_key = f"dex_trades_{chain}_{token_address}_{hours}h"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        
        # Get token transfers (proxy for DEX trades) with reduced limit
        endpoint = f"/erc20/{token_address}/transfers"
        params = {
            "chain": chain_id,
            "limit": 100  # Reduced from 500 to respect API limits
        }
        
        result = await self._make_request(endpoint, params)
        
        if result:
            transfers = result.get('result', [])
            analysis = self._analyze_dex_activity(token_address, transfers, hours)
            
            self.cache[cache_key] = (time.time(), analysis)
            return analysis
        
        return None
    
    async def detect_rugpull_signals(self, token_address: str, chain: str = 'ethereum') -> Dict[str, float]:
        """Detect potential rugpull signals"""
        cache_key = f"rugpull_signals_{chain}_{token_address}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        signals = {}
        
        # Get token metadata
        token_data = await self.get_token_metadata(token_address, chain)
        if not token_data:
            return {}
        
        # Signal 1: Contract verification
        signals['unverified_contract'] = 0.8 if not token_data.verified else 0.0
        
        # Signal 2: Low holder count
        if token_data.holder_count < 100:
            signals['low_holder_count'] = 0.7
        elif token_data.holder_count < 500:
            signals['low_holder_count'] = 0.4
        else:
            signals['low_holder_count'] = 0.0
        
        # Signal 3: High concentration (need to implement)
        signals['high_concentration'] = 0.0  # Placeholder
        
        # Signal 4: Liquidity concerns
        if token_data.liquidity_usd < 10000:
            signals['low_liquidity'] = 0.9
        elif token_data.liquidity_usd < 50000:
            signals['low_liquidity'] = 0.5
        else:
            signals['low_liquidity'] = 0.0
        
        # Signal 5: New token (higher risk)
        token_age_hours = (datetime.now() - token_data.created_at).total_seconds() / 3600
        if token_age_hours < 24:
            signals['very_new_token'] = 0.8
        elif token_age_hours < 168:  # 1 week
            signals['new_token'] = 0.4
        else:
            signals['new_token'] = 0.0
        
        # Calculate overall risk score
        if signals:
            signals['overall_risk'] = min(1.0, sum(signals.values()) / len(signals))
        else:
            signals['overall_risk'] = 0.5
        
        self.cache[cache_key] = (time.time(), signals)
        return signals
    
    def _calculate_wallet_risk_score(self, balance_usd: float, token_count: int, tx_count: int) -> float:
        """Calculate wallet risk score based on behavior"""
        risk_score = 0.0
        
        # High balance = lower risk
        if balance_usd > 1000000:  # $1M+
            risk_score += 0.1
        elif balance_usd > 100000:  # $100K+
            risk_score += 0.3
        else:
            risk_score += 0.6
        
        # Diversification
        if token_count > 20:
            risk_score += 0.1
        elif token_count > 5:
            risk_score += 0.3
        else:
            risk_score += 0.5
        
        # Activity level
        if tx_count > 1000:
            risk_score += 0.1
        elif tx_count > 100:
            risk_score += 0.3
        else:
            risk_score += 0.5
        
        return min(1.0, risk_score / 3)
    
    def _get_wallet_labels(self, balance_usd: float, token_count: int) -> List[str]:
        """Generate wallet labels based on behavior"""
        labels = []
        
        if balance_usd > 1000000:
            labels.append('mega_whale')
        elif balance_usd > 100000:
            labels.append('whale')
        elif balance_usd > 10000:
            labels.append('dolphin')
        else:
            labels.append('retail')
        
        if token_count > 50:
            labels.append('diversified')
        elif token_count > 20:
            labels.append('moderate_portfolio')
        elif token_count < 5:
            labels.append('concentrated')
        
        return labels
    
    def _analyze_dex_activity(self, token_address: str, transfers: List[Dict], hours: int) -> DexTradeAnalysis:
        """Analyze DEX trading activity from transfers"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_transfers = []
        total_volume = 0
        unique_addresses = set()
        large_transfers = 0
        
        for transfer in transfers:
            transfer_time = datetime.fromisoformat(transfer.get('block_timestamp', '').replace('Z', '+00:00'))
            
            if transfer_time > cutoff_time:
                recent_transfers.append(transfer)
                
                # Calculate USD value (simplified)
                value = float(transfer.get('value_formatted', 0))
                total_volume += value
                
                unique_addresses.add(transfer.get('from_address', ''))
                unique_addresses.add(transfer.get('to_address', ''))
                
                # Detect large transfers (potential whale activity)
                if value > 10000:  # $10K+ transfers
                    large_transfers += 1
        
        return DexTradeAnalysis(
            token_address=token_address,
            dex_name='unknown',  # Would need DEX detection
            volume_24h=total_volume,
            trades_24h=len(recent_transfers),
            buyers_24h=len(unique_addresses) // 2,  # Rough estimate
            sellers_24h=len(unique_addresses) // 2,
            avg_trade_size=total_volume / max(1, len(recent_transfers)),
            price_impact=0.0,  # Would need price data
            liquidity_score=min(1.0, total_volume / 100000),  # Rough estimate
            whale_activity=large_transfers > 0
        )
    
    async def get_top_holders(self, token_address: str, chain: str = 'ethereum', limit: int = 20) -> List[Dict]:
        """Get top token holders"""
        cache_key = f"top_holders_{chain}_{token_address}_{limit}"
        
        if self._should_use_cache(cache_key):
            return self.cache[cache_key][1]
        
        chain_id = self.chains.get(chain, '0x1')
        endpoint = f"/erc20/{token_address}/owners"
        params = {
            "chain": chain_id,
            "limit": limit
        }
        
        result = await self._make_request(endpoint, params)
        
        if result:
            holders = result.get('result', [])
            self.cache[cache_key] = (time.time(), holders)
            return holders
        
        return []
    
    async def get_comprehensive_token_analysis(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """Get comprehensive analysis combining all Moralis data"""
        logger.info(f"🔍 Starting comprehensive analysis for {token_address} on {chain}")
        
        analysis = {
            'token_metadata': None,
            'rugpull_signals': {},
            'dex_analysis': None,
            'top_holders': [],
            'holder_concentration': {},
            'risk_assessment': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get basic token data
            analysis['token_metadata'] = await self.get_token_metadata(token_address, chain)
            
            # Get rugpull signals
            analysis['rugpull_signals'] = await self.detect_rugpull_signals(token_address, chain)
            
            # Get DEX trading analysis
            analysis['dex_analysis'] = await self.get_dex_trades(token_address, chain)
            
            # Get top holders for concentration analysis
            holders = await self.get_top_holders(token_address, chain)
            analysis['top_holders'] = holders
            
            # Calculate holder concentration
            if holders and analysis['token_metadata']:
                total_supply = analysis['token_metadata'].total_supply
                
                if total_supply > 0:
                    top_10_percentage = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:10])
                    top_5_percentage = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:5])
                    
                    analysis['holder_concentration'] = {
                        'top_5_holders_percentage': top_5_percentage,
                        'top_10_holders_percentage': top_10_percentage,
                        'concentration_risk': 'high' if top_5_percentage > 50 else 'medium' if top_5_percentage > 25 else 'low'
                    }
            
            # Overall risk assessment
            risk_factors = []
            risk_score = 0.0
            
            if analysis['rugpull_signals']:
                risk_score += analysis['rugpull_signals'].get('overall_risk', 0.5) * 0.4
                if analysis['rugpull_signals'].get('overall_risk', 0) > 0.7:
                    risk_factors.append('High rugpull risk signals')
            
            if analysis['holder_concentration'].get('top_5_holders_percentage', 0) > 50:
                risk_score += 0.3
                risk_factors.append('High holder concentration')
            
            if analysis['token_metadata'] and analysis['token_metadata'].liquidity_usd < 50000:
                risk_score += 0.2
                risk_factors.append('Low liquidity')
            
            if analysis['dex_analysis'] and analysis['dex_analysis'].trades_24h < 10:
                risk_score += 0.1
                risk_factors.append('Low trading activity')
            
            analysis['risk_assessment'] = {
                'overall_risk_score': min(1.0, risk_score),
                'risk_level': 'high' if risk_score > 0.7 else 'medium' if risk_score > 0.4 else 'low',
                'risk_factors': risk_factors,
                'recommendation': 'avoid' if risk_score > 0.8 else 'caution' if risk_score > 0.5 else 'consider'
            }
            
            logger.info(f"✅ Comprehensive analysis completed for {token_address}")
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive analysis: {e}")
        
        return analysis
    
    async def getDiscoveryToken(self, chain: str = 'ethereum', limit: int = 50) -> List[Dict]:
        """Get discovery tokens using available Moralis endpoints"""
        try:
            # Try the most likely working endpoints for token discovery
            endpoints = [
                f"/erc20/metadata",  # Get token metadata
                f"/token/transfers",  # Token transfers
                f"/{chain}/erc20/metadata"  # Chain-specific metadata
            ]
            
            for endpoint in endpoints:
                try:
                    params = {'limit': limit}
                    if 'metadata' in endpoint:
                        # For metadata endpoints, we need specific token addresses
                        # This is more of a utility method, so return empty for now
                        continue
                    
                    result = await self._make_request(endpoint, params)
                    if result and result.get('result'):
                        return result['result'][:limit]
                        
                except Exception as e:
                    logger.debug(f"Discovery endpoint {endpoint} failed: {e}")
                    continue
            
            logger.info("💡 getDiscoveryToken: Using transfers endpoint as discovery method")
            # Fallback: use recent transfers to discover active tokens
            transfers_result = await self._make_request("/erc20/transfers", {'limit': 100})
            
            if transfers_result and transfers_result.get('result'):
                # Extract unique tokens from transfers
                seen_tokens = set()
                discovered_tokens = []
                
                for transfer in transfers_result['result']:
                    token_address = transfer.get('address', '').lower()
                    if token_address and token_address not in seen_tokens:
                        seen_tokens.add(token_address)
                        discovered_tokens.append({
                            'address': token_address,
                            'symbol': transfer.get('symbol', ''),
                            'name': transfer.get('name', ''),
                            'decimals': transfer.get('decimals', 18),
                            'discovery_method': 'transfers'
                        })
                        
                        if len(discovered_tokens) >= limit:
                            break
                
                return discovered_tokens
                
        except Exception as e:
            logger.warning(f"getDiscoveryToken failed: {e}")
            
        return []
    
    async def getFilteredTokens(self, chain: str = 'ethereum', min_volume: float = 50000, 
                               min_market_cap: float = 500000, limit: int = 50) -> List[Dict]:
        """Get filtered tokens based on volume and market cap criteria"""
        try:
            # Since Moralis doesn't have direct filtering endpoints that work,
            # we'll use a combination of available endpoints and filter ourselves
            
            # Get recent token transfers to find active tokens
            transfers_result = await self._make_request("/erc20/transfers", {'limit': 200})
            
            if transfers_result and transfers_result.get('result'):
                # Count activity by token
                token_activity = {}
                
                for transfer in transfers_result['result']:
                    token_address = transfer.get('address', '').lower()
                    if token_address:
                        if token_address not in token_activity:
                            token_activity[token_address] = {
                                'count': 0,
                                'symbol': transfer.get('symbol', ''),
                                'name': transfer.get('name', ''),
                                'decimals': transfer.get('decimals', 18),
                                'total_value': 0
                            }
                        
                        token_activity[token_address]['count'] += 1
                        # Estimate value from transfer
                        value = float(transfer.get('value', 0)) / (10 ** transfer.get('decimals', 18))
                        token_activity[token_address]['total_value'] += value
                
                # Sort by activity and create filtered results
                sorted_tokens = sorted(token_activity.items(), 
                                     key=lambda x: x[1]['count'], 
                                     reverse=True)
                
                filtered_tokens = []
                for token_address, activity in sorted_tokens[:limit]:
                    # Estimate metrics based on activity
                    estimated_volume = activity['count'] * 1000  # Rough estimate
                    estimated_market_cap = estimated_volume * 10  # Conservative estimate
                    
                    if estimated_volume >= min_volume and estimated_market_cap >= min_market_cap:
                        filtered_tokens.append({
                            'address': token_address,
                            'symbol': activity['symbol'],
                            'name': activity['name'],
                            'decimals': activity['decimals'],
                            'estimated_volume_24h': estimated_volume,
                            'estimated_market_cap': estimated_market_cap,
                            'activity_count': activity['count'],
                            'discovery_method': 'activity_filter'
                        })
                
                return filtered_tokens
                
        except Exception as e:
            logger.warning(f"getFilteredTokens failed: {e}")
            
        return []
    
    async def getMultipleTokenAnalytics(self, token_addresses: List[str], 
                                      chain: str = 'ethereum') -> Dict[str, Dict]:
        """Get analytics for multiple tokens in batch"""
        try:
            analytics = {}
            
            # Process in small batches to avoid rate limits
            batch_size = 5
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i:i + batch_size]
                
                for address in batch:
                    try:
                        # Get basic token info
                        token_info = await self.get_token_metadata(address, chain)
                        if token_info:
                            analytics[address.lower()] = {
                                'metadata': {
                                    'symbol': token_info.symbol,
                                    'name': token_info.name,
                                    'decimals': token_info.decimals,
                                    'total_supply': token_info.total_supply
                                },
                                'price': None,
                                'stats': None
                            }
                        
                        # Get price if available
                        try:
                            price_data = await self.get_token_price(address, chain)
                            if price_data and address.lower() in analytics:
                                analytics[address.lower()]['price'] = price_data
                        except:
                            pass
                        
                        # Add a small delay between tokens
                        await asyncio.sleep(0.2)
                            
                    except Exception as token_error:
                        logger.debug(f"Failed to get analytics for {address}: {token_error}")
                        continue
                
                # Longer delay between batches
                await asyncio.sleep(1.0)
            
            return analytics
            
        except Exception as e:
            logger.error(f"getMultipleTokenAnalytics failed: {e}")
            return {}

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_entries = len(self.cache)
        fresh_entries = sum(1 for cached_time, _ in self.cache.values() 
                          if time.time() - cached_time < self.cache_ttl)
        
        return {
            'total_entries': total_entries,
            'fresh_entries': fresh_entries,
            'cache_hit_ratio': fresh_entries / max(1, total_entries)
        }


# Example usage and testing
async def test_moralis_provider():
    """Test the Moralis provider"""
    # Load API key from config
    import json
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['moralis']
    except:
        print("❌ Please add your Moralis API key to config.json")
        return
    
    provider = MoralisProvider(api_key)
    
    print("🧪 Testing Moralis Provider...")
    
    # Test token metadata
    print("\n1️⃣ Testing Token Metadata...")
    usdc_address = "0xA0b86a33E6441b811c0a1b26fb18b5b3d05db2b0"  # USDC on Ethereum
    metadata = await provider.get_token_metadata(usdc_address)
    
    if metadata:
        print(f"  Token: {metadata.name} ({metadata.symbol})")
        print(f"  Total Supply: {metadata.total_supply:,.0f}")
        print(f"  Holders: {metadata.holder_count:,}")
        print(f"  Verified: {metadata.verified}")
    
    # Test whale analysis
    print("\n2️⃣ Testing Whale Analysis...")
    whale_address = "0xf977814e90da44bfa03b6295a0616a897441acec"  # Binance Hot Wallet
    whale_analysis = await provider.analyze_whale_wallet(whale_address)
    
    if whale_analysis:
        print(f"  Wallet Value: ${whale_analysis.balance_usd:,.0f}")
        print(f"  Token Holdings: {len(whale_analysis.token_holdings)}")
        print(f"  Risk Score: {whale_analysis.risk_score:.2f}")
        print(f"  Labels: {', '.join(whale_analysis.labels)}")
    
    # Test rugpull detection
    print("\n3️⃣ Testing Rugpull Detection...")
    test_token = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"  # UNI token
    rugpull_signals = await provider.detect_rugpull_signals(test_token)
    
    if rugpull_signals:
        print(f"  Overall Risk: {rugpull_signals.get('overall_risk', 0):.2f}")
        for signal, score in rugpull_signals.items():
            if signal != 'overall_risk' and score > 0:
                print(f"  {signal}: {score:.2f}")
    
    # Test comprehensive analysis
    print("\n4️⃣ Testing Comprehensive Analysis...")
    analysis = await provider.get_comprehensive_token_analysis(test_token)
    
    if analysis['risk_assessment']:
        print(f"  Risk Level: {analysis['risk_assessment']['risk_level']}")
        print(f"  Recommendation: {analysis['risk_assessment']['recommendation']}")
        if analysis['risk_assessment']['risk_factors']:
            print(f"  Risk Factors: {', '.join(analysis['risk_assessment']['risk_factors'])}")
    
    async def getDiscoveryToken(self, chain: str = 'ethereum', limit: int = 50) -> List[Dict]:
        """Get discovery tokens using Moralis Discovery API"""
        try:
            url = f"{self.base_url}/discovery/tokens"
            params = {
                'chain': chain,
                'limit': limit,
                'order': 'DESC'
            }
            
            result = await self._make_request(url, params)
            
            if result and result.get('result'):
                return result['result']
            
        except Exception as e:
            logger.warning(f"getDiscoveryToken failed: {e}")
            
        return []
    
    async def getFilteredTokens(self, chain: str = 'ethereum', min_volume: float = 50000, 
                               min_market_cap: float = 500000, limit: int = 50) -> List[Dict]:
        """Get filtered tokens based on volume and market cap criteria"""
        try:
            # Try multiple discovery endpoints
            endpoints = [
                f"{self.base_url}/market-data/erc20s/trending",
                f"{self.base_url}/market-data/erc20s/top-gainers",
                f"{self.base_url}/discovery/trending"
            ]
            
            for endpoint in endpoints:
                try:
                    params = {
                        'chain': chain,
                        'limit': limit,
                        'min_volume': min_volume,
                        'min_market_cap': min_market_cap
                    }
                    
                    result = await self._make_request(endpoint, params)
                    
                    if result and (result.get('result') or result.get('tokens')):
                        tokens = result.get('result') or result.get('tokens')
                        
                        # Filter based on criteria
                        filtered = []
                        for token in tokens:
                            volume = float(token.get('volume_24h', token.get('volume', 0)))
                            market_cap = float(token.get('market_cap', 0))
                            
                            if volume >= min_volume and market_cap >= min_market_cap:
                                filtered.append(token)
                        
                        if filtered:
                            return filtered
                            
                except Exception as endpoint_error:
                    logger.warning(f"Endpoint {endpoint.split('/')[-1]} failed: {endpoint_error}")
                    continue
            
        except Exception as e:
            logger.warning(f"getFilteredTokens failed: {e}")
            
        return []
    
    async def getMultipleTokenAnalytics(self, token_addresses: List[str], 
                                      chain: str = 'ethereum') -> Dict[str, Dict]:
        """Get analytics for multiple tokens in batch"""
        try:
            analytics = {}
            
            # Process in batches to avoid rate limits
            batch_size = 10
            for i in range(0, len(token_addresses), batch_size):
                batch = token_addresses[i:i + batch_size]
                
                # Get batch of token metadata
                metadata_url = f"{self.base_url}/erc20/metadata"
                metadata_params = {
                    'chain': chain,
                    'addresses': batch
                }
                
                metadata_result = await self._make_request(metadata_url, metadata_params)
                
                if metadata_result and metadata_result.get('result'):
                    for token_meta in metadata_result['result']:
                        address = token_meta.get('address', '').lower()
                        if address:
                            analytics[address] = {
                                'metadata': token_meta,
                                'price': None,
                                'stats': None
                            }
                
                # Get prices for batch
                for address in batch:
                    try:
                        # Get price data
                        price_data = await self.get_token_price(address, chain)
                        if price_data and address.lower() in analytics:
                            analytics[address.lower()]['price'] = price_data
                        
                        # Get token stats
                        stats_data = await self.get_token_stats(address, chain)
                        if stats_data and address.lower() in analytics:
                            analytics[address.lower()]['stats'] = stats_data
                            
                    except Exception as token_error:
                        logger.warning(f"Failed to get analytics for {address}: {token_error}")
                        continue
                
                # Rate limiting between batches
                await asyncio.sleep(0.5)
            
            return analytics
            
        except Exception as e:
            logger.error(f"getMultipleTokenAnalytics failed: {e}")
            return {}

    # Cache statistics
    print("\n📊 Cache Statistics...")
    cache_stats = provider.get_cache_stats()
    print(f"  Total Entries: {cache_stats['total_entries']}")
    print(f"  Fresh Entries: {cache_stats['fresh_entries']}")
    print(f"  Cache Hit Ratio: {cache_stats['cache_hit_ratio']:.1%}")
    
    print("\n✅ Moralis provider test completed!")


if __name__ == "__main__":
    asyncio.run(test_moralis_provider())