"""
High Volatility Low Market Cap Trading Strategy
Specialized strategy for small fund allocation targeting tokens with:
- Low market cap (< $50M)
- Decent liquidity (> $100K daily volume)
- High volatility potential
- Proper risk management for small allocations
"""

import asyncio
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import requests
# from web3 import Web3  # Optional - for future blockchain integration

from ..data.trade_tracking import TradeTrackingSystem, TradeCandidate, ResearchDocument, TerminationReason

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LowCapCandidate:
    """Low market cap token candidate"""
    symbol: str
    contract_address: str
    market_cap: float
    daily_volume: float
    liquidity_score: float
    volatility_score: float
    rugpull_risk_score: float
    holder_count: int
    token_age_days: int
    social_momentum: float
    dev_activity_score: float
    audit_status: str

@dataclass
class PositionSizing:
    """Position sizing for high volatility low cap trades"""
    base_allocation_pct: float = 0.5  # % of small fund allocation
    max_allocation_pct: float = 2.0   # Maximum % for high confidence trades
    stop_loss_pct: float = 15.0       # Stop loss percentage
    take_profit_pct: float = 30.0     # Take profit percentage
    max_time_in_market_hours: int = 48  # Maximum holding time
    volatility_multiplier: float = 1.2  # Increase allocation for higher volatility

class HighVolatilityLowCapStrategy:
    """Strategy targeting high volatility low market cap tokens"""
    
    def __init__(self, 
                 small_fund_usd: float = 1000.0,  # Small allocation for high risk trades
                 coinmarketcap_api_key: str = None,
                 coingecko_api_key: str = None,
                 dappradar_api_key: str = None,
                 min_market_cap: float = 100000.0,   # $100K minimum
                 max_market_cap: float = 50000000.0, # $50M maximum
                 min_daily_volume: float = 100000.0,  # $100K minimum daily volume
                 min_liquidity_usd: float = 50000.0): # $50K minimum liquidity
        
        self.small_fund_usd = small_fund_usd
        self.coinmarketcap_api_key = coinmarketcap_api_key
        self.coingecko_api_key = coingecko_api_key
        self.dappradar_api_key = dappradar_api_key
        
        # Filtering criteria
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_daily_volume = min_daily_volume
        self.min_liquidity_usd = min_liquidity_usd
        
        # Initialize systems
        self.trade_tracker = TradeTrackingSystem(
            db_path="low_cap_trades.db",
            artifacts_dir="low_cap_artifacts"
        )
        
        self.position_sizing = PositionSizing()
        
        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}
        
        logger.info(f"Initialized High Volatility Low Cap Strategy with ${small_fund_usd} allocation")
    
    async def scan_low_cap_opportunities(self) -> List[LowCapCandidate]:
        """Scan for low market cap trading opportunities"""
        candidates = []
        
        try:
            # Get low cap tokens from CoinGecko
            coingecko_candidates = await self._get_coingecko_low_caps()
            candidates.extend(coingecko_candidates)
            
            # Get trending tokens from CoinMarketCap
            cmc_candidates = await self._get_cmc_trending_low_caps()
            candidates.extend(cmc_candidates)
            
            # Get DeFi tokens from DappRadar
            defi_candidates = await self._get_dappradar_defi_tokens()
            candidates.extend(defi_candidates)
            
        except Exception as e:
            logger.error(f"Error scanning opportunities: {e}")
        
        # Remove duplicates and filter
        unique_candidates = self._deduplicate_candidates(candidates)
        filtered_candidates = self._filter_candidates(unique_candidates)
        
        logger.info(f"Found {len(filtered_candidates)} low cap candidates")
        return filtered_candidates
    
    async def _get_coingecko_low_caps(self) -> List[LowCapCandidate]:
        """Get low market cap tokens from CoinGecko"""
        candidates = []
        
        if not self.coingecko_api_key:
            return candidates
        
        try:
            # Get trending coins
            url = "https://api.coingecko.com/api/v3/search/trending"
            headers = {"x-cg-demo-api-key": self.coingecko_api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for coin in data.get('coins', [])[:20]:  # Top 20 trending
                    coin_id = coin['item']['id']
                    
                    # Get detailed coin data
                    coin_data = await self._get_coingecko_coin_details(coin_id)
                    if coin_data:
                        candidates.append(coin_data)
                        
        except Exception as e:
            logger.error(f"Error fetching CoinGecko data: {e}")
        
        return candidates
    
    async def _get_coingecko_coin_details(self, coin_id: str) -> Optional[LowCapCandidate]:
        """Get detailed coin information from CoinGecko"""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            headers = {"x-cg-demo-api-key": self.coingecko_api_key}
            params = {
                "localization": "false",
                "tickers": "true",
                "market_data": "true",
                "community_data": "true",
                "developer_data": "true"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            data = response.json()
            
            # Extract key metrics
            market_data = data.get('market_data', {})
            market_cap = market_data.get('market_cap', {}).get('usd', 0)
            daily_volume = market_data.get('total_volume', {}).get('usd', 0)
            
            # Skip if doesn't meet criteria
            if not self.min_market_cap <= market_cap <= self.max_market_cap:
                return None
            if daily_volume < self.min_daily_volume:
                return None
            
            # Calculate scores
            liquidity_score = min(daily_volume / self.min_daily_volume, 10.0)
            volatility_score = self._calculate_volatility_score(market_data)
            rugpull_risk = self._calculate_rugpull_risk(data)
            social_momentum = self._calculate_social_momentum(data)
            dev_activity = self._calculate_dev_activity(data)
            
            return LowCapCandidate(
                symbol=data.get('symbol', '').upper(),
                contract_address=data.get('contract_address', ''),
                market_cap=market_cap,
                daily_volume=daily_volume,
                liquidity_score=liquidity_score,
                volatility_score=volatility_score,
                rugpull_risk_score=rugpull_risk,
                holder_count=0,  # Not available in this API
                token_age_days=self._calculate_token_age(data),
                social_momentum=social_momentum,
                dev_activity_score=dev_activity,
                audit_status="unknown"
            )
            
        except Exception as e:
            logger.error(f"Error getting coin details for {coin_id}: {e}")
            return None
    
    async def _get_cmc_trending_low_caps(self) -> List[LowCapCandidate]:
        """Get trending low market cap tokens from CoinMarketCap"""
        candidates = []
        
        if not self.coinmarketcap_api_key:
            return candidates
        
        try:
            # Get trending tokens
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/trending/latest"
            headers = {"X-CMC_PRO_API_KEY": self.coinmarketcap_api_key}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                for item in data.get('data', [])[:15]:  # Top 15 trending
                    coin_data = item.get('quote', {}).get('USD', {})
                    market_cap = coin_data.get('market_cap', 0)
                    volume_24h = coin_data.get('volume_24h', 0)
                    
                    # Check if meets criteria
                    if (self.min_market_cap <= market_cap <= self.max_market_cap and 
                        volume_24h >= self.min_daily_volume):
                        
                        candidates.append(LowCapCandidate(
                            symbol=item.get('symbol', ''),
                            contract_address="",  # Not provided by CMC
                            market_cap=market_cap,
                            daily_volume=volume_24h,
                            liquidity_score=min(volume_24h / self.min_daily_volume, 10.0),
                            volatility_score=abs(coin_data.get('percent_change_24h', 0)) / 10.0,
                            rugpull_risk_score=0.5,  # Default neutral score
                            holder_count=0,
                            token_age_days=0,
                            social_momentum=0.5,
                            dev_activity_score=0.5,
                            audit_status="unknown"
                        ))
                        
        except Exception as e:
            logger.error(f"Error fetching CMC trending data: {e}")
        
        return candidates
    
    async def _get_dappradar_defi_tokens(self) -> List[LowCapCandidate]:
        """Get DeFi tokens from DappRadar with fallback data"""
        candidates = []
        
        # Fallback DeFi tokens for demonstration
        fallback_tokens = [
            {
                "symbol": "SUSHI",
                "contract_address": "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
                "market_cap": 45000000,
                "daily_volume": 2500000,
                "category": "dex"
            },
            {
                "symbol": "UNI", 
                "contract_address": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
                "market_cap": 6800000000,
                "daily_volume": 85000000,
                "category": "dex"
            },
            {
                "symbol": "COMP",
                "contract_address": "0xc00e94Cb662C3520282E6f5717214004A7f26888", 
                "market_cap": 890000000,
                "daily_volume": 12000000,
                "category": "lending"
            }
        ]
        
        for token_data in fallback_tokens:
            market_cap = token_data["market_cap"]
            daily_volume = token_data["daily_volume"]
            
            # Check if meets low cap criteria
            if (self.min_market_cap <= market_cap <= self.max_market_cap and 
                daily_volume >= self.min_daily_volume):
                
                candidates.append(LowCapCandidate(
                    symbol=token_data["symbol"],
                    contract_address=token_data["contract_address"],
                    market_cap=market_cap,
                    daily_volume=daily_volume,
                    liquidity_score=min(daily_volume / self.min_daily_volume, 10.0),
                    volatility_score=6.0,  # DeFi tokens tend to be volatile
                    rugpull_risk_score=0.3,  # Lower risk for established DeFi
                    holder_count=10000,
                    token_age_days=365,
                    social_momentum=0.7,
                    dev_activity_score=0.8,
                    audit_status="audited"
                ))
        
        return candidates
    
    def _calculate_volatility_score(self, market_data: Dict) -> float:
        """Calculate volatility score from market data"""
        try:
            # Use price change percentages as volatility proxy
            changes = [
                abs(market_data.get('price_change_percentage_1h_in_currency', {}).get('usd', 0)),
                abs(market_data.get('price_change_percentage_24h_in_currency', {}).get('usd', 0)),
                abs(market_data.get('price_change_percentage_7d_in_currency', {}).get('usd', 0))
            ]
            
            avg_volatility = sum(changes) / len(changes)
            return min(avg_volatility / 10.0, 10.0)  # Normalize to 0-10 scale
            
        except Exception:
            return 5.0  # Default medium volatility
    
    def _calculate_rugpull_risk(self, coin_data: Dict) -> float:
        """Calculate rugpull risk score (0 = low risk, 1 = high risk)"""
        risk_score = 0.0
        
        try:
            # Check various risk factors
            
            # 1. Token age (newer = higher risk)
            age_days = self._calculate_token_age(coin_data)
            if age_days < 30:
                risk_score += 0.3
            elif age_days < 90:
                risk_score += 0.2
            elif age_days < 365:
                risk_score += 0.1
            
            # 2. Market cap (very low = higher risk)
            market_cap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
            if market_cap < 1000000:  # < $1M
                risk_score += 0.3
            elif market_cap < 5000000:  # < $5M
                risk_score += 0.2
            
            # 3. Liquidity relative to market cap
            volume = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
            if market_cap > 0:
                volume_to_mcap = volume / market_cap
                if volume_to_mcap < 0.01:  # Very low volume relative to mcap
                    risk_score += 0.2
            
            # 4. Developer activity
            if not coin_data.get('developer_data', {}).get('commits_count_4_weeks', 0):
                risk_score += 0.1
            
            # 5. Community data
            community = coin_data.get('community_data', {})
            if (community.get('twitter_followers', 0) < 1000 and 
                community.get('telegram_channel_user_count', 0) < 1000):
                risk_score += 0.1
            
        except Exception as e:
            logger.warning(f"Error calculating rugpull risk: {e}")
            return 0.5  # Default medium risk
        
        return min(risk_score, 1.0)
    
    def _calculate_social_momentum(self, coin_data: Dict) -> float:
        """Calculate social momentum score"""
        try:
            community = coin_data.get('community_data', {})
            
            # Factor in various social metrics
            twitter_followers = community.get('twitter_followers', 0)
            reddit_subscribers = community.get('reddit_subscribers', 0)
            telegram_users = community.get('telegram_channel_user_count', 0)
            
            # Normalize and combine
            social_score = 0.0
            if twitter_followers > 10000:
                social_score += 0.4
            elif twitter_followers > 1000:
                social_score += 0.2
            
            if reddit_subscribers > 5000:
                social_score += 0.3
            elif reddit_subscribers > 1000:
                social_score += 0.15
            
            if telegram_users > 5000:
                social_score += 0.3
            elif telegram_users > 1000:
                social_score += 0.15
            
            return min(social_score, 1.0)
            
        except Exception:
            return 0.5  # Default medium momentum
    
    def _calculate_dev_activity(self, coin_data: Dict) -> float:
        """Calculate developer activity score"""
        try:
            dev_data = coin_data.get('developer_data', {})
            
            commits_4w = dev_data.get('commits_count_4_weeks', 0)
            additions_4w = dev_data.get('code_additions_count_4_weeks', 0)
            deletions_4w = dev_data.get('code_deletions_count_4_weeks', 0)
            
            # Calculate activity score
            activity_score = 0.0
            if commits_4w > 50:
                activity_score += 0.5
            elif commits_4w > 10:
                activity_score += 0.3
            elif commits_4w > 0:
                activity_score += 0.1
            
            if additions_4w > 1000:
                activity_score += 0.3
            elif additions_4w > 100:
                activity_score += 0.2
            
            if dev_data.get('forks', 0) > 100:
                activity_score += 0.2
            
            return min(activity_score, 1.0)
            
        except Exception:
            return 0.5  # Default medium activity
    
    def _calculate_token_age(self, coin_data: Dict) -> int:
        """Calculate token age in days"""
        try:
            genesis_date = coin_data.get('genesis_date')
            if genesis_date:
                genesis = datetime.fromisoformat(genesis_date.replace('Z', '+00:00'))
                return (datetime.now(timezone.utc) - genesis).days
        except Exception:
            pass
        
        return 0  # Unknown age
    
    def _deduplicate_candidates(self, candidates: List[LowCapCandidate]) -> List[LowCapCandidate]:
        """Remove duplicate candidates based on symbol"""
        seen_symbols = set()
        unique_candidates = []
        
        for candidate in candidates:
            if candidate.symbol not in seen_symbols:
                seen_symbols.add(candidate.symbol)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _filter_candidates(self, candidates: List[LowCapCandidate]) -> List[LowCapCandidate]:
        """Apply filtering criteria to candidates"""
        filtered = []
        
        for candidate in candidates:
            # Basic criteria check
            if (self.min_market_cap <= candidate.market_cap <= self.max_market_cap and
                candidate.daily_volume >= self.min_daily_volume and
                candidate.rugpull_risk_score < 0.8 and  # Exclude very high risk
                candidate.liquidity_score >= 1.0):  # Minimum liquidity
                
                filtered.append(candidate)
        
        # Sort by opportunity score (combination of volatility and low risk)
        filtered.sort(key=lambda x: (x.volatility_score * (1 - x.rugpull_risk_score)), reverse=True)
        
        return filtered[:10]  # Top 10 opportunities
    
    def evaluate_trade_candidate(self, candidate: LowCapCandidate) -> Tuple[bool, float, str]:
        """Evaluate if candidate should be traded"""
        # Calculate opportunity score
        opportunity_score = (
            candidate.volatility_score * 0.3 +
            candidate.liquidity_score * 0.25 +
            candidate.social_momentum * 0.2 +
            candidate.dev_activity_score * 0.15 +
            (1 - candidate.rugpull_risk_score) * 0.1
        )
        
        # Trade rationale
        rationale_parts = []
        
        if candidate.volatility_score > 6.0:
            rationale_parts.append(f"High volatility ({candidate.volatility_score:.1f})")
        
        if candidate.liquidity_score > 5.0:
            rationale_parts.append(f"Good liquidity ({candidate.liquidity_score:.1f})")
        
        if candidate.social_momentum > 0.6:
            rationale_parts.append(f"Strong social momentum ({candidate.social_momentum:.1f})")
        
        if candidate.rugpull_risk_score < 0.4:
            rationale_parts.append(f"Low rugpull risk ({candidate.rugpull_risk_score:.1f})")
        
        if candidate.audit_status == "audited":
            rationale_parts.append("Audited contract")
        
        rationale = "; ".join(rationale_parts) if rationale_parts else "Low score candidate"
        
        # Decision threshold
        should_trade = opportunity_score > 5.0 and candidate.rugpull_risk_score < 0.7
        
        return should_trade, opportunity_score, rationale
    
    def calculate_position_size(self, candidate: LowCapCandidate, confidence: float) -> float:
        """Calculate position size based on candidate and confidence"""
        # Base allocation
        base_size = self.small_fund_usd * (self.position_sizing.base_allocation_pct / 100.0)
        
        # Adjust for confidence
        confidence_multiplier = min(confidence / 0.5, 2.0)  # Cap at 2x
        
        # Adjust for volatility (higher volatility = larger position for momentum)
        volatility_multiplier = min(candidate.volatility_score / 5.0, self.position_sizing.volatility_multiplier)
        
        # Adjust for liquidity (higher liquidity = can go larger)
        liquidity_multiplier = min(candidate.liquidity_score / 3.0, 1.5)
        
        # Reduce for rugpull risk
        risk_reduction = 1.0 - (candidate.rugpull_risk_score * 0.5)
        
        # Calculate final size
        position_size = (base_size * confidence_multiplier * volatility_multiplier * 
                        liquidity_multiplier * risk_reduction)
        
        # Apply maximum allocation limit
        max_size = self.small_fund_usd * (self.position_sizing.max_allocation_pct / 100.0)
        position_size = min(position_size, max_size)
        
        return max(position_size, base_size * 0.5)  # Minimum 50% of base allocation
    
    async def create_trade_candidate(self, low_cap_candidate: LowCapCandidate) -> Optional[TradeCandidate]:
        """Create trade candidate with full research documentation"""
        
        # Evaluate the candidate
        should_trade, confidence, rationale = self.evaluate_trade_candidate(low_cap_candidate)
        
        if not should_trade:
            logger.info(f"Skipping {low_cap_candidate.symbol}: {rationale}")
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(low_cap_candidate, confidence)
        
        # Create parameter snapshot
        params_snapshot = {
            "strategy": "high_volatility_low_cap",
            "stop_loss_pct": self.position_sizing.stop_loss_pct,
            "take_profit_pct": self.position_sizing.take_profit_pct,
            "max_time_hours": self.position_sizing.max_time_in_market_hours,
            "position_size_usd": position_size,
            "small_fund_total": self.small_fund_usd,
            "confidence_threshold": 5.0,
            "rugpull_risk_limit": 0.7
        }
        
        # Store parameter snapshot as artifact
        params_json = json.dumps(params_snapshot, indent=2).encode('utf-8')
        params_ptr = self.trade_tracker.store_artifact(
            data=params_json,
            artifact_type="parameter_snapshot",
            file_extension=".json"
        )
        
        # Create trade candidate
        trade_candidate = self.trade_tracker.create_candidate(
            instrument=f"{low_cap_candidate.symbol}/USDT",
            model_version="high_vol_low_cap_v1.0",
            confidence=confidence,
            params_ptr=params_ptr,
            created_by="high_volatility_low_cap_strategy",
            contract_address=low_cap_candidate.contract_address
        )
        
        # Create comprehensive research document
        research_doc = ResearchDocument(
            uuid=trade_candidate.uuid,
            timestamp=datetime.now(timezone.utc),
            instrument=f"{low_cap_candidate.symbol}/USDT",
            contract_address=low_cap_candidate.contract_address,
            creator_wallets=[],  # Would be populated from blockchain analysis
            initial_market_cap=low_cap_candidate.market_cap,
            total_supply=0.0,  # Would be populated from contract
            liquidity_pools={},  # Would be populated from DEX analysis
            liquidity_provider_addresses=[],
            holder_count=low_cap_candidate.holder_count,
            on_chain_age_days=low_cap_candidate.token_age_days,
            audit_status=low_cap_candidate.audit_status,
            verified_source_flag=low_cap_candidate.audit_status == "audited",
            social_snippets_ptr="",  # Would store actual social media posts
            news_snippets_ptr="",    # Would store actual news articles
            social_sentiment_scores={"overall": low_cap_candidate.social_momentum},
            news_sentiment_scores={"overall": 0.5},
            whale_snapshot_ptr="",   # Would store whale wallet analysis
            rugpull_heuristic_scores={
                "overall_risk": low_cap_candidate.rugpull_risk_score,
                "liquidity_risk": max(0, 0.8 - low_cap_candidate.liquidity_score / 10.0),
                "dev_risk": max(0, 0.8 - low_cap_candidate.dev_activity_score),
                "age_risk": max(0, 0.8 - min(low_cap_candidate.token_age_days / 365.0, 1.0))
            },
            trade_rationale=rationale,
            model_feature_vector=[
                low_cap_candidate.market_cap,
                low_cap_candidate.daily_volume,
                low_cap_candidate.liquidity_score,
                low_cap_candidate.volatility_score,
                low_cap_candidate.rugpull_risk_score,
                low_cap_candidate.social_momentum,
                low_cap_candidate.dev_activity_score,
                confidence
            ],
            parameter_snapshot=params_snapshot
        )
        
        # Store research document
        self.trade_tracker.store_research_document(research_doc)
        
        logger.info(f"Created trade candidate {trade_candidate.uuid} for {low_cap_candidate.symbol}")
        logger.info(f"Position size: ${position_size:.2f} (confidence: {confidence:.2f})")
        logger.info(f"Rationale: {rationale}")
        
        return trade_candidate
    
    async def run_strategy_cycle(self):
        """Run one complete strategy cycle"""
        logger.info("Starting high volatility low cap strategy cycle")
        
        # 1. Scan for opportunities
        candidates = await self.scan_low_cap_opportunities()
        
        if not candidates:
            logger.info("No suitable candidates found")
            return
        
        # 2. Evaluate and create trade candidates
        trade_candidates = []
        for candidate in candidates:
            trade_candidate = await self.create_trade_candidate(candidate)
            if trade_candidate:
                trade_candidates.append(trade_candidate)
        
        logger.info(f"Created {len(trade_candidates)} trade candidates")
        
        # 3. Check fund allocation
        total_allocated = sum(
            json.loads(tc.params_ptr.replace('parameter_snapshot', ''))
            .get('position_size_usd', 0) 
            for tc in trade_candidates
        )
        
        logger.info(f"Total allocation requested: ${total_allocated:.2f} / ${self.small_fund_usd:.2f}")
        
        return trade_candidates

# Example usage and testing
async def main():
    """Test the high volatility low cap strategy"""
    
    # Initialize strategy with small fund
    strategy = HighVolatilityLowCapStrategy(
        small_fund_usd=1000.0,
        coinmarketcap_api_key="6cad35f36d7b4e069b8dcb0eb9d17d56",
        coingecko_api_key="CG-uKph8trS6RiycsxwVQtxfxvF",
        dappradar_api_key="xD9Fvb0Nb285BLRPfKLgL44ULe6nR8Fm90i894xA"
    )
    
    # Run strategy cycle
    trade_candidates = await strategy.run_strategy_cycle()
    
    if trade_candidates:
        print(f"\n✅ Generated {len(trade_candidates)} trade candidates:")
        for tc in trade_candidates:
            print(f"  - {tc.instrument}: Confidence {tc.confidence:.2f}")

if __name__ == "__main__":
    asyncio.run(main())