"""
Enhanced Blockchain Analysis Integration
Combines Moralis on-chain data with trading signals for comprehensive token analysis
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .moralis_provider import MoralisProvider, TokenMetrics, WalletAnalysis, DexTradeAnalysis

logger = logging.getLogger(__name__)

@dataclass
class TokenAnalysisResult:
    """Comprehensive token analysis result"""
    symbol: str
    contract_address: str
    chain: str
    
    # Basic metrics
    price_usd: float
    market_cap: float
    volume_24h: float
    holder_count: int
    liquidity_usd: float
    
    # Risk assessment
    rugpull_risk_score: float
    concentration_risk: str
    liquidity_risk: str
    overall_risk_level: str
    
    # Trading signals
    whale_activity: bool
    volume_trend: str
    price_momentum: str
    sentiment_score: float
    
    # Recommendation
    trading_recommendation: str
    confidence_score: float
    risk_factors: List[str]
    
    timestamp: datetime

@dataclass 
class WhaleMovement:
    """Whale movement detection"""
    wallet_address: str
    transaction_hash: str
    token_symbol: str
    amount: float
    usd_value: float
    transaction_type: str  # 'buy', 'sell', 'transfer'
    timestamp: datetime
    impact_score: float

class EnhancedBlockchainAnalyzer:
    """
    Enhanced blockchain analyzer using Moralis for comprehensive token analysis
    """
    
    def __init__(self, moralis_api_key: str, config: Dict = None):
        self.moralis = MoralisProvider(moralis_api_key)
        self.config = config or {}
        
        # Analysis thresholds
        self.whale_threshold = 100000  # $100K USD
        self.liquidity_threshold = 50000  # $50K USD
        self.holder_threshold = 1000  # Min holders for low risk
        
        # Risk scoring weights
        self.risk_weights = {
            'rugpull_signals': 0.4,
            'concentration': 0.2,
            'liquidity': 0.2,
            'age': 0.1,
            'verification': 0.1
        }
        
        logger.info("🔗 Enhanced Blockchain Analyzer initialized with Moralis")
    
    async def analyze_token_comprehensive(self, contract_address: str, symbol: str, chain: str = 'ethereum') -> TokenAnalysisResult:
        """
        Perform comprehensive token analysis combining all data sources
        """
        logger.info(f"🔍 Starting comprehensive analysis for {symbol} ({contract_address})")
        
        try:
            # Get comprehensive Moralis analysis
            moralis_analysis = await self.moralis.get_comprehensive_token_analysis(contract_address, chain)
            
            # Extract key metrics
            token_metadata = moralis_analysis.get('token_metadata')
            rugpull_signals = moralis_analysis.get('rugpull_signals', {})
            dex_analysis = moralis_analysis.get('dex_analysis')
            holder_concentration = moralis_analysis.get('holder_concentration', {})
            risk_assessment = moralis_analysis.get('risk_assessment', {})
            
            # Calculate trading signals
            trading_signals = await self._calculate_trading_signals(moralis_analysis)
            
            # Create comprehensive result
            result = TokenAnalysisResult(
                symbol=symbol,
                contract_address=contract_address,
                chain=chain,
                
                # Basic metrics
                price_usd=token_metadata.price_usd if token_metadata else 0.0,
                market_cap=token_metadata.market_cap if token_metadata else 0.0,
                volume_24h=dex_analysis.volume_24h if dex_analysis else 0.0,
                holder_count=token_metadata.holder_count if token_metadata else 0,
                liquidity_usd=token_metadata.liquidity_usd if token_metadata else 0.0,
                
                # Risk assessment
                rugpull_risk_score=rugpull_signals.get('overall_risk', 0.5),
                concentration_risk=holder_concentration.get('concentration_risk', 'unknown'),
                liquidity_risk=self._assess_liquidity_risk(token_metadata.liquidity_usd if token_metadata else 0),
                overall_risk_level=risk_assessment.get('risk_level', 'unknown'),
                
                # Trading signals
                whale_activity=dex_analysis.whale_activity if dex_analysis else False,
                volume_trend=trading_signals['volume_trend'],
                price_momentum=trading_signals['price_momentum'],
                sentiment_score=trading_signals['sentiment_score'],
                
                # Recommendation
                trading_recommendation=risk_assessment.get('recommendation', 'caution'),
                confidence_score=trading_signals['confidence_score'],
                risk_factors=risk_assessment.get('risk_factors', []),
                
                timestamp=datetime.now()
            )
            
            logger.info(f"✅ Analysis completed for {symbol} - Risk: {result.overall_risk_level}, Recommendation: {result.trading_recommendation}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in comprehensive token analysis: {e}")
            # Return default result with high risk
            return self._create_default_result(symbol, contract_address, chain)
    
    async def detect_whale_movements(self, token_address: str, chain: str = 'ethereum', hours: int = 24) -> List[WhaleMovement]:
        """
        Detect significant whale movements for a token
        """
        logger.info(f"🐋 Detecting whale movements for {token_address} (last {hours}h)")
        
        try:
            # Get recent transfers
            transfers = await self.moralis._make_request(
                f"/erc20/{token_address}/transfers",
                {"chain": self.moralis.chains.get(chain, '0x1'), "limit": 500}
            )
            
            if not transfers:
                return []
            
            whale_movements = []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            for transfer in transfers.get('result', []):
                # Parse transfer data
                timestamp_str = transfer.get('block_timestamp', '')
                if not timestamp_str:
                    continue
                
                transfer_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if transfer_time < cutoff_time:
                    continue
                
                # Calculate USD value
                value_raw = float(transfer.get('value', 0))
                decimals = int(transfer.get('decimals', 18))
                value_formatted = value_raw / (10 ** decimals)
                
                # Get current token price for USD calculation
                token_data = await self.moralis.get_token_price(token_address, chain)
                price_usd = float(token_data.get('usd_price', 0)) if token_data else 0
                
                usd_value = value_formatted * price_usd
                
                # Filter for whale-sized movements
                if usd_value >= self.whale_threshold:
                    # Determine transaction type
                    from_address = transfer.get('from_address', '').lower()
                    to_address = transfer.get('to_address', '').lower()
                    
                    # Check if it's a DEX transaction (simplified)
                    dex_addresses = ['0x7a250d5630b4cf539739df2c5dacb4c659f2488d']  # Uniswap V2 Router
                    is_dex_tx = any(addr in [from_address, to_address] for addr in dex_addresses)
                    
                    if is_dex_tx:
                        tx_type = 'sell' if from_address not in dex_addresses else 'buy'
                    else:
                        tx_type = 'transfer'
                    
                    # Calculate impact score
                    impact_score = min(1.0, usd_value / 1000000)  # Scale to $1M max
                    
                    whale_movement = WhaleMovement(
                        wallet_address=from_address if tx_type == 'sell' else to_address,
                        transaction_hash=transfer.get('transaction_hash', ''),
                        token_symbol=transfer.get('symbol', 'UNKNOWN'),
                        amount=value_formatted,
                        usd_value=usd_value,
                        transaction_type=tx_type,
                        timestamp=transfer_time,
                        impact_score=impact_score
                    )
                    
                    whale_movements.append(whale_movement)
            
            # Sort by USD value descending
            whale_movements.sort(key=lambda x: x.usd_value, reverse=True)
            
            logger.info(f"🐋 Found {len(whale_movements)} whale movements")
            return whale_movements[:20]  # Return top 20
            
        except Exception as e:
            logger.error(f"❌ Error detecting whale movements: {e}")
            return []
    
    async def analyze_liquidity_pools(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """
        Analyze liquidity pools for a token
        """
        logger.info(f"💧 Analyzing liquidity pools for {token_address}")
        
        try:
            # This would require more complex DEX analysis
            # For now, return basic liquidity information from token stats
            token_stats = await self.moralis.get_token_stats(token_address, chain)
            
            if token_stats:
                return {
                    'total_liquidity_usd': token_stats.get('total_liquidity_usd', 0),
                    'pool_count': token_stats.get('pool_count', 0),
                    'major_pools': [],  # Would need DEX-specific analysis
                    'liquidity_distribution': {},
                    'impermanent_loss_risk': 'medium'  # Default
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error analyzing liquidity pools: {e}")
            return {}
    
    async def check_token_security(self, contract_address: str, chain: str = 'ethereum') -> Dict:
        """
        Check token security features and potential risks
        """
        logger.info(f"🔒 Checking security for {contract_address}")
        
        try:
            # Get rugpull signals from Moralis
            rugpull_signals = await self.moralis.detect_rugpull_signals(contract_address, chain)
            
            # Get token metadata for verification status
            token_metadata = await self.moralis.get_token_metadata(contract_address, chain)
            
            # Get top holders for concentration analysis
            top_holders = await self.moralis.get_top_holders(contract_address, chain)
            
            security_analysis = {
                'contract_verified': token_metadata.verified if token_metadata else False,
                'rugpull_risk_score': rugpull_signals.get('overall_risk', 0.5),
                'holder_concentration': self._analyze_holder_concentration(top_holders),
                'liquidity_locked': False,  # Would need additional contract analysis
                'mint_function': 'unknown',  # Would need contract source analysis
                'pause_function': 'unknown',
                'security_score': 0.5,  # Calculated below
                'risk_factors': []
            }
            
            # Calculate overall security score
            score = 1.0
            risk_factors = []
            
            if not security_analysis['contract_verified']:
                score -= 0.3
                risk_factors.append('Unverified contract')
            
            if rugpull_signals.get('overall_risk', 0) > 0.7:
                score -= 0.4
                risk_factors.append('High rugpull risk')
            
            concentration = security_analysis['holder_concentration']
            if concentration['top_5_percentage'] > 60:
                score -= 0.2
                risk_factors.append('Very high holder concentration')
            elif concentration['top_5_percentage'] > 40:
                score -= 0.1
                risk_factors.append('High holder concentration')
            
            security_analysis['security_score'] = max(0.0, score)
            security_analysis['risk_factors'] = risk_factors
            
            return security_analysis
            
        except Exception as e:
            logger.error(f"❌ Error checking token security: {e}")
            return {'security_score': 0.0, 'risk_factors': ['Analysis failed']}
    
    def _analyze_holder_concentration(self, holders: List[Dict]) -> Dict:
        """Analyze holder concentration from top holders data"""
        if not holders:
            return {'top_5_percentage': 0, 'top_10_percentage': 0, 'concentration_risk': 'unknown'}
        
        top_5_percentage = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:5])
        top_10_percentage = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:10])
        
        if top_5_percentage > 60:
            risk_level = 'very_high'
        elif top_5_percentage > 40:
            risk_level = 'high'
        elif top_5_percentage > 25:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'top_5_percentage': top_5_percentage,
            'top_10_percentage': top_10_percentage,
            'concentration_risk': risk_level,
            'largest_holder_percentage': float(holders[0].get('percentage_relative_to_total_supply', 0)) if holders else 0
        }
    
    async def _calculate_trading_signals(self, moralis_analysis: Dict) -> Dict:
        """Calculate trading signals from Moralis analysis"""
        signals = {
            'volume_trend': 'neutral',
            'price_momentum': 'neutral', 
            'sentiment_score': 0.5,
            'confidence_score': 0.5
        }
        
        try:
            dex_analysis = moralis_analysis.get('dex_analysis')
            
            if dex_analysis:
                # Volume trend analysis
                if dex_analysis.volume_24h > 100000:
                    signals['volume_trend'] = 'increasing'
                elif dex_analysis.volume_24h < 10000:
                    signals['volume_trend'] = 'decreasing'
                
                # Whale activity affects sentiment
                if dex_analysis.whale_activity:
                    signals['sentiment_score'] *= 1.2  # Boost sentiment
                
                # Trade frequency
                if dex_analysis.trades_24h > 100:
                    signals['confidence_score'] += 0.2
            
            # Risk assessment affects confidence
            risk_assessment = moralis_analysis.get('risk_assessment', {})
            risk_level = risk_assessment.get('risk_level', 'medium')
            
            if risk_level == 'low':
                signals['confidence_score'] += 0.3
            elif risk_level == 'high':
                signals['confidence_score'] -= 0.3
            
            # Normalize scores
            signals['sentiment_score'] = min(1.0, max(0.0, signals['sentiment_score']))
            signals['confidence_score'] = min(1.0, max(0.0, signals['confidence_score']))
            
        except Exception as e:
            logger.error(f"Error calculating trading signals: {e}")
        
        return signals
    
    def _assess_liquidity_risk(self, liquidity_usd: float) -> str:
        """Assess liquidity risk level"""
        if liquidity_usd >= 500000:
            return 'low'
        elif liquidity_usd >= 100000:
            return 'medium'
        elif liquidity_usd >= 25000:
            return 'high'
        else:
            return 'very_high'
    
    def _create_default_result(self, symbol: str, contract_address: str, chain: str) -> TokenAnalysisResult:
        """Create default result when analysis fails"""
        return TokenAnalysisResult(
            symbol=symbol,
            contract_address=contract_address,
            chain=chain,
            price_usd=0.0,
            market_cap=0.0,
            volume_24h=0.0,
            holder_count=0,
            liquidity_usd=0.0,
            rugpull_risk_score=0.8,  # High risk when analysis fails
            concentration_risk='unknown',
            liquidity_risk='very_high',
            overall_risk_level='high',
            whale_activity=False,
            volume_trend='unknown',
            price_momentum='unknown',
            sentiment_score=0.3,
            trading_recommendation='avoid',
            confidence_score=0.1,
            risk_factors=['Analysis failed'],
            timestamp=datetime.now()
        )
    
    async def get_trending_tokens(self, chain: str = 'ethereum', min_volume: float = 50000) -> List[Dict]:
        """
        Get trending tokens with basic analysis
        """
        logger.info(f"📈 Finding trending tokens on {chain}")
        
        try:
            # This would typically require a trending tokens endpoint
            # For now, return empty list as Moralis doesn't have a direct trending endpoint
            # In practice, you'd combine this with other data sources
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting trending tokens: {e}")
            return []


# Example usage and testing
async def test_enhanced_blockchain_analyzer():
    """Test the enhanced blockchain analyzer"""
    # Load API key from config
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['moralis']
    except:
        print("❌ Please add your Moralis API key to config.json")
        return
    
    analyzer = EnhancedBlockchainAnalyzer(api_key)
    
    print("🧪 Testing Enhanced Blockchain Analyzer...")
    
    # Test comprehensive token analysis
    print("\n1️⃣ Testing Comprehensive Token Analysis...")
    test_token = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"  # UNI token
    analysis = await analyzer.analyze_token_comprehensive(test_token, "UNI", "ethereum")
    
    print(f"  Token: {analysis.symbol}")
    print(f"  Price: ${analysis.price_usd:.4f}")
    print(f"  Market Cap: ${analysis.market_cap:,.0f}")
    print(f"  Risk Level: {analysis.overall_risk_level}")
    print(f"  Recommendation: {analysis.trading_recommendation}")
    print(f"  Confidence: {analysis.confidence_score:.2f}")
    
    # Test whale movement detection
    print("\n2️⃣ Testing Whale Movement Detection...")
    whale_movements = await analyzer.detect_whale_movements(test_token, "ethereum", 24)
    
    print(f"  Found {len(whale_movements)} whale movements")
    for i, movement in enumerate(whale_movements[:3]):
        print(f"    {i+1}. {movement.transaction_type.upper()}: ${movement.usd_value:,.0f} ({movement.amount:,.0f} tokens)")
    
    # Test security analysis
    print("\n3️⃣ Testing Security Analysis...")
    security = await analyzer.check_token_security(test_token, "ethereum")
    
    print(f"  Security Score: {security['security_score']:.2f}")
    print(f"  Contract Verified: {security['contract_verified']}")
    if security['risk_factors']:
        print(f"  Risk Factors: {', '.join(security['risk_factors'])}")
    
    print("\n✅ Enhanced blockchain analyzer test completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_blockchain_analyzer())