"""
Top Holders Analysis - Critical for Rugpull Detection
Checks token holder concentration to identify dangerous tokens
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TokenHolderAnalyzer:
    """
    Analyzes token holder distribution to detect concentration risks
    """
    
    def __init__(self):
        # Critical thresholds for holder concentration
        self.danger_thresholds = {
            'top_1_holder': 40,      # Top 1 holder should not own >40%
            'top_5_holders': 60,     # Top 5 holders should not own >60%
            'top_10_holders': 75,    # Top 10 holders should not own >75%
            'min_holders': 100       # Minimum 100 unique holders
        }
        
        # Known safe addresses (exchanges, burn addresses, etc.)
        self.safe_addresses = {
            '0x000000000000000000000000000000000000dead',  # Burn address
            '0x0000000000000000000000000000000000000000',  # Null address
            '0x7a250d5630b4cf539739df2c5dacb4c659f2488d',  # Uniswap V2 Router
            '0xe592427a0aece92de3edee1f18e0157c05861564',  # Uniswap V3 Router
            '0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45',  # Uniswap V3 Router 2
            '0xef1c6e67703c7bd7107eed8303fbe6ec2554bf6b',  # Uniswap Universal Router
        }
    
    async def _get_etherscan_data(self, url: str, params: Dict) -> Optional[Dict]:
        """Get data from Etherscan API"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('status') == '1':
                            return data
                    return None
        except Exception as e:
            logger.error(f"Etherscan request failed: {e}")
            return None
    
    async def get_top_holders_etherscan(self, contract_address: str, api_key: str = None) -> List[Dict]:
        """
        Get top holders using Etherscan API (if you have API key)
        """
        if not api_key:
            logger.warning("No Etherscan API key provided")
            return []
        
        url = "https://api.etherscan.io/api"
        params = {
            'module': 'token',
            'action': 'tokenholderlist',
            'contractaddress': contract_address,
            'page': 1,
            'offset': 100,  # Get top 100 holders
            'apikey': api_key
        }
        
        data = await self._get_etherscan_data(url, params)
        
        if data and data.get('result'):
            holders = []
            total_supply = 0
            
            # First, get total supply
            supply_params = {
                'module': 'stats',
                'action': 'tokensupply',
                'contractaddress': contract_address,
                'apikey': api_key
            }
            
            supply_data = await self._get_etherscan_data(url, supply_params)
            if supply_data and supply_data.get('result'):
                total_supply = int(supply_data['result'])
            
            # Process holders
            for holder in data['result']:
                balance = int(holder['TokenHolderQuantity'])
                percentage = (balance / total_supply * 100) if total_supply > 0 else 0
                
                holders.append({
                    'address': holder['TokenHolderAddress'],
                    'balance': balance,
                    'percentage': percentage,
                    'is_contract': False,  # Would need additional check
                    'is_safe_address': holder['TokenHolderAddress'].lower() in self.safe_addresses
                })
            
            return sorted(holders, key=lambda x: x['balance'], reverse=True)
        
        return []
    
    async def estimate_holders_from_dexscreener(self, contract_address: str) -> Dict:
        """
        Estimate holder distribution from DexScreener data (indirect method)
        """
        print("📊 Estimating holder distribution from trading patterns...")
        
        url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('pairs'):
                            best_pair = max(data['pairs'], key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
                            
                            # Analyze trading patterns to estimate concentration
                            market_cap = float(best_pair.get('marketCap', 0))
                            liquidity = float(best_pair.get('liquidity', {}).get('usd', 0))
                            volume_24h = float(best_pair.get('volume', {}).get('h24', 0))
                            
                            # Get transaction data
                            txns_24h = best_pair.get('txns', {}).get('h24', {})
                            buys = txns_24h.get('buys', 0)
                            sells = txns_24h.get('sells', 0)
                            total_txns = buys + sells
                            
                            # Estimate holder concentration based on trading patterns
                            concentration_risk = self._estimate_concentration_risk(
                                market_cap, liquidity, volume_24h, total_txns
                            )
                            
                            return {
                                'estimated_holders': self._estimate_holder_count(market_cap, total_txns),
                                'concentration_risk_level': concentration_risk,
                                'market_cap': market_cap,
                                'liquidity_ratio': (liquidity / market_cap * 100) if market_cap > 0 else 0,
                                'volume_to_mcap_ratio': (volume_24h / market_cap * 100) if market_cap > 0 else 0,
                                'daily_transactions': total_txns,
                                'buy_sell_ratio': buys / max(1, sells)
                            }
        
        except Exception as e:
            logger.error(f"DexScreener analysis failed: {e}")
        
        return {}
    
    def _estimate_concentration_risk(self, market_cap: float, liquidity: float, volume: float, transactions: int) -> str:
        """
        Estimate concentration risk based on trading patterns
        """
        risk_score = 0
        
        # Low liquidity relative to market cap suggests concentration
        if market_cap > 0:
            liquidity_ratio = liquidity / market_cap
            if liquidity_ratio < 0.02:  # <2% liquidity
                risk_score += 3
            elif liquidity_ratio < 0.05:  # <5% liquidity
                risk_score += 2
        
        # Low transaction count suggests few active traders
        if transactions < 20:
            risk_score += 3
        elif transactions < 50:
            risk_score += 2
        elif transactions < 100:
            risk_score += 1
        
        # High volume relative to liquidity suggests large holders trading
        if liquidity > 0:
            volume_ratio = volume / liquidity
            if volume_ratio > 2:  # Volume > 2x liquidity
                risk_score += 2
            elif volume_ratio > 1:
                risk_score += 1
        
        # Convert score to risk level
        if risk_score >= 6:
            return 'extreme'
        elif risk_score >= 4:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _estimate_holder_count(self, market_cap: float, transactions: int) -> int:
        """
        Rough estimation of holder count based on market dynamics
        """
        # Base estimate from market cap
        base_holders = min(10000, max(50, int(market_cap / 50000)))  # ~1 holder per $50K market cap
        
        # Adjust based on transaction activity
        activity_multiplier = min(2.0, max(0.5, transactions / 100))
        
        estimated = int(base_holders * activity_multiplier)
        return max(10, estimated)  # Minimum 10 holders
    
    async def check_known_rugpull_patterns(self, contract_address: str) -> Dict:
        """
        Check for known rugpull patterns in holder distribution
        """
        print("🔍 Checking for rugpull patterns...")
        
        # Get DexScreener data for pattern analysis
        dex_data = await self.estimate_holders_from_dexscreener(contract_address)
        
        patterns = {
            'rugpull_risk_score': 0,
            'detected_patterns': [],
            'risk_factors': []
        }
        
        if dex_data:
            risk_score = 0
            
            # Pattern 1: Very low holder estimate
            estimated_holders = dex_data.get('estimated_holders', 0)
            if estimated_holders < 50:
                risk_score += 40
                patterns['detected_patterns'].append('Very few estimated holders')
                patterns['risk_factors'].append(f'Estimated holders: {estimated_holders}')
            elif estimated_holders < 100:
                risk_score += 20
                patterns['risk_factors'].append(f'Low estimated holders: {estimated_holders}')
            
            # Pattern 2: High concentration risk
            concentration_risk = dex_data.get('concentration_risk_level', 'unknown')
            if concentration_risk == 'extreme':
                risk_score += 30
                patterns['detected_patterns'].append('Extreme concentration risk')
            elif concentration_risk == 'high':
                risk_score += 20
                patterns['detected_patterns'].append('High concentration risk')
            
            # Pattern 3: Low liquidity ratio
            liquidity_ratio = dex_data.get('liquidity_ratio', 0)
            if liquidity_ratio < 2:
                risk_score += 20
                patterns['detected_patterns'].append('Very low liquidity ratio')
                patterns['risk_factors'].append(f'Liquidity ratio: {liquidity_ratio:.1f}%')
            
            # Pattern 4: Imbalanced trading
            buy_sell_ratio = dex_data.get('buy_sell_ratio', 1)
            if buy_sell_ratio < 0.3:  # More than 3:1 sell ratio
                risk_score += 25
                patterns['detected_patterns'].append('Heavy selling pressure')
            elif buy_sell_ratio > 3:  # More than 3:1 buy ratio (pump?)
                risk_score += 15
                patterns['detected_patterns'].append('Unusual buying pressure')
            
            # Pattern 5: Low transaction count
            daily_txns = dex_data.get('daily_transactions', 0)
            if daily_txns < 20:
                risk_score += 15
                patterns['detected_patterns'].append('Very low trading activity')
            
            patterns['rugpull_risk_score'] = min(100, risk_score)
        
        return patterns
    
    async def comprehensive_holder_analysis(self, contract_address: str, etherscan_api_key: str = None) -> Dict:
        """
        Complete holder analysis combining all available methods
        """
        print(f"\n👥 COMPREHENSIVE HOLDER ANALYSIS")
        print(f"Contract: {contract_address}")
        print("=" * 50)
        
        analysis = {
            'contract_address': contract_address,
            'timestamp': datetime.now().isoformat(),
            'holder_safety_score': 0,
            'concentration_verdict': 'unknown',
            'top_holders': [],
            'estimated_distribution': {},
            'rugpull_patterns': {},
            'warnings': [],
            'critical_issues': []
        }
        
        try:
            # Method 1: Try to get actual holder data (if Etherscan API available)
            if etherscan_api_key:
                print("📋 Getting actual holder data from Etherscan...")
                top_holders = await self.get_top_holders_etherscan(contract_address, etherscan_api_key)
                analysis['top_holders'] = top_holders[:20]  # Top 20 holders
                
                if top_holders:
                    # Analyze actual concentration
                    top_1_pct = top_holders[0]['percentage'] if top_holders else 0
                    top_5_pct = sum(h['percentage'] for h in top_holders[:5])
                    top_10_pct = sum(h['percentage'] for h in top_holders[:10])
                    
                    print(f"   Top 1 holder: {top_1_pct:.1f}%")
                    print(f"   Top 5 holders: {top_5_pct:.1f}%")
                    print(f"   Top 10 holders: {top_10_pct:.1f}%")
                    
                    # Check against thresholds
                    concentration_issues = []
                    
                    if top_1_pct > self.danger_thresholds['top_1_holder']:
                        concentration_issues.append(f"Top holder owns {top_1_pct:.1f}% (danger: >{self.danger_thresholds['top_1_holder']}%)")
                    
                    if top_5_pct > self.danger_thresholds['top_5_holders']:
                        concentration_issues.append(f"Top 5 holders own {top_5_pct:.1f}% (danger: >{self.danger_thresholds['top_5_holders']}%)")
                    
                    if top_10_pct > self.danger_thresholds['top_10_holders']:
                        concentration_issues.append(f"Top 10 holders own {top_10_pct:.1f}% (danger: >{self.danger_thresholds['top_10_holders']}%)")
                    
                    # Calculate safety score based on actual data
                    safety_score = 100
                    if top_1_pct > 50:
                        safety_score -= 50
                        analysis['critical_issues'].append('Single holder owns majority')
                    elif top_1_pct > 30:
                        safety_score -= 30
                    
                    if top_5_pct > 80:
                        safety_score -= 40
                        analysis['critical_issues'].append('Extreme concentration in top 5')
                    elif top_5_pct > 60:
                        safety_score -= 20
                    
                    analysis['holder_safety_score'] = max(0, safety_score)
                    analysis['warnings'].extend(concentration_issues)
            
            # Method 2: Estimate from trading patterns
            print("📊 Estimating distribution from trading patterns...")
            estimated_data = await self.estimate_holders_from_dexscreener(contract_address)
            analysis['estimated_distribution'] = estimated_data
            
            # Method 3: Check rugpull patterns
            print("🔍 Checking rugpull patterns...")
            rugpull_analysis = await self.check_known_rugpull_patterns(contract_address)
            analysis['rugpull_patterns'] = rugpull_analysis
            
            # Overall verdict
            if analysis['holder_safety_score'] == 0:  # No actual data
                # Use estimated risk
                estimated_risk = rugpull_analysis.get('rugpull_risk_score', 50)
                analysis['holder_safety_score'] = 100 - estimated_risk
            
            # Determine verdict
            if analysis['holder_safety_score'] >= 80:
                analysis['concentration_verdict'] = 'SAFE - Good distribution'
            elif analysis['holder_safety_score'] >= 60:
                analysis['concentration_verdict'] = 'CAUTION - Some concentration'
            elif analysis['holder_safety_score'] >= 40:
                analysis['concentration_verdict'] = 'HIGH RISK - Concentrated'
            else:
                analysis['concentration_verdict'] = 'DANGER - Avoid trading'
            
            # Add pattern warnings
            if rugpull_analysis.get('detected_patterns'):
                analysis['warnings'].extend(rugpull_analysis['detected_patterns'])
            
            # Print summary
            print(f"\n📊 HOLDER ANALYSIS SUMMARY:")
            print(f"   Safety Score: {analysis['holder_safety_score']:.0f}/100")
            print(f"   Verdict: {analysis['concentration_verdict']}")
            
            if estimated_data:
                print(f"   Estimated Holders: ~{estimated_data.get('estimated_holders', 0):,}")
                print(f"   Concentration Risk: {estimated_data.get('concentration_risk_level', 'unknown')}")
            
            if analysis['critical_issues']:
                print(f"   🚨 CRITICAL ISSUES:")
                for issue in analysis['critical_issues']:
                    print(f"      • {issue}")
            
            if analysis['warnings']:
                print(f"   ⚠️  WARNINGS:")
                for warning in analysis['warnings']:
                    print(f"      • {warning}")
            
        except Exception as e:
            logger.error(f"Holder analysis failed: {e}")
            analysis['concentration_verdict'] = 'ERROR - Analysis failed'
            analysis['critical_issues'] = [f"Analysis error: {str(e)}"]
        
        return analysis


# Example usage
async def test_holder_analysis():
    """Test holder analysis"""
    analyzer = TokenHolderAnalyzer()
    
    print("👥 Testing Token Holder Analysis")
    print("=" * 40)
    
    # Test with USDC (should have good distribution)
    usdc_address = "0xA0b86a33E6441b811c0a1b26fb18b5b3d05db2b0"
    
    # You can add your Etherscan API key here for actual holder data
    etherscan_key = None  # Replace with your API key
    
    result = await analyzer.comprehensive_holder_analysis(usdc_address, etherscan_key)
    
    print(f"\n✅ Analysis completed!")
    return result


if __name__ == "__main__":
    asyncio.run(test_holder_analysis())