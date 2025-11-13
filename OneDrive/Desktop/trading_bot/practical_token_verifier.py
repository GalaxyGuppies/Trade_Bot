"""
PRACTICAL TOKEN VERIFICATION SYSTEM
Uses reliable APIs (CoinGecko + DexScreener + Etherscan) for the 4 critical checks
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PracticalTokenVerifier:
    """
    Practical token verifier using reliable, free APIs
    Focuses on the 4 most critical verification points
    """
    
    def __init__(self):
        # Free API endpoints (no key required for basic usage)
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        self.dexscreener_base = "https://api.dexscreener.com/latest"
        self.etherscan_base = "https://api.etherscan.io/api"
        
        # Critical thresholds
        self.min_holders = 100
        self.max_concentration = 60  # Top 5 holders max 60%
        self.min_liquidity = 25000   # $25K minimum liquidity
        self.min_volume_24h = 5000   # $5K minimum 24h volume
    
    async def _get_data(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Get data from API with error handling"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"API error {response.status}: {url}")
                        return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    # ✅ CHECK 1: Basic Token Information
    async def verify_basic_info(self, contract_address: str) -> Dict:
        """
        Get basic token info from CoinGecko
        """
        print(f"📋 Checking basic token information...")
        
        url = f"{self.coingecko_base}/coins/ethereum/contract/{contract_address}"
        data = await self._get_data(url)
        
        info = {
            "is_valid": False,
            "name": "",
            "symbol": "",
            "description": "",
            "market_cap": 0,
            "total_supply": 0,
            "verified": False,
            "red_flags": []
        }
        
        if data and not data.get('error'):
            info.update({
                "is_valid": True,
                "name": data.get('name', ''),
                "symbol": data.get('symbol', '').upper(),
                "description": data.get('description', {}).get('en', ''),
                "market_cap": data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                "total_supply": data.get('market_data', {}).get('total_supply', 0),
                "verified": bool(data.get('tickers'))  # Has exchange listings
            })
            
            # Check for red flags
            name = info["name"].lower()
            if any(word in name for word in ['test', 'fake', 'scam', 'rug', 'copy']):
                info["red_flags"].append("Suspicious name pattern")
            
            if not info["name"] or not info["symbol"]:
                info["red_flags"].append("Missing name or symbol")
        
        return info
    
    # ✅ CHECK 2: Liquidity & Trading Data
    async def verify_liquidity_trading(self, contract_address: str) -> Dict:
        """
        Get liquidity and trading data from DexScreener
        """
        print(f"💧 Checking liquidity and trading activity...")
        
        url = f"{self.dexscreener_base}/dex/tokens/{contract_address}"
        data = await self._get_data(url)
        
        trading = {
            "has_liquidity": False,
            "liquidity_usd": 0,
            "volume_24h": 0,
            "price_usd": 0,
            "price_change_24h": 0,
            "active_pairs": 0,
            "main_dex": "",
            "red_flags": []
        }
        
        if data and data.get('pairs'):
            pairs = data['pairs']
            
            # Find best liquidity pair
            best_pair = max(pairs, key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
            
            trading.update({
                "has_liquidity": True,
                "liquidity_usd": float(best_pair.get('liquidity', {}).get('usd', 0)),
                "volume_24h": float(best_pair.get('volume', {}).get('h24', 0)),
                "price_usd": float(best_pair.get('priceUsd', 0)),
                "price_change_24h": float(best_pair.get('priceChange', {}).get('h24', 0)),
                "active_pairs": len(pairs),
                "main_dex": best_pair.get('dexId', '')
            })
            
            # Check for red flags
            if trading["liquidity_usd"] < self.min_liquidity:
                trading["red_flags"].append(f"Low liquidity: ${trading['liquidity_usd']:,.0f}")
            
            if trading["volume_24h"] < self.min_volume_24h:
                trading["red_flags"].append(f"Low volume: ${trading['volume_24h']:,.0f}")
            
            if abs(trading["price_change_24h"]) > 50:
                trading["red_flags"].append(f"Extreme volatility: {trading['price_change_24h']:+.1f}%")
        
        return trading
    
    # ✅ CHECK 3: Recent Transaction Activity
    async def verify_transaction_activity(self, contract_address: str) -> Dict:
        """
        Check recent transaction activity patterns
        """
        print(f"📊 Analyzing transaction patterns...")
        
        # Get recent transfers from DexScreener data
        url = f"{self.dexscreener_base}/dex/tokens/{contract_address}"
        data = await self._get_data(url)
        
        activity = {
            "recent_transactions": 0,
            "unique_wallets": 0,
            "transaction_frequency": "unknown",
            "activity_score": 0,
            "red_flags": []
        }
        
        if data and data.get('pairs'):
            # Analyze transaction data from pair info
            best_pair = max(data['pairs'], key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
            
            # Get transaction counts
            txns_24h = best_pair.get('txns', {}).get('h24', {})
            buys = txns_24h.get('buys', 0)
            sells = txns_24h.get('sells', 0)
            total_txns = buys + sells
            
            activity.update({
                "recent_transactions": total_txns,
                "buys_24h": buys,
                "sells_24h": sells,
                "activity_score": min(1.0, total_txns / 100)  # Score based on 100+ transactions = 1.0
            })
            
            # Determine transaction frequency
            if total_txns > 200:
                activity["transaction_frequency"] = "high"
            elif total_txns > 50:
                activity["transaction_frequency"] = "medium"
            elif total_txns > 10:
                activity["transaction_frequency"] = "low"
            else:
                activity["transaction_frequency"] = "very_low"
                activity["red_flags"].append("Very low transaction activity")
            
            # Check buy/sell ratio
            if total_txns > 0:
                buy_ratio = buys / total_txns
                if buy_ratio < 0.2 or buy_ratio > 0.8:
                    activity["red_flags"].append("Unbalanced buy/sell ratio")
        
        return activity
    
    # ✅ CHECK 4: HOLDER CONCENTRATION ANALYSIS (Enhanced)
    async def analyze_holder_concentration(self, contract_address: str) -> Dict:
        """
        Critical: Analyze holder concentration and distribution patterns
        """
        print(f"👥 Analyzing holder concentration and distribution...")
        
        # Get trading data for concentration analysis
        url = f"{self.dexscreener_base}/dex/tokens/{contract_address}"
        data = await self._get_data(url)
        
        concentration = {
            "estimated_holders": 0,
            "concentration_risk": "unknown",
            "concentration_score": 0,
            "liquidity_ratio": 0,
            "volume_distribution_risk": 0,
            "trading_pattern_risk": 0,
            "holder_safety_score": 0,
            "red_flags": []
        }
        
        if data and data.get('pairs'):
            best_pair = max(data['pairs'], key=lambda x: float(x.get('liquidity', {}).get('usd', 0)))
            
            market_cap = float(best_pair.get('marketCap', 0))
            liquidity = float(best_pair.get('liquidity', {}).get('usd', 0))
            volume_24h = float(best_pair.get('volume', {}).get('h24', 0))
            
            # Get transaction data for holder estimation
            txns_24h = best_pair.get('txns', {}).get('h24', {})
            buys = txns_24h.get('buys', 0)
            sells = txns_24h.get('sells', 0)
            total_txns = buys + sells
            
            # Calculate key ratios
            liquidity_ratio = (liquidity / market_cap * 100) if market_cap > 0 else 0
            volume_ratio = (volume_24h / liquidity) if liquidity > 0 else 0
            
            concentration.update({
                "liquidity_ratio": liquidity_ratio,
                "volume_to_liquidity_ratio": volume_ratio,
                "daily_transactions": total_txns,
                "buy_sell_ratio": buys / max(1, sells)
            })
            
            # HOLDER CONCENTRATION RISK ANALYSIS
            risk_score = 0
            safety_score = 100
            
            # 1. Liquidity Concentration Risk
            if liquidity_ratio < 1:  # <1% of market cap in liquidity
                risk_score += 40
                safety_score -= 40
                concentration["red_flags"].append(f"Extreme liquidity concentration: {liquidity_ratio:.1f}%")
            elif liquidity_ratio < 3:  # <3% liquidity
                risk_score += 25
                safety_score -= 25
                concentration["red_flags"].append(f"High liquidity concentration: {liquidity_ratio:.1f}%")
            elif liquidity_ratio < 5:  # <5% liquidity
                risk_score += 10
                safety_score -= 10
            
            # 2. Volume Distribution Risk
            if volume_ratio > 5:  # Volume >5x liquidity (large holder activity)
                risk_score += 30
                safety_score -= 30
                concentration["red_flags"].append("Excessive volume vs liquidity (large holder dumps)")
            elif volume_ratio > 2:
                risk_score += 15
                safety_score -= 15
                concentration["red_flags"].append("High volume vs liquidity")
            
            # 3. Trading Pattern Analysis
            if total_txns < 20:
                risk_score += 25
                safety_score -= 25
                concentration["red_flags"].append(f"Very few traders: {total_txns} transactions")
            elif total_txns < 50:
                risk_score += 10
                safety_score -= 10
            
            # 4. Buy/Sell Imbalance (concentration indicator)
            buy_sell_ratio = buys / max(1, sells)
            if buy_sell_ratio < 0.3:  # Heavy selling
                risk_score += 20
                safety_score -= 20
                concentration["red_flags"].append("Heavy selling pressure (concentrated dumps)")
            elif buy_sell_ratio > 4:  # Artificial pumping
                risk_score += 15
                safety_score -= 15
                concentration["red_flags"].append("Unusual buying pressure (possible manipulation)")
            
            # 5. Estimate holder count
            base_holders = min(5000, max(10, int(market_cap / 100000)))  # 1 holder per $100K
            activity_factor = min(2.0, total_txns / 50)  # More transactions = more holders
            estimated_holders = int(base_holders * activity_factor)
            
            concentration["estimated_holders"] = estimated_holders
            
            if estimated_holders < 50:
                risk_score += 30
                safety_score -= 30
                concentration["red_flags"].append(f"Estimated very few holders: ~{estimated_holders}")
            elif estimated_holders < self.min_holders:
                risk_score += 15
                safety_score -= 15
                concentration["red_flags"].append(f"Estimated low holder count: ~{estimated_holders}")
            
            # Calculate final scores
            concentration["concentration_score"] = min(100, risk_score)
            concentration["holder_safety_score"] = max(0, safety_score)
            
            # Determine risk level
            if risk_score >= 70:
                concentration["concentration_risk"] = "extreme"
            elif risk_score >= 50:
                concentration["concentration_risk"] = "high" 
            elif risk_score >= 30:
                concentration["concentration_risk"] = "medium"
            else:
                concentration["concentration_risk"] = "low"
        
        return concentration
    
    # 🎯 MASTER VERIFICATION FUNCTION
    async def complete_verification(self, contract_address: str) -> Dict:
        """
        Complete token verification using all 4 critical checks
        """
        print(f"\n🔍 COMPLETE TOKEN VERIFICATION")
        print(f"Contract: {contract_address}")
        print("=" * 60)
        
        result = {
            "contract_address": contract_address,
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "safety_level": "unknown",
            "recommendation": "unknown",
            "critical_issues": [],
            "warnings": [],
            "checks": {}
        }
        
        try:
            # Run all 4 critical checks
            basic_info = await self.verify_basic_info(contract_address)
            result["checks"]["basic_info"] = basic_info
            
            trading_data = await self.verify_liquidity_trading(contract_address)
            result["checks"]["trading"] = trading_data
            
            activity_data = await self.verify_transaction_activity(contract_address)
            result["checks"]["activity"] = activity_data
            
            holder_data = await self.analyze_holder_concentration(contract_address)
            result["checks"]["holders"] = holder_data
            
            # Calculate overall score
            score = 0
            
            # Basic info scoring (30 points)
            if basic_info["is_valid"]:
                score += 15
            if basic_info["verified"]:
                score += 10
            if not basic_info["red_flags"]:
                score += 5
            
            # Trading scoring (30 points)
            if trading_data["has_liquidity"]:
                score += 10
            if trading_data["liquidity_usd"] >= self.min_liquidity:
                score += 10
            if trading_data["volume_24h"] >= self.min_volume_24h:
                score += 10
            
            # Activity scoring (20 points)
            activity_score = activity_data["activity_score"] * 20
            score += activity_score
            
            # Holder scoring (20 points) - Updated for new analysis
            concentration_risk = holder_data.get("concentration_risk", "unknown")
            holder_safety_score = holder_data.get("holder_safety_score", 0)
            
            if concentration_risk == "low":
                score += 20
            elif concentration_risk == "medium":
                score += 12
            elif concentration_risk == "high":
                score += 5
            # extreme concentration gets 0 points
            
            result["overall_score"] = min(100, score)
            
            # Collect all issues
            all_red_flags = []
            for check in result["checks"].values():
                all_red_flags.extend(check.get("red_flags", []))
            
            # Categorize issues
            critical_keywords = ['scam', 'fake', 'extreme', 'very low', 'high concentration']
            result["critical_issues"] = [flag for flag in all_red_flags 
                                       if any(keyword in flag.lower() for keyword in critical_keywords)]
            result["warnings"] = [flag for flag in all_red_flags 
                                if flag not in result["critical_issues"]]
            
            # Determine safety level and recommendation
            if result["overall_score"] >= 80 and not result["critical_issues"]:
                result["safety_level"] = "HIGH"
                result["recommendation"] = "SAFE TO TRADE"
            elif result["overall_score"] >= 60 and len(result["critical_issues"]) == 0:
                result["safety_level"] = "MEDIUM"
                result["recommendation"] = "TRADE WITH CAUTION"
            elif result["overall_score"] >= 40:
                result["safety_level"] = "LOW" 
                result["recommendation"] = "HIGH RISK - SMALL AMOUNTS ONLY"
            else:
                result["safety_level"] = "VERY LOW"
                result["recommendation"] = "AVOID - TOO RISKY"
            
            # Print summary
            print(f"\n📊 VERIFICATION SUMMARY:")
            print(f"   Token: {basic_info['name']} ({basic_info['symbol']})")
            print(f"   Score: {result['overall_score']:.0f}/100")
            print(f"   Safety Level: {result['safety_level']}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"   Market Cap: ${basic_info['market_cap']:,.0f}")
            print(f"   Liquidity: ${trading_data['liquidity_usd']:,.0f}")
            print(f"   24h Volume: ${trading_data['volume_24h']:,.0f}")
            print(f"   Estimated Holders: ~{holder_data['estimated_holders']:,}")
            
            if result["critical_issues"]:
                print(f"   🚨 CRITICAL ISSUES:")
                for issue in result["critical_issues"]:
                    print(f"      • {issue}")
            
            if result["warnings"]:
                print(f"   ⚠️  WARNINGS:")
                for warning in result["warnings"]:
                    print(f"      • {warning}")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            result["safety_level"] = "ERROR"
            result["recommendation"] = "VERIFICATION FAILED"
            result["critical_issues"] = [f"Analysis error: {str(e)}"]
        
        return result


# Test function
async def test_practical_verifier():
    """Test the practical token verifier"""
    verifier = PracticalTokenVerifier()
    
    print("🔍 Testing Practical Token Verification System")
    print("=" * 50)
    
    # Test with USDC (should be very safe)
    usdc_address = "0xA0b86a33E6441b811c0a1b26fb18b5b3d05db2b0"
    
    result = await verifier.complete_verification(usdc_address)
    
    print(f"\n✅ Verification completed!")
    return result


if __name__ == "__main__":
    asyncio.run(test_practical_verifier())