"""
Essential Moralis Endpoints for Token Verification & Legitimacy
Focus on the most critical data points for safe trading
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TokenVerificationAnalyzer:
    """
    Focused analyzer using only the most essential Moralis endpoints
    for token verification and legitimacy checks
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://deep-index.moralis.io/api/v2"
        self.headers = {
            "X-API-Key": api_key,
            "accept": "application/json"
        }
        
        # Essential thresholds for legitimacy
        self.min_holder_count = 100
        self.min_liquidity_usd = 25000
        self.whale_threshold_usd = 50000
        self.max_top5_concentration = 60  # Max 60% held by top 5 wallets
    
    async def _request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        text = await response.text()
                        logger.warning(f"API error {response.status}: {text[:200]}")
                        return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    # 🎯 ENDPOINT 1: TOKEN METADATA (Basic legitimacy)
    async def verify_token_basics(self, token_address: str, chain: str = "0x1") -> Dict:
        """
        Essential: Get basic token information
        Endpoint: /erc20/{address}/metadata
        """
        print(f"📋 Checking basic token metadata...")
        
        result = await self._request(f"/erc20/{token_address}/metadata", {"chain": chain})
        
        verification = {
            "is_valid_token": False,
            "has_name_symbol": False,
            "reasonable_decimals": False,
            "red_flags": []
        }
        
        if result:
            name = result.get('name', '')
            symbol = result.get('symbol', '')
            decimals = result.get('decimals', 0)
            
            verification["is_valid_token"] = True
            verification["token_name"] = name
            verification["token_symbol"] = symbol
            verification["decimals"] = decimals
            
            # Basic checks
            if name and symbol:
                verification["has_name_symbol"] = True
            else:
                verification["red_flags"].append("Missing name or symbol")
            
            if 6 <= decimals <= 18:
                verification["reasonable_decimals"] = True
            else:
                verification["red_flags"].append(f"Unusual decimals: {decimals}")
            
            # Red flag patterns
            if any(word in name.lower() for word in ['test', 'fake', 'scam', 'rug']):
                verification["red_flags"].append("Suspicious name pattern")
        
        return verification
    
    # 🎯 ENDPOINT 2: TOP HOLDERS (Concentration risk)
    async def analyze_holder_concentration(self, token_address: str, chain: str = "0x1") -> Dict:
        """
        Critical: Check if token is controlled by few wallets
        Endpoint: /erc20/{address}/owners
        """
        print(f"👥 Analyzing holder concentration...")
        
        result = await self._request(f"/erc20/{token_address}/owners", {
            "chain": chain, 
            "limit": 20
        })
        
        analysis = {
            "total_holders": 0,
            "top_5_percentage": 0,
            "top_10_percentage": 0,
            "concentration_risk": "unknown",
            "red_flags": []
        }
        
        if result and result.get('result'):
            holders = result['result']
            analysis["total_holders"] = len(holders)
            
            # Calculate concentration
            top_5_pct = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:5])
            top_10_pct = sum(float(h.get('percentage_relative_to_total_supply', 0)) for h in holders[:10])
            
            analysis["top_5_percentage"] = top_5_pct
            analysis["top_10_percentage"] = top_10_pct
            
            # Risk assessment
            if top_5_pct > 80:
                analysis["concentration_risk"] = "extreme"
                analysis["red_flags"].append(f"Top 5 holders own {top_5_pct:.1f}% - EXTREME RISK")
            elif top_5_pct > 60:
                analysis["concentration_risk"] = "high"
                analysis["red_flags"].append(f"Top 5 holders own {top_5_pct:.1f}% - High concentration")
            elif top_5_pct > 40:
                analysis["concentration_risk"] = "medium"
            else:
                analysis["concentration_risk"] = "low"
            
            # Check for holder count
            if len(holders) < self.min_holder_count:
                analysis["red_flags"].append(f"Very few holders ({len(holders)})")
        
        return analysis
    
    # 🎯 ENDPOINT 3: TOKEN TRANSFERS (Real activity check)
    async def verify_real_trading_activity(self, token_address: str, chain: str = "0x1") -> Dict:
        """
        Essential: Check for real trading activity vs fake volume
        Endpoint: /erc20/{address}/transfers
        """
        print(f"💹 Verifying real trading activity...")
        
        result = await self._request(f"/erc20/{token_address}/transfers", {
            "chain": chain,
            "limit": 100
        })
        
        activity = {
            "total_transfers": 0,
            "unique_addresses": 0,
            "recent_activity": 0,
            "avg_transaction_size": 0,
            "real_trading_score": 0,
            "red_flags": []
        }
        
        if result and result.get('result'):
            transfers = result['result']
            activity["total_transfers"] = len(transfers)
            
            # Analyze transfers
            unique_addresses = set()
            recent_transfers = 0
            total_value = 0
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            for transfer in transfers:
                # Track unique addresses
                unique_addresses.add(transfer.get('from_address', ''))
                unique_addresses.add(transfer.get('to_address', ''))
                
                # Check recent activity
                timestamp_str = transfer.get('block_timestamp', '')
                if timestamp_str:
                    transfer_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    if transfer_time > cutoff_time:
                        recent_transfers += 1
                
                # Track value
                value = float(transfer.get('value', 0))
                total_value += value
            
            activity["unique_addresses"] = len(unique_addresses)
            activity["recent_activity"] = recent_transfers
            activity["avg_transaction_size"] = total_value / max(1, len(transfers))
            
            # Calculate real trading score
            address_diversity = min(1.0, len(unique_addresses) / 50)  # 50+ unique addresses = 1.0
            recent_activity_score = min(1.0, recent_transfers / 20)   # 20+ recent transfers = 1.0
            
            activity["real_trading_score"] = (address_diversity + recent_activity_score) / 2
            
            # Red flags
            if len(unique_addresses) < 20:
                activity["red_flags"].append("Very few unique trading addresses")
            
            if recent_transfers < 5:
                activity["red_flags"].append("Very low recent trading activity")
        
        return activity
    
    # 🎯 ENDPOINT 4: WALLET TOKEN BALANCES (Whale verification)
    async def verify_whale_wallets(self, token_address: str, whale_addresses: List[str], chain: str = "0x1") -> Dict:
        """
        Important: Verify if large holders are real whales or suspicious
        Endpoint: /{address}/erc20
        """
        print(f"🐋 Verifying whale wallet legitimacy...")
        
        whale_analysis = {
            "verified_whales": 0,
            "suspicious_wallets": 0,
            "whale_details": [],
            "red_flags": []
        }
        
        for wallet_address in whale_addresses[:5]:  # Check top 5 whales
            result = await self._request(f"/{wallet_address}/erc20", {
                "chain": chain,
                "limit": 50
            })
            
            if result and result.get('result'):
                tokens = result['result']
                
                wallet_info = {
                    "address": wallet_address,
                    "total_tokens": len(tokens),
                    "diversified": len(tokens) > 10,
                    "total_value_usd": sum(float(t.get('usd_value', 0)) for t in tokens),
                    "suspicious_signs": []
                }
                
                # Check for suspicious patterns
                if len(tokens) < 3:
                    wallet_info["suspicious_signs"].append("Very low token diversity")
                
                # Check if wallet only holds this token
                target_token_only = all(
                    t.get('token_address', '').lower() == token_address.lower() 
                    for t in tokens
                )
                if target_token_only:
                    wallet_info["suspicious_signs"].append("Only holds this token")
                
                # Determine if legitimate whale
                if (wallet_info["total_value_usd"] > self.whale_threshold_usd and 
                    wallet_info["diversified"] and 
                    not wallet_info["suspicious_signs"]):
                    whale_analysis["verified_whales"] += 1
                elif wallet_info["suspicious_signs"]:
                    whale_analysis["suspicious_wallets"] += 1
                
                whale_analysis["whale_details"].append(wallet_info)
        
        return whale_analysis
    
    # 🎯 MASTER FUNCTION: Complete Token Verification
    async def complete_token_verification(self, token_address: str, chain: str = "0x1") -> Dict:
        """
        Run all essential checks and provide final verdict
        """
        print(f"\n🔍 COMPLETE TOKEN VERIFICATION for {token_address}")
        print("=" * 60)
        
        verification_result = {
            "token_address": token_address,
            "timestamp": datetime.now().isoformat(),
            "overall_score": 0,
            "verdict": "unknown",
            "critical_issues": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            # 1. Basic token verification
            basic_check = await self.verify_token_basics(token_address, chain)
            verification_result["details"]["basic_info"] = basic_check
            
            if not basic_check["is_valid_token"]:
                verification_result["critical_issues"].append("Invalid or non-existent token")
                verification_result["verdict"] = "AVOID - Invalid token"
                return verification_result
            
            # 2. Holder concentration analysis
            concentration = await self.analyze_holder_concentration(token_address, chain)
            verification_result["details"]["concentration"] = concentration
            
            # 3. Trading activity verification
            activity = await self.verify_real_trading_activity(token_address, chain)
            verification_result["details"]["activity"] = activity
            
            # 4. Whale verification (if we have top holders)
            if concentration.get("total_holders", 0) > 0:
                # Get top holder addresses for whale verification
                # This would require parsing the holder data
                whale_check = {"verified_whales": 0, "suspicious_wallets": 0}
                verification_result["details"]["whales"] = whale_check
            
            # Calculate overall score
            score = 0
            max_score = 0
            
            # Basic info scoring
            if basic_check.get("has_name_symbol"):
                score += 20
            max_score += 20
            
            if basic_check.get("reasonable_decimals"):
                score += 10
            max_score += 10
            
            # Concentration scoring
            concentration_risk = concentration.get("concentration_risk", "unknown")
            if concentration_risk == "low":
                score += 30
            elif concentration_risk == "medium":
                score += 15
            max_score += 30
            
            # Activity scoring
            real_trading_score = activity.get("real_trading_score", 0)
            score += real_trading_score * 30
            max_score += 30
            
            # Holder count scoring
            holder_count = concentration.get("total_holders", 0)
            if holder_count >= self.min_holder_count:
                score += 10
            max_score += 10
            
            # Calculate final score
            verification_result["overall_score"] = (score / max_score) * 100 if max_score > 0 else 0
            
            # Collect all red flags
            all_red_flags = []
            all_red_flags.extend(basic_check.get("red_flags", []))
            all_red_flags.extend(concentration.get("red_flags", []))
            all_red_flags.extend(activity.get("red_flags", []))
            
            # Determine final verdict
            if verification_result["overall_score"] >= 80 and not all_red_flags:
                verification_result["verdict"] = "SAFE - High confidence"
            elif verification_result["overall_score"] >= 60 and len(all_red_flags) <= 1:
                verification_result["verdict"] = "CAUTION - Moderate risk"
            elif verification_result["overall_score"] >= 40:
                verification_result["verdict"] = "HIGH RISK - Trade carefully"
            else:
                verification_result["verdict"] = "AVOID - Too many red flags"
            
            verification_result["critical_issues"] = [flag for flag in all_red_flags if any(word in flag.lower() for word in ['extreme', 'scam', 'fake'])]
            verification_result["warnings"] = [flag for flag in all_red_flags if flag not in verification_result["critical_issues"]]
            
            # Print summary
            print(f"📊 VERIFICATION SUMMARY:")
            print(f"   Score: {verification_result['overall_score']:.1f}/100")
            print(f"   Verdict: {verification_result['verdict']}")
            print(f"   Token: {basic_check.get('token_name', 'Unknown')} ({basic_check.get('token_symbol', 'Unknown')})")
            print(f"   Holders: {concentration.get('total_holders', 0)}")
            print(f"   Top 5 Concentration: {concentration.get('top_5_percentage', 0):.1f}%")
            print(f"   Trading Activity Score: {activity.get('real_trading_score', 0):.2f}")
            
            if verification_result["critical_issues"]:
                print(f"   ❌ CRITICAL ISSUES: {', '.join(verification_result['critical_issues'])}")
            
            if verification_result["warnings"]:
                print(f"   ⚠️  WARNINGS: {', '.join(verification_result['warnings'])}")
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            verification_result["verdict"] = "ERROR - Verification failed"
            verification_result["critical_issues"].append(f"Analysis error: {str(e)}")
        
        return verification_result


# Example usage and testing
async def test_essential_verification():
    """Test the essential token verification"""
    
    # Load API key
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        api_key = config['api_keys']['moralis']
    except:
        print("❌ Please add your Moralis API key to config.json")
        return
    
    analyzer = TokenVerificationAnalyzer(api_key)
    
    print("🔍 Testing Essential Token Verification System")
    print("=" * 50)
    
    # Test with a well-known legitimate token (UNI)
    uni_address = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"
    
    verification = await analyzer.complete_token_verification(uni_address)
    
    print(f"\n✅ Verification completed!")
    print(f"Final verdict: {verification['verdict']}")
    
    return verification


if __name__ == "__main__":
    asyncio.run(test_essential_verification())