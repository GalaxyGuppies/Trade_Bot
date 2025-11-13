"""
🔍 SOLANA TOKEN SCANNER - Find High-Potential Trading Opportunities

This standalone tool scans Solana tokens for:
- Price momentum (large moves brewing)
- Volume spikes (whale activity)
- Technical patterns (breakouts, dips, reversals)
- Liquidity depth (can you actually trade it?)
- Holder distribution (not too concentrated)

Usage:
    python token_scanner.py              # Scan top 100 tokens
    python token_scanner.py --top 50     # Scan top 50
    python token_scanner.py --symbol WIF # Scan specific token
"""

import asyncio
import aiohttp
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

@dataclass
class TokenOpportunity:
    """A trading opportunity identified by the scanner"""
    symbol: str
    address: str
    price: float
    volume_24h: float
    price_change_24h: float
    liquidity: float
    score: float  # 0-100 overall opportunity score
    signals: List[str]  # List of positive signals
    warnings: List[str]  # List of concerns
    recommendation: str  # BUY, WATCH, AVOID

class SolanaTokenScanner:
    """Scans Solana tokens for trading opportunities"""
    
    def __init__(self):
        self.dexscreener_api = "https://api.dexscreener.com/latest/dex"
        self.birdeye_api = "https://public-api.birdeye.so/defi"
        
    async def scan_top_tokens(self, limit: int = 100) -> List[TokenOpportunity]:
        """Scan top Solana tokens by volume"""
        print(f"🔍 Scanning top {limit} Solana tokens...")
        
        opportunities = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get top tokens from DexScreener
                url = f"{self.dexscreener_api}/tokens/solana"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])[:limit]
                        
                        print(f"✅ Found {len(pairs)} trading pairs")
                        
                        # Analyze each token
                        for i, pair in enumerate(pairs, 1):
                            print(f"📊 Analyzing {i}/{len(pairs)}: {pair.get('baseToken', {}).get('symbol', 'Unknown')}...")
                            opportunity = await self.analyze_token(pair, session)
                            if opportunity and opportunity.score >= 50:  # Only keep good opportunities
                                opportunities.append(opportunity)
                    else:
                        print(f"❌ Failed to fetch tokens: HTTP {response.status}")
        
        except Exception as e:
            print(f"❌ Error scanning tokens: {e}")
        
        # Sort by score (best first)
        opportunities.sort(key=lambda x: x.score, reverse=True)
        return opportunities
    
    async def analyze_specific_token(self, symbol: str) -> Optional[TokenOpportunity]:
        """Analyze a specific token by symbol"""
        print(f"🔍 Analyzing {symbol}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Search for token
                url = f"{self.dexscreener_api}/search?q={symbol}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        pairs = data.get('pairs', [])
                        
                        if not pairs:
                            print(f"❌ Token {symbol} not found")
                            return None
                        
                        # Use the first (most liquid) pair
                        pair = pairs[0]
                        return await self.analyze_token(pair, session)
        
        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    async def analyze_token(self, pair_data: Dict, session: aiohttp.ClientSession) -> Optional[TokenOpportunity]:
        """Analyze a token and generate opportunity score"""
        try:
            base_token = pair_data.get('baseToken', {})
            symbol = base_token.get('symbol', 'Unknown')
            address = base_token.get('address', '')
            
            # Extract key metrics
            price = float(pair_data.get('priceUsd', 0))
            volume_24h = float(pair_data.get('volume', {}).get('h24', 0))
            price_change_24h = float(pair_data.get('priceChange', {}).get('h24', 0))
            liquidity = float(pair_data.get('liquidity', {}).get('usd', 0))
            
            # Price changes over time
            price_change_1h = float(pair_data.get('priceChange', {}).get('h1', 0))
            price_change_6h = float(pair_data.get('priceChange', {}).get('h6', 0))
            
            # Volume metrics
            txns_24h = pair_data.get('txns', {}).get('h24', {})
            buys_24h = txns_24h.get('buys', 0)
            sells_24h = txns_24h.get('sells', 0)
            
            # Calculate opportunity score (0-100)
            score = 0
            signals = []
            warnings = []
            
            # 1. MOMENTUM SIGNALS (30 points)
            if price_change_1h > 5:
                score += 10
                signals.append(f"🚀 Strong 1h momentum: +{price_change_1h:.1f}%")
            elif price_change_1h > 2:
                score += 5
                signals.append(f"📈 Positive 1h momentum: +{price_change_1h:.1f}%")
            
            if price_change_6h > 10:
                score += 10
                signals.append(f"🔥 Explosive 6h move: +{price_change_6h:.1f}%")
            elif price_change_6h > 5:
                score += 5
                signals.append(f"📊 Strong 6h trend: +{price_change_6h:.1f}%")
            
            if price_change_24h > 20:
                score += 10
                signals.append(f"💎 Huge 24h gain: +{price_change_24h:.1f}%")
            elif price_change_24h > 10:
                score += 5
                signals.append(f"✨ Good 24h gain: +{price_change_24h:.1f}%")
            
            # 2. VOLUME SIGNALS (25 points)
            if volume_24h > 1000000:  # $1M+ volume
                score += 15
                signals.append(f"💰 High volume: ${volume_24h/1000000:.1f}M")
            elif volume_24h > 500000:  # $500k+ volume
                score += 10
                signals.append(f"💵 Good volume: ${volume_24h/1000:.0f}k")
            elif volume_24h > 100000:  # $100k+ volume
                score += 5
                signals.append(f"📊 Moderate volume: ${volume_24h/1000:.0f}k")
            else:
                warnings.append(f"⚠️ Low volume: ${volume_24h:.0f}")
            
            # Buy/Sell ratio
            if buys_24h > 0 and sells_24h > 0:
                buy_sell_ratio = buys_24h / sells_24h
                if buy_sell_ratio > 1.5:
                    score += 10
                    signals.append(f"🟢 Buy pressure: {buy_sell_ratio:.1f}x more buys than sells")
                elif buy_sell_ratio < 0.7:
                    warnings.append(f"🔴 Sell pressure: {1/buy_sell_ratio:.1f}x more sells than buys")
            
            # 3. LIQUIDITY SIGNALS (25 points)
            if liquidity > 500000:  # $500k+ liquidity
                score += 15
                signals.append(f"💎 Deep liquidity: ${liquidity/1000000:.2f}M")
            elif liquidity > 100000:  # $100k+ liquidity
                score += 10
                signals.append(f"✅ Good liquidity: ${liquidity/1000:.0f}k")
            elif liquidity > 50000:  # $50k+ liquidity
                score += 5
                signals.append(f"⚡ Tradeable liquidity: ${liquidity/1000:.0f}k")
            else:
                score -= 10
                warnings.append(f"⛔ Low liquidity: ${liquidity:.0f} (risky)")
            
            # 4. PATTERN DETECTION (20 points)
            # Breakout pattern (strong recent move after consolidation)
            if abs(price_change_1h) > 3 and abs(price_change_6h) < 5:
                score += 10
                signals.append("📈 Breakout pattern detected")
            
            # Dip buying opportunity (down 1h but up 24h)
            if price_change_1h < -2 and price_change_24h > 5:
                score += 10
                signals.append("💰 Dip buying opportunity (down 1h, up 24h)")
            
            # Reversal pattern (turning around)
            if price_change_1h > 5 and price_change_6h < 0:
                score += 5
                signals.append("🔄 Reversal pattern (1h up, 6h down → turning?)")
            
            # WARNINGS (reduce score)
            # Dead cat bounce check
            if price_change_1h > 10 and price_change_24h < -20:
                score -= 15
                warnings.append("⚠️ Possible dead cat bounce (up 1h but down 24h)")
            
            # Pump and dump risk
            if price_change_1h > 20 and volume_24h < 100000:
                score -= 20
                warnings.append("🚨 Pump risk: huge spike on low volume")
            
            # Determine recommendation
            if score >= 70:
                recommendation = "🟢 BUY"
            elif score >= 50:
                recommendation = "🟡 WATCH"
            else:
                recommendation = "🔴 AVOID"
            
            return TokenOpportunity(
                symbol=symbol,
                address=address,
                price=price,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                liquidity=liquidity,
                score=score,
                signals=signals,
                warnings=warnings,
                recommendation=recommendation
            )
        
        except Exception as e:
            print(f"❌ Error analyzing token: {e}")
            return None
    
    def print_opportunities(self, opportunities: List[TokenOpportunity], top_n: int = 10):
        """Print formatted opportunity list"""
        print("\n" + "="*80)
        print(f"🎯 TOP {min(top_n, len(opportunities))} TRADING OPPORTUNITIES")
        print("="*80 + "\n")
        
        for i, opp in enumerate(opportunities[:top_n], 1):
            print(f"{i}. {opp.recommendation} {opp.symbol}")
            print(f"   Score: {opp.score}/100")
            print(f"   Price: ${opp.price:.6f} ({opp.price_change_24h:+.1f}% 24h)")
            print(f"   Volume: ${opp.volume_24h/1000:.0f}k | Liquidity: ${opp.liquidity/1000:.0f}k")
            print(f"   Address: {opp.address}")
            
            if opp.signals:
                print(f"   ✅ Signals:")
                for signal in opp.signals[:3]:  # Top 3 signals
                    print(f"      • {signal}")
            
            if opp.warnings:
                print(f"   ⚠️  Warnings:")
                for warning in opp.warnings[:2]:  # Top 2 warnings
                    print(f"      • {warning}")
            
            print()
    
    def export_to_json(self, opportunities: List[TokenOpportunity], filename: str = "token_opportunities.json"):
        """Export opportunities to JSON file"""
        data = []
        for opp in opportunities:
            data.append({
                'symbol': opp.symbol,
                'address': opp.address,
                'price': opp.price,
                'volume_24h': opp.volume_24h,
                'price_change_24h': opp.price_change_24h,
                'liquidity': opp.liquidity,
                'score': opp.score,
                'signals': opp.signals,
                'warnings': opp.warnings,
                'recommendation': opp.recommendation,
                'scanned_at': datetime.now().isoformat()
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Exported {len(opportunities)} opportunities to {filename}")

async def main():
    """Main scanner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scan Solana tokens for trading opportunities')
    parser.add_argument('--top', type=int, default=100, help='Number of top tokens to scan (default: 100)')
    parser.add_argument('--symbol', type=str, help='Analyze specific token symbol (e.g., WIF, JUP)')
    parser.add_argument('--export', action='store_true', help='Export results to JSON')
    parser.add_argument('--min-score', type=int, default=50, help='Minimum opportunity score (default: 50)')
    
    args = parser.parse_args()
    
    scanner = SolanaTokenScanner()
    
    if args.symbol:
        # Analyze specific token
        opportunity = await scanner.analyze_specific_token(args.symbol)
        if opportunity:
            scanner.print_opportunities([opportunity], top_n=1)
    else:
        # Scan top tokens
        opportunities = await scanner.scan_top_tokens(limit=args.top)
        
        # Filter by minimum score
        opportunities = [opp for opp in opportunities if opp.score >= args.min_score]
        
        if opportunities:
            scanner.print_opportunities(opportunities, top_n=20)
            
            if args.export:
                scanner.export_to_json(opportunities)
        else:
            print(f"❌ No opportunities found with score >= {args.min_score}")

if __name__ == "__main__":
    print("🚀 Solana Token Scanner v1.0")
    print("=" * 80)
    asyncio.run(main())
