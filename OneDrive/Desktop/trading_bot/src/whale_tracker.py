"""
🐋 Whale Wallet Tracker for Solana

Monitors whale wallets in real-time and detects trades.
When a whale buys a token, your bot copies the trade automatically.

Features:
- Real-time transaction monitoring via Solana RPC
- Token swap detection (Jupiter, Raydium, etc.)
- Automatic signal generation for copy-trading
- Wallet verification and activity checking
"""

import asyncio
import aiohttp
import base58
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

@dataclass
class WhaleSignal:
    """A whale trade signal for copy-trading"""
    whale_wallet: str
    token_symbol: str
    token_address: str
    action: str  # 'BUY' or 'SELL'
    amount_sol: float
    amount_tokens: float
    price: float
    timestamp: datetime
    confidence: float  # 0-1, based on wallet reputation

class WhaleTracker:
    """Tracks whale wallets and generates copy-trade signals"""
    
    def __init__(self, whale_wallets: List[str], rpc_url: str = "https://api.mainnet-beta.solana.com"):
        self.whale_wallets = whale_wallets
        self.rpc_url = rpc_url
        self.running = False
        self.signals = []
        self.wallet_stats = {}
        
        # Token addresses we care about (SOL and common tokens)
        self.sol_address = "So11111111111111111111111111111111111111112"
        self.wsol_address = "So11111111111111111111111111111111111111112"
        
    async def verify_wallet(self, wallet_address: str) -> Dict:
        """Verify if a wallet is actually a whale (high activity/balance)"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get wallet balance
                balance_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getBalance",
                    "params": [wallet_address]
                }
                
                async with session.post(
                    self.rpc_url,
                    json=balance_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    balance_data = await response.json()
                    balance_lamports = balance_data.get('result', {}).get('value', 0)
                    balance_sol = balance_lamports / 1e9
                
                # Get recent transaction signatures
                sigs_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [wallet_address, {"limit": 100}]
                }
                
                async with session.post(
                    self.rpc_url,
                    json=sigs_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    sigs_data = await response.json()
                    recent_txs = sigs_data.get('result', [])
                
                # Calculate activity metrics
                total_txs = len(recent_txs)
                
                # Check if transactions are recent
                if recent_txs:
                    latest_tx = recent_txs[0]
                    latest_time = latest_tx.get('blockTime', 0)
                    hours_since_last = (datetime.now().timestamp() - latest_time) / 3600 if latest_time else 999
                else:
                    hours_since_last = 999
                
                # Whale criteria
                is_whale = balance_sol >= 10  # At least 10 SOL
                is_active = hours_since_last < 24  # Active in last 24 hours
                is_trader = total_txs >= 10  # At least 10 recent transactions
                
                return {
                    'address': wallet_address,
                    'balance_sol': balance_sol,
                    'recent_transactions': total_txs,
                    'hours_since_last_tx': hours_since_last,
                    'is_whale': is_whale,
                    'is_active': is_active,
                    'is_trader': is_trader,
                    'is_verified': is_whale and is_active and is_trader,
                    'confidence': self._calculate_confidence(balance_sol, total_txs, hours_since_last)
                }
        
        except Exception as e:
            print(f"❌ Error verifying wallet {wallet_address}: {e}")
            return {
                'address': wallet_address,
                'error': str(e),
                'is_verified': False,
                'confidence': 0
            }
    
    def _calculate_confidence(self, balance: float, txs: int, hours_since: float) -> float:
        """Calculate confidence score for whale wallet (0-1)"""
        score = 0.0
        
        # Balance score (0-0.4)
        if balance >= 1000:
            score += 0.4
        elif balance >= 100:
            score += 0.3
        elif balance >= 10:
            score += 0.2
        
        # Activity score (0-0.3)
        if hours_since < 1:
            score += 0.3
        elif hours_since < 6:
            score += 0.2
        elif hours_since < 24:
            score += 0.1
        
        # Transaction volume score (0-0.3)
        if txs >= 50:
            score += 0.3
        elif txs >= 25:
            score += 0.2
        elif txs >= 10:
            score += 0.1
        
        return min(score, 1.0)
    
    async def get_recent_trades(self, wallet_address: str, limit: int = 20) -> List[WhaleSignal]:
        """Get recent trades from a whale wallet"""
        signals = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get recent transaction signatures
                sigs_payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignaturesForAddress",
                    "params": [wallet_address, {"limit": limit}]
                }
                
                async with session.post(
                    self.rpc_url,
                    json=sigs_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    sigs_data = await response.json()
                    signatures = sigs_data.get('result', [])
                
                # Analyze each transaction
                for sig_info in signatures[:10]:  # Check last 10 transactions
                    sig = sig_info.get('signature')
                    block_time = sig_info.get('blockTime')
                    
                    if not sig or not block_time:
                        continue
                    
                    # Get transaction details
                    tx_payload = {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "getTransaction",
                        "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}]
                    }
                    
                    async with session.post(
                        self.rpc_url,
                        json=tx_payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as tx_response:
                        tx_data = await tx_response.json()
                        
                        # Parse transaction for token swaps
                        signal = self._parse_swap_transaction(
                            tx_data.get('result', {}),
                            wallet_address,
                            block_time
                        )
                        
                        if signal:
                            signals.append(signal)
        
        except Exception as e:
            print(f"❌ Error getting trades for {wallet_address}: {e}")
        
        return signals
    
    def _parse_swap_transaction(self, tx_result: Dict, wallet: str, block_time: int) -> Optional[WhaleSignal]:
        """Parse a transaction to detect token swaps"""
        try:
            if not tx_result or 'meta' not in tx_result:
                return None
            
            meta = tx_result.get('meta', {})
            transaction = tx_result.get('transaction', {})
            message = transaction.get('message', {})
            instructions = message.get('instructions', [])
            
            # Look for token swap instructions (Jupiter, Raydium, etc.)
            for instruction in instructions:
                program = instruction.get('programId')
                
                # Jupiter V6 program
                if program == 'JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4':
                    # This is a Jupiter swap
                    parsed = instruction.get('parsed')
                    if parsed:
                        # Extract swap details
                        # This would need more detailed parsing based on Jupiter's instruction format
                        pass
            
            # Alternatively, analyze token balance changes
            pre_balances = meta.get('preTokenBalances', [])
            post_balances = meta.get('postTokenBalances', [])
            
            # Find token balance changes for this wallet
            for pre, post in zip(pre_balances, post_balances):
                if pre.get('owner') == wallet and post.get('owner') == wallet:
                    pre_amount = float(pre.get('uiTokenAmount', {}).get('uiAmount', 0))
                    post_amount = float(post.get('uiTokenAmount', {}).get('uiAmount', 0))
                    mint = post.get('mint')
                    
                    change = post_amount - pre_amount
                    
                    # Significant change = trade
                    if abs(change) > 0:
                        action = 'BUY' if change > 0 else 'SELL'
                        
                        # Get wallet confidence from cache
                        confidence = self.wallet_stats.get(wallet, {}).get('confidence', 0.5)
                        
                        return WhaleSignal(
                            whale_wallet=wallet,
                            token_symbol='UNKNOWN',  # Would need to lookup
                            token_address=mint,
                            action=action,
                            amount_sol=0,  # Would need to calculate
                            amount_tokens=abs(change),
                            price=0,  # Would need to calculate
                            timestamp=datetime.fromtimestamp(block_time),
                            confidence=confidence
                        )
        
        except Exception as e:
            print(f"❌ Error parsing transaction: {e}")
        
        return None
    
    async def monitor_wallets(self, callback=None):
        """Monitor whale wallets in real-time for new trades"""
        print("🐋 Starting whale wallet monitoring...")
        self.running = True
        
        # Verify wallets first
        print("\n📊 Verifying whale wallets...")
        for wallet in self.whale_wallets:
            stats = await self.verify_wallet(wallet)
            self.wallet_stats[wallet] = stats
            
            if stats.get('is_verified'):
                print(f"✅ {wallet[:8]}... - VERIFIED WHALE")
                print(f"   Balance: {stats['balance_sol']:.2f} SOL")
                print(f"   Transactions: {stats['recent_transactions']}")
                print(f"   Confidence: {stats['confidence']:.0%}")
            else:
                print(f"⚠️ {wallet[:8]}... - Not active/verified")
                print(f"   Balance: {stats.get('balance_sol', 0):.2f} SOL")
                print(f"   Issue: {stats.get('error', 'Low activity or balance')}")
        
        # Start monitoring loop
        while self.running:
            try:
                for wallet in self.whale_wallets:
                    if not self.wallet_stats.get(wallet, {}).get('is_verified'):
                        continue  # Skip unverified wallets
                    
                    # Check for new trades
                    recent_trades = await self.get_recent_trades(wallet, limit=5)
                    
                    for signal in recent_trades:
                        # Check if this is a new signal
                        if not self._is_duplicate_signal(signal):
                            self.signals.append(signal)
                            print(f"\n🐋 WHALE SIGNAL DETECTED!")
                            print(f"   Wallet: {signal.whale_wallet[:8]}...")
                            print(f"   Action: {signal.action}")
                            print(f"   Token: {signal.token_address[:8]}...")
                            print(f"   Amount: {signal.amount_tokens:.2f}")
                            print(f"   Confidence: {signal.confidence:.0%}")
                            
                            if callback:
                                await callback(signal)
                
                # Wait before next check (poll every 10 seconds)
                await asyncio.sleep(10)
            
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def _is_duplicate_signal(self, signal: WhaleSignal) -> bool:
        """Check if we've already processed this signal"""
        # Check last 100 signals
        for existing in self.signals[-100:]:
            if (existing.whale_wallet == signal.whale_wallet and
                existing.token_address == signal.token_address and
                existing.action == signal.action and
                abs((existing.timestamp - signal.timestamp).total_seconds()) < 60):
                return True
        return False
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        print("🛑 Whale monitoring stopped")

# Standalone verification tool
async def verify_whale_wallets(wallet_addresses: List[str]):
    """Verify if wallet addresses are active whales"""
    tracker = WhaleTracker(wallet_addresses)
    
    print("🔍 Verifying Whale Wallets\n")
    print("=" * 80)
    
    for wallet in wallet_addresses:
        print(f"\n📍 Checking: {wallet}")
        stats = await tracker.verify_wallet(wallet)
        
        print(f"\n{'✅ VERIFIED' if stats.get('is_verified') else '❌ NOT VERIFIED'}")
        print(f"Balance: {stats.get('balance_sol', 0):.4f} SOL")
        print(f"Recent Transactions: {stats.get('recent_transactions', 0)}")
        print(f"Hours Since Last Trade: {stats.get('hours_since_last_tx', 999):.1f}")
        print(f"Confidence Score: {stats.get('confidence', 0):.0%}")
        
        if stats.get('is_verified'):
            print(f"\n🐋 Fetching recent trades...")
            trades = await tracker.get_recent_trades(wallet, limit=10)
            print(f"Found {len(trades)} recent token trades")
            
            if trades:
                print("\nMost Recent Trades:")
                for i, trade in enumerate(trades[:5], 1):
                    print(f"  {i}. {trade.action} {trade.token_address[:8]}... at {trade.timestamp}")
        
        print("-" * 80)

if __name__ == "__main__":
    # Verify the provided whale wallets
    whale_addresses = [
        'Ad7CwwXixx1MAFMCcoF4krxbJRyejjyAgNJv4iaKZVCq',
        'JCRGumoE9Qi5BBgULTgdgTLjSgkCMSbF62ZZfGs84JeU'
    ]
    
    print("🐋 Whale Wallet Verification Tool\n")
    asyncio.run(verify_whale_wallets(whale_addresses))
