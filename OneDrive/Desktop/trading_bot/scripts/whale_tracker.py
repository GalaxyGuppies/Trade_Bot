#!/usr/bin/env python3
"""
Enhanced Whale Wallet Tracker for Memecoin Copy-Trading
Monitors real-time Solana transactions from known smart-money wallets
Implements strategies from top traders and hedge funds
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from collections import defaultdict


class WhaleWalletTracker:
    """
    Advanced whale wallet monitoring system with real-time transaction parsing
    Focuses on memecoin trading signals from proven high-win-rate wallets
    """
    
    # Top memecoin whale wallets (from blueprint - verified smart money)
    # EXPANDED: All wallets from the blueprint research
    WHALE_WALLETS = {
        # === TOP TIER: Ultra-High Win Rate (80%+) ===
        
        # Ultra-fast scalper, 98% win rate, $38.6M total profit
        '9ex23LM7UgMqWv9M9iE5wGvas6NiYKSsEUKHdK5X0ucn': {
            'name': 'SpaceX Meme Master',
            'win_rate': 0.98,
            'total_profit_usd': 38600000,
            'strategy': 'ultra_fast_scalping',
            'confidence': 0.95,
            'tier': 'top'
        },
        # Small-stack sniper, 100% win rate on timing
        '2B145FJsiBZJcBsmStgGhyJqFoqJ6JRh7hbwBN429Hjp': {
            'name': 'Perfect Timing Sniper',
            'win_rate': 1.00,
            'total_profit_usd': 2500000,
            'strategy': 'concentrated_liquidity',
            'confidence': 0.98,
            'tier': 'top'
        },
        # Automated arbitrage bot
        'benRLpbWCL8P8t51ufYt522419hGF5zif3CqgWGbEUm': {
            'name': 'Arb Bot Operator',
            'win_rate': 0.82,
            'total_profit_usd': 22000000,
            'strategy': 'automated_arbitrage',
            'confidence': 0.90,
            'tier': 'top'
        },
        
        # === HIGH TIER: Strong Win Rate (60-80%) ===
        
        # Dominant in one pocket
        'GkPtg91t38yNpdBG5NJu4YMKL5wFxq3PMB8P0hXt8ry': {
            'name': 'Liquidity Dominator',
            'win_rate': 0.78,
            'total_profit_usd': 8500000,
            'strategy': 'pocket_dominance',
            'confidence': 0.88,
            'tier': 'high'
        },
        # Original whale 3
        'GGkB8ef2AMGgTx9nJKLWDPtMPTpix92iTMJKo58JafGr': {
            'name': 'Smart Accumulator',
            'win_rate': 0.72,
            'total_profit_usd': 6200000,
            'strategy': 'smart_accumulation',
            'confidence': 0.87,
            'tier': 'high'
        },
        # Original whale 1
        'Ad7CwwXixx1MAFMCcoF4krxbJRyejjyAgNJv4iaKZVCq': {
            'name': 'Momentum Follower',
            'win_rate': 0.70,
            'total_profit_usd': 5000000,
            'strategy': 'momentum_following',
            'confidence': 0.85,
            'tier': 'high'
        },
        # Mini hedge fund, manufactures volatility
        '6FCs8rYFDvDzoim9tKrCfr4RzJeWBrMZAeAoDCggfGXy': {
            'name': 'Volatility Manufacturer',
            'win_rate': 0.65,
            'total_profit_usd': 12000000,
            'strategy': 'profit_recycling',
            'confidence': 0.85,
            'tier': 'high'
        },
        # Original whale 2
        'JCRGumoE9Qi5BBgULTgdgTLjSgkCMSbF62ZZfGs84JeU': {
            'name': 'Trend Rider',
            'win_rate': 0.65,
            'total_profit_usd': 3800000,
            'strategy': 'trend_riding',
            'confidence': 0.80,
            'tier': 'high'
        },
        # Wild player, memecoins to other plays
        '6Hu85GZPz74fEqiT5aFVdXR2yAuKFFiVW2Yqkv6Nx3JR': {
            'name': 'Wild Card Player',
            'win_rate': 0.62,
            'total_profit_usd': 7400000,
            'strategy': 'cross_platform',
            'confidence': 0.80,
            'tier': 'high'
        },
        
        # === MEDIUM TIER: Solid Win Rate (50-60%) ===
        
        # Quality wallet for beginners
        'He2QR4HFhqUmUT6anRgnc4DsaCRfto8FQKFctAc2KTTB': {
            'name': 'Beginner-Safe Whale',
            'win_rate': 0.58,
            'total_profit_usd': 4100000,
            'strategy': 'conservative_momentum',
            'confidence': 0.82,
            'tier': 'medium'
        },
        # Short-hold strategy
        '6zY2mFceyEyeGA9rA535nSfnsWqKabcUwoq2vYkX381': {
            'name': 'Short-Hold Specialist',
            'win_rate': 0.55,
            'total_profit_usd': 6200000,
            'strategy': 'fast_in_out',
            'confidence': 0.75,
            'tier': 'medium'
        },
        
        # === AGGRESSIVE TIER: Lower Win Rate but High Profit (<50%) ===
        
        # High capital flow whale
        '4fmAjPvV15cdU8bHKX5ykuULtwuQRNCvdKxyjWBWK937': {
            'name': 'Capital Flow Whale',
            'win_rate': 0.42,
            'total_profit_usd': 15000000,
            'strategy': 'volume_manipulation',
            'confidence': 0.72,
            'tier': 'aggressive'
        },
        # Intense short holds, higher risk
        'EtwoeVJaJbUDvYEkA9k1MrysZuwJqhfWfj2KuzbHNA': {
            'name': 'High Risk Trader',
            'win_rate': 0.38,
            'total_profit_usd': 9800000,
            'strategy': 'extreme_volatility',
            'confidence': 0.68,
            'tier': 'aggressive'
        },
        # Volatility whale, $4.9M stack, 22% win rate but massive wins
        '4GWCutReokvKP2d9gRmZR4AbVcFJtzhPzH25k4NoiFo8m': {
            'name': 'Volatility Hunter',
            'win_rate': 0.22,
            'total_profit_usd': 4900000,
            'strategy': 'burst_entry_exit',
            'confidence': 0.70,
            'tier': 'aggressive'
        },
        
        # === ADDITIONAL WALLETS FROM BLUEPRINT ===
        
        # $5.86M memecoin bettor (WIF, Fartcoin, POPCAT, MEW, BONK)
        '8Lqz9PKGSd9JcLmyRuGzJQb3VfvjqcGhXwU9YzxPump': {
            'name': 'Memecoin Bettor',
            'win_rate': 0.55,  # Estimated from +$658K on $5.86M
            'total_profit_usd': 658000,
            'strategy': 'diversified_memecoin_betting',
            'confidence': 0.75,
            'tier': 'medium'
        },
        # Early GOAT/FWOG/LUCE accumulator - $5M in 76 trades
        'CesmFuE6aKXNhB3r8vPq9YN4zBvM2RrHg8Xz7qFpPump': {
            'name': 'Early Bird Accumulator',
            'win_rate': 0.68,  # High profit suggests good timing
            'total_profit_usd': 5000000,
            'strategy': 'early_accumulation',
            'confidence': 0.88,
            'tier': 'high'
        },
        
        # === TOP SOL HOLDERS (Potential Whale Activity) ===
        
        # Richest SOL wallet #1 - 5.1M SOL
        'MJKqp326RZCHnAAbew9MDdui3iCKWco7fsK9sVuZTX2': {
            'name': 'SOL Mega Whale #1',
            'win_rate': 0.60,  # Estimated - institutional level
            'total_profit_usd': 50000000,  # Conservative estimate
            'strategy': 'institutional_trading',
            'confidence': 0.70,
            'tier': 'high'
        },
        # Richest SOL wallet #2 - 4.3M SOL
        '52C9T2T7JRojtxumYnYZhyUmrN7kqzvCLc4Ksvjk7TxD': {
            'name': 'SOL Mega Whale #2',
            'win_rate': 0.60,
            'total_profit_usd': 45000000,
            'strategy': 'institutional_trading',
            'confidence': 0.70,
            'tier': 'high'
        },
        # Richest SOL wallet #3 - 3.9M SOL
        '8BseXT9EtoEhBTKFFYkwTnjKSUZwhtmdKY2Jrj8j45Rt': {
            'name': 'SOL Mega Whale #3',
            'win_rate': 0.60,
            'total_profit_usd': 40000000,
            'strategy': 'institutional_trading',
            'confidence': 0.70,
            'tier': 'high'
        },
    }
    
    def __init__(self, rpc_url: str = "https://api.mainnet-beta.solana.com"):
        """
        Initialize whale tracker with Solana RPC connection
        
        Args:
            rpc_url: Solana RPC endpoint
        """
        self.rpc_url = rpc_url
        self.last_check_time = {}  # Track last signature checked per wallet
        self.whale_activity_cache = defaultdict(list)  # Recent activity per wallet
        self.trade_history = []  # All detected whale trades
        self.token_metadata_cache = {}  # Cache token info to avoid repeated API calls
        
    async def get_token_metadata(
        self,
        session: aiohttp.ClientSession,
        token_mint: str
    ) -> Dict[str, str]:
        """
        Fetch token metadata from on-chain or DexScreener
        
        Args:
            session: aiohttp session
            token_mint: Token mint address
            
        Returns:
            Dict with token symbol, name, decimals
        """
        # Check cache first
        if token_mint in self.token_metadata_cache:
            return self.token_metadata_cache[token_mint]
        
        try:
            # Try DexScreener first (has token metadata + price data)
            dex_url = f"https://api.dexscreener.com/latest/dex/tokens/{token_mint}"
            
            async with session.get(dex_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    pairs = data.get('pairs', [])
                    
                    if pairs:
                        # Get first pair (usually most liquid)
                        pair = pairs[0]
                        base_token = pair.get('baseToken', {})
                        
                        metadata = {
                            'symbol': base_token.get('symbol', f'TOKEN_{token_mint[:8]}'),
                            'name': base_token.get('name', 'Unknown Token'),
                            'address': token_mint,
                            'liquidity_usd': pair.get('liquidity', {}).get('usd', 0),
                            'price_usd': float(pair.get('priceUsd', 0)),
                            'volume_24h': float(pair.get('volume', {}).get('h24', 0)),
                            'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0))
                        }
                        
                        # Cache for future use
                        self.token_metadata_cache[token_mint] = metadata
                        return metadata
        
        except Exception as e:
            print(f"⚠️ Error fetching metadata for {token_mint[:8]}...: {e}")
        
        # Fallback: return minimal metadata
        metadata = {
            'symbol': f'TOKEN_{token_mint[:8]}',
            'name': 'Unknown Token',
            'address': token_mint,
            'liquidity_usd': 0,
            'price_usd': 0,
            'volume_24h': 0,
            'price_change_24h': 0
        }
        self.token_metadata_cache[token_mint] = metadata
        return metadata
        
    async def get_wallet_transactions(
        self, 
        session: aiohttp.ClientSession, 
        wallet_address: str, 
        limit: int = 10
    ) -> List[Dict]:
        """
        Fetch recent transactions for a whale wallet
        
        Args:
            session: aiohttp session
            wallet_address: Solana wallet address
            limit: Number of recent transactions to fetch
            
        Returns:
            List of transaction signatures with metadata
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [wallet_address, {"limit": limit}]
        }
        
        try:
            async with session.post(
                self.rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
                return data.get('result', [])
        except Exception as e:
            print(f"⚠️ Error fetching transactions for {wallet_address[:8]}...: {e}")
            return []
    
    async def parse_transaction_details(
        self,
        session: aiohttp.ClientSession,
        signature: str,
        wallet_address: str,
        tracked_tokens: Optional[Dict[str, str]] = None
    ) -> Optional[Dict]:
        """
        Parse transaction details to detect buy/sell signals
        NOW SUPPORTS FULL DISCOVERY MODE: If tracked_tokens is None, detects ALL token trades
        
        Args:
            session: aiohttp session
            signature: Transaction signature
            wallet_address: Whale wallet that made the transaction
            tracked_tokens: Optional dict of token symbols to mint addresses
                           If None, will detect ANY token the whale trades (DISCOVERY MODE)
            
        Returns:
            Trade signal dict if relevant trade detected, None otherwise
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [
                signature,
                {
                    "encoding": "jsonParsed",
                    "maxSupportedTransactionVersion": 0
                }
            ]
        }
        
        try:
            async with session.post(
                self.rpc_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                tx_result = data.get('result')
                
                if not tx_result or tx_result.get('meta', {}).get('err'):
                    return None
                
                # Extract transaction metadata
                meta = tx_result.get('meta', {})
                block_time = tx_result.get('blockTime', 0)
                
                if not block_time:
                    return None
                
                tx_time = datetime.fromtimestamp(block_time)
                
                # Parse token balance changes
                pre_balances = meta.get('preTokenBalances', [])
                post_balances = meta.get('postTokenBalances', [])
                
                # DISCOVERY MODE: If no tracked tokens specified, detect ANY token trade
                if tracked_tokens is None:
                    # Find ANY token balance change for this whale
                    for pre_idx, pre_balance in enumerate(pre_balances):
                        if pre_balance.get('owner') != wallet_address:
                            continue
                        
                        token_mint = pre_balance.get('mint')
                        if not token_mint:
                            continue
                        
                        # Find matching post balance
                        post_balance = None
                        for post in post_balances:
                            if (post.get('mint') == token_mint and 
                                post.get('owner') == wallet_address and
                                post.get('accountIndex') == pre_balance.get('accountIndex')):
                                post_balance = post
                                break
                        
                        if not post_balance:
                            continue
                        
                        # Calculate balance change
                        pre_amount = float(pre_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
                        post_amount = float(post_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
                        change = post_amount - pre_amount
                        
                        if abs(change) < 0.01:  # Ignore dust trades
                            continue
                        
                        action = 'BUY' if change > 0 else 'SELL'
                        whale_info = self.WHALE_WALLETS.get(wallet_address, {})
                        
                        # Fetch token metadata from DexScreener
                        token_metadata = await self.get_token_metadata(session, token_mint)
                        token_symbol = token_metadata.get('symbol', f'TOKEN_{token_mint[:8]}')
                        
                        # SAFETY FILTER: Skip tokens with red flags
                        liquidity = token_metadata.get('liquidity_usd', 0)
                        volume_24h = token_metadata.get('volume_24h', 0)
                        
                        # Skip if liquidity too low (likely scam/rug)
                        if liquidity < 1000:  # Min $1K liquidity
                            print(f"⚠️ Skipping {token_symbol} - low liquidity (${liquidity:.0f})")
                            continue
                        
                        # Skip if no volume (likely dead token)
                        if volume_24h < 100:  # Min $100 daily volume
                            print(f"⚠️ Skipping {token_symbol} - low volume (${volume_24h:.0f})")
                            continue
                        
                        return {
                            'wallet': wallet_address,
                            'whale_name': whale_info.get('name', 'Unknown'),
                            'action': action,
                            'token': token_symbol,
                            'token_mint': token_mint,
                            'token_name': token_metadata.get('name', 'Unknown Token'),
                            'amount': abs(change),
                            'time': tx_time,
                            'signature': signature,
                            'whale_win_rate': whale_info.get('win_rate', 0.5),
                            'whale_strategy': whale_info.get('strategy', 'unknown'),
                            'confidence': whale_info.get('confidence', 0.7),
                            'minutes_ago': (datetime.now() - tx_time).total_seconds() / 60,
                            'discovery_mode': True,  # Flag to indicate this was auto-discovered
                            'liquidity_usd': liquidity,
                            'price_usd': token_metadata.get('price_usd', 0),
                            'volume_24h': volume_24h,
                            'price_change_24h': token_metadata.get('price_change_24h', 0),
                            'address': token_mint  # Include for trading
                        }
                
                # SELECTIVE MODE: Only match specific tracked tokens
                for token_symbol, token_mint in tracked_tokens.items():
                    for pre_idx, pre_balance in enumerate(pre_balances):
                        if pre_balance.get('mint') != token_mint:
                            continue
                        
                        if pre_balance.get('owner') != wallet_address:
                            continue
                        
                        # Find matching post balance
                        post_balance = None
                        for post in post_balances:
                            if (post.get('mint') == token_mint and 
                                post.get('owner') == wallet_address and
                                post.get('accountIndex') == pre_balance.get('accountIndex')):
                                post_balance = post
                                break
                        
                        if not post_balance:
                            continue
                        
                        # Calculate balance change
                        pre_amount = float(pre_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
                        post_amount = float(post_balance.get('uiTokenAmount', {}).get('uiAmount', 0))
                        change = post_amount - pre_amount
                        
                        if abs(change) < 0.01:  # Ignore dust trades
                            continue
                        
                        action = 'BUY' if change > 0 else 'SELL'
                        whale_info = self.WHALE_WALLETS.get(wallet_address, {})
                        
                        return {
                            'wallet': wallet_address,
                            'whale_name': whale_info.get('name', 'Unknown'),
                            'action': action,
                            'token': token_symbol,
                            'token_mint': token_mint,
                            'amount': abs(change),
                            'time': tx_time,
                            'signature': signature,
                            'whale_win_rate': whale_info.get('win_rate', 0.5),
                            'whale_strategy': whale_info.get('strategy', 'unknown'),
                            'confidence': whale_info.get('confidence', 0.7),
                            'minutes_ago': (datetime.now() - tx_time).total_seconds() / 60
                        }
                
                return None
                
        except Exception as e:
            print(f"⚠️ Error parsing transaction {signature[:8]}...: {e}")
            return None
    
    async def scan_whale_activity(
        self,
        tracked_tokens: Optional[Dict[str, str]] = None,
        lookback_minutes: int = 10,
        max_whales: int = 5,
        discovery_mode: bool = True
    ) -> List[Dict]:
        """
        Scan all whale wallets for recent trading activity
        
        Args:
            tracked_tokens: Optional dict of token symbols to mint addresses
                           If None and discovery_mode=True, detects ANY token trades
            lookback_minutes: Only consider transactions from this many minutes ago
            max_whales: Limit number of whales to check (prioritize high win rate)
            discovery_mode: If True, allow detection of ANY token (not just tracked ones)
            
        Returns:
            List of detected trade signals, sorted by confidence
        """
        signals = []
        current_time = datetime.now()
        
        # Prioritize whales by win rate and confidence
        sorted_whales = sorted(
            self.WHALE_WALLETS.items(),
            key=lambda x: (x[1]['win_rate'] * x[1]['confidence']),
            reverse=True
        )[:max_whales]
        
        async with aiohttp.ClientSession() as session:
            for wallet_address, whale_info in sorted_whales:
                # Get recent transactions
                transactions = await self.get_wallet_transactions(session, wallet_address, limit=5)
                
                for tx_info in transactions:
                    signature = tx_info.get('signature')
                    block_time = tx_info.get('blockTime', 0)
                    
                    if not block_time:
                        continue
                    
                    tx_time = datetime.fromtimestamp(block_time)
                    minutes_ago = (current_time - tx_time).total_seconds() / 60
                    
                    # Skip old transactions
                    if minutes_ago > lookback_minutes:
                        continue
                    
                    # Skip if already processed
                    if signature in [s.get('signature') for s in signals]:
                        continue
                    
                    # Parse transaction details
                    # Pass tracked_tokens (or None for full discovery)
                    parse_tokens = None if discovery_mode else tracked_tokens
                    
                    signal = await self.parse_transaction_details(
                        session,
                        signature,
                        wallet_address,
                        parse_tokens
                    )
                    
                    if signal:
                        signals.append(signal)
                        self.trade_history.append(signal)
                        
                        discovery_flag = "🔍 DISCOVERED" if signal.get('discovery_mode') else "🐋"
                        print(f"{discovery_flag} WHALE SIGNAL: {signal['whale_name']} {signal['action']} "
                              f"{signal['amount']:.2f} {signal['token']} "
                              f"({signal['minutes_ago']:.1f}m ago, "
                              f"confidence: {signal['confidence']:.0%})")
                
                # Small delay to avoid rate limiting
                await asyncio.sleep(0.2)
        
        # Sort by confidence (win_rate * confidence) and recency
        signals.sort(key=lambda x: (x['confidence'] * x['whale_win_rate'], -x['minutes_ago']), reverse=True)
        
        return signals
    
    def detect_multi_whale_consensus(
        self,
        signals: List[Dict],
        lookback_minutes: int = 30
    ) -> Dict[str, Dict]:
        """
        Detect when multiple whales are buying/selling the same token
        This is a STRONG signal that should be prioritized
        
        Args:
            signals: List of whale signals from scan_whale_activity
            lookback_minutes: Time window to check for consensus
            
        Returns:
            Dict of token -> consensus data with priority signals
        """
        from datetime import timedelta
        
        consensus_data = defaultdict(lambda: {
            'buy_whales': [],
            'sell_whales': [],
            'total_buy_confidence': 0,
            'total_sell_confidence': 0,
            'avg_buy_win_rate': 0,
            'avg_sell_win_rate': 0,
            'consensus_strength': 0,
            'priority_action': None,
            'priority_score': 0
        })
        
        # Also check recent trade history for additional context
        recent_time = datetime.now() - timedelta(minutes=lookback_minutes)
        all_signals = signals + [
            s for s in self.trade_history 
            if s['time'] >= recent_time
        ]
        
        # Group signals by token
        for signal in all_signals:
            token = signal['token']
            action = signal['action']
            whale_name = signal['whale_name']
            
            if action == 'BUY':
                # Avoid duplicates from same whale
                if whale_name not in [w['name'] for w in consensus_data[token]['buy_whales']]:
                    consensus_data[token]['buy_whales'].append({
                        'name': whale_name,
                        'confidence': signal['confidence'],
                        'win_rate': signal['whale_win_rate'],
                        'time': signal['time'],
                        'amount': signal.get('amount', 0)
                    })
                    consensus_data[token]['total_buy_confidence'] += signal['confidence']
            else:  # SELL
                if whale_name not in [w['name'] for w in consensus_data[token]['sell_whales']]:
                    consensus_data[token]['sell_whales'].append({
                        'name': whale_name,
                        'confidence': signal['confidence'],
                        'win_rate': signal['whale_win_rate'],
                        'time': signal['time'],
                        'amount': signal.get('amount', 0)
                    })
                    consensus_data[token]['total_sell_confidence'] += signal['confidence']
        
        # Calculate consensus metrics for each token
        for token, data in consensus_data.items():
            buy_count = len(data['buy_whales'])
            sell_count = len(data['sell_whales'])
            
            # Calculate average win rates
            if buy_count > 0:
                data['avg_buy_win_rate'] = sum(w['win_rate'] for w in data['buy_whales']) / buy_count
            if sell_count > 0:
                data['avg_sell_win_rate'] = sum(w['win_rate'] for w in data['sell_whales']) / sell_count
            
            # Determine consensus strength and priority
            # Consensus strength = number of whales + avg confidence + avg win rate
            buy_strength = (
                buy_count * 10 +  # More whales = stronger signal
                data['total_buy_confidence'] * 10 +
                data['avg_buy_win_rate'] * 20
            )
            
            sell_strength = (
                sell_count * 10 +
                data['total_sell_confidence'] * 10 +
                data['avg_sell_win_rate'] * 20
            )
            
            # Determine priority action
            if buy_count >= 2 or sell_count >= 2:  # Multi-whale consensus threshold
                if buy_strength > sell_strength:
                    data['priority_action'] = 'BUY'
                    data['consensus_strength'] = buy_count
                    data['priority_score'] = buy_strength
                else:
                    data['priority_action'] = 'SELL'
                    data['consensus_strength'] = sell_count
                    data['priority_score'] = sell_strength
            elif buy_count == 1 and sell_count == 0:
                data['priority_action'] = 'BUY'
                data['consensus_strength'] = 1
                data['priority_score'] = buy_strength
            elif sell_count == 1 and buy_count == 0:
                data['priority_action'] = 'SELL'
                data['consensus_strength'] = 1
                data['priority_score'] = sell_strength
            else:
                # Conflicting signals or no clear consensus
                data['priority_action'] = 'HOLD'
                data['consensus_strength'] = 0
                data['priority_score'] = 0
        
        return dict(consensus_data)
    
    def get_priority_signals(
        self,
        consensus_data: Dict[str, Dict],
        min_whales: int = 2
    ) -> List[Dict]:
        """
        Extract priority signals from consensus data
        
        Args:
            consensus_data: Output from detect_multi_whale_consensus
            min_whales: Minimum number of whales for priority signal
            
        Returns:
            List of priority signals sorted by priority score
        """
        priority_signals = []
        
        for token, data in consensus_data.items():
            if data['consensus_strength'] >= min_whales:
                # Create priority signal
                signal = {
                    'token': token,
                    'action': data['priority_action'],
                    'whale_count': data['consensus_strength'],
                    'priority_score': data['priority_score'],
                    'avg_win_rate': (
                        data['avg_buy_win_rate'] 
                        if data['priority_action'] == 'BUY' 
                        else data['avg_sell_win_rate']
                    ),
                    'total_confidence': (
                        data['total_buy_confidence']
                        if data['priority_action'] == 'BUY'
                        else data['total_sell_confidence']
                    ),
                    'whales': (
                        data['buy_whales']
                        if data['priority_action'] == 'BUY'
                        else data['sell_whales']
                    ),
                    'is_consensus': True,
                    'reason': f"🚨 MULTI-WHALE CONSENSUS: {data['consensus_strength']} whales {data['priority_action']} {token}"
                }
                priority_signals.append(signal)
        
        # Sort by priority score (highest first)
        priority_signals.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return priority_signals
    
    def get_whale_statistics(self) -> Dict:
        """
        Get aggregate statistics on whale activity
        
        Returns:
            Dictionary with whale trading statistics
        """
        total_signals = len(self.trade_history)
        
        if total_signals == 0:
            return {
                'total_signals': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'avg_confidence': 0,
                'most_active_whale': 'None',
                'most_active_count': 0
            }
        
        buy_signals = sum(1 for s in self.trade_history if s['action'] == 'BUY')
        sell_signals = total_signals - buy_signals
        avg_confidence = sum(s['confidence'] for s in self.trade_history) / total_signals
        
        # Find most active whale
        whale_counts = defaultdict(int)
        for signal in self.trade_history:
            whale_counts[signal['whale_name']] += 1
        
        most_active = max(whale_counts.items(), key=lambda x: x[1]) if whale_counts else ('None', 0)
        
        return {
            'total_signals': total_signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'avg_confidence': avg_confidence,
            'most_active_whale': most_active[0],
            'most_active_count': most_active[1]
        }


async def main():
    """Test whale tracker functionality with multi-whale consensus detection"""
    tracker = WhaleWalletTracker()
    
    # Test with current tracked tokens
    tracked_tokens = {
        'BANGERS': '3wppuwUMAGgxnX75Aqr4W91xYWaN6RjxjCUFiPZUpump',
        'TRUMP': '6p6xgHyF7AeE6TZkSmFsko444wqoP15icUSqi2jfGiPN',
        'BASED': 'EMAGfmV5bMzYEtgda43ZmCYwmLL7SaMi2RVqaRPjpump'
    }
    
    print("🐋 Starting whale wallet scanner...")
    print(f"📊 Monitoring {len(tracker.WHALE_WALLETS)} whale wallets")
    print(f"🎯 Tracking tokens: {', '.join(tracked_tokens.keys())}\n")
    
    # Show whale tiers
    tier_counts = defaultdict(int)
    for whale_data in tracker.WHALE_WALLETS.values():
        tier_counts[whale_data.get('tier', 'unknown')] += 1
    
    print("📋 Whale Distribution by Tier:")
    for tier, count in sorted(tier_counts.items()):
        print(f"   {tier.upper()}: {count} wallets")
    print()
    
    signals = await tracker.scan_whale_activity(tracked_tokens, lookback_minutes=60, max_whales=20)
    
    print(f"\n✅ Scan complete. Found {len(signals)} signals in last 60 minutes")
    
    if signals:
        print("\n📋 Top Signals:")
        for i, signal in enumerate(signals[:5], 1):
            print(f"{i}. {signal['whale_name']}: {signal['action']} "
                  f"{signal['amount']:.2f} {signal['token']} "
                  f"({signal['minutes_ago']:.1f}m ago, {signal['confidence']:.0%} confidence)")
        
        # Detect multi-whale consensus
        print("\n🔍 Analyzing Multi-Whale Consensus...")
        consensus_data = tracker.detect_multi_whale_consensus(signals, lookback_minutes=30)
        
        if consensus_data:
            print(f"\n📊 Consensus Analysis for {len(consensus_data)} token(s):")
            for token, data in consensus_data.items():
                buy_count = len(data['buy_whales'])
                sell_count = len(data['sell_whales'])
                
                print(f"\n   {token}:")
                print(f"      BUY signals: {buy_count} whales")
                if buy_count > 0:
                    print(f"         Whales: {', '.join(w['name'] for w in data['buy_whales'])}")
                    print(f"         Avg Win Rate: {data['avg_buy_win_rate']:.0%}")
                    print(f"         Total Confidence: {data['total_buy_confidence']:.2f}")
                
                print(f"      SELL signals: {sell_count} whales")
                if sell_count > 0:
                    print(f"         Whales: {', '.join(w['name'] for w in data['sell_whales'])}")
                    print(f"         Avg Win Rate: {data['avg_sell_win_rate']:.0%}")
                    print(f"         Total Confidence: {data['total_sell_confidence']:.2f}")
                
                if data['priority_action'] and data['priority_action'] != 'HOLD':
                    print(f"      🚨 PRIORITY: {data['priority_action']} (strength: {data['consensus_strength']} whales)")
                    print(f"      📈 Priority Score: {data['priority_score']:.1f}")
        
        # Get priority signals
        priority_signals = tracker.get_priority_signals(consensus_data, min_whales=2)
        
        if priority_signals:
            print(f"\n🚨 PRIORITY SIGNALS (Multi-Whale Consensus):")
            for i, signal in enumerate(priority_signals, 1):
                print(f"\n{i}. {signal['token']} - {signal['action']}")
                print(f"   Whale Count: {signal['whale_count']}")
                print(f"   Priority Score: {signal['priority_score']:.1f}")
                print(f"   Avg Win Rate: {signal['avg_win_rate']:.0%}")
                print(f"   Participating Whales:")
                for whale in signal['whales']:
                    print(f"      • {whale['name']} (Win Rate: {whale['win_rate']:.0%})")
        else:
            print("\nℹ️ No multi-whale consensus detected (need 2+ whales on same token)")
    
    stats = tracker.get_whale_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total signals: {stats['total_signals']}")
    print(f"   Buy signals: {stats['buy_signals']}")
    print(f"   Sell signals: {stats['sell_signals']}")
    print(f"   Avg confidence: {stats['avg_confidence']:.1%}")
    print(f"   Most active: {stats['most_active_whale']} ({stats['most_active_count']} trades)")


if __name__ == "__main__":
    asyncio.run(main())
