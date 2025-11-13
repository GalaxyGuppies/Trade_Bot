"""
Enhanced Multi-Chain Wallet Balance Detection
Supports Ethereum, Solana, and other major networks
"""

import asyncio
import logging
import requests
import json
from typing import Dict, Optional, List, Union
from dataclasses import dataclass
from datetime import datetime
import aiohttp
import re

logger = logging.getLogger(__name__)

@dataclass
class WalletBalance:
    """Wallet balance information"""
    address: str
    chain: str
    native_balance: float  # ETH, SOL, etc.
    usd_balance: float
    token_balances: Dict[str, float]
    last_updated: datetime

class MultiChainWalletDetector:
    """
    Enhanced wallet detector supporting multiple blockchains
    """
    
    def __init__(self, wallet_addresses: Union[str, Dict[str, str]]):
        """
        Initialize with wallet addresses
        
        Args:
            wallet_addresses: Either a single address string or dict like:
                            {'ethereum': '0x...', 'solana': 'abc123...'}
        """
        if isinstance(wallet_addresses, str):
            # Auto-detect chain based on address format
            self.addresses = self._auto_detect_chain(wallet_addresses)
        else:
            self.addresses = wallet_addresses
        
        self.balances = {}
        
        # API endpoints
        self.endpoints = {
            'ethereum': {
                'etherscan': 'https://api.etherscan.io/api',
                'coingecko': 'https://api.coingecko.com/api/v3'
            },
            'solana': {
                'rpc': 'https://api.mainnet-beta.solana.com',
                'coingecko': 'https://api.coingecko.com/api/v3'
            }
        }
        
        # Price cache
        self.token_prices = {}
        
        logger.info(f"🔍 Multi-chain wallet detector initialized for: {self.addresses}")
    
    def _auto_detect_chain(self, address: str) -> Dict[str, str]:
        """Auto-detect blockchain based on address format"""
        addresses = {}
        
        # Ethereum: starts with 0x, 42 characters
        if re.match(r'^0x[a-fA-F0-9]{40}$', address):
            addresses['ethereum'] = address
            logger.info(f"✅ Detected Ethereum address: {address}")
        
        # Solana: base58, typically 32-44 characters, no 0x prefix
        elif re.match(r'^[1-9A-HJ-NP-Za-km-z]{32,44}$', address) and not address.startswith('0x'):
            addresses['solana'] = address
            logger.info(f"✅ Detected Solana address: {address}")
        
        else:
            # Default to Ethereum for now
            addresses['ethereum'] = address
            logger.warning(f"⚠️ Unknown address format, defaulting to Ethereum: {address}")
        
        return addresses
    
    async def get_ethereum_balance(self, address: str) -> WalletBalance:
        """Get Ethereum balance"""
        try:
            logger.info(f"🔍 Checking Ethereum balance for {address[:10]}...")
            
            # Get ETH balance
            eth_balance = await self._get_eth_balance_etherscan(address)
            if eth_balance is None:
                eth_balance = 0.0
            
            # Get ETH price
            eth_price = await self._get_token_price('ethereum')
            usd_balance = eth_balance * eth_price
            
            # Get token balances (simplified for now)
            token_balances = {}
            
            balance = WalletBalance(
                address=address,
                chain='ethereum',
                native_balance=eth_balance,
                usd_balance=usd_balance,
                token_balances=token_balances,
                last_updated=datetime.now()
            )
            
            logger.info(f"✅ Ethereum: {eth_balance:.4f} ETH (~${usd_balance:,.2f})")
            return balance
            
        except Exception as e:
            logger.error(f"Error getting Ethereum balance: {e}")
            return WalletBalance(
                address=address,
                chain='ethereum',
                native_balance=0.0,
                usd_balance=0.0,
                token_balances={},
                last_updated=datetime.now()
            )
    
    async def get_solana_balance(self, address: str) -> WalletBalance:
        """Get Solana balance"""
        try:
            logger.info(f"🔍 Checking Solana balance for {address[:10]}...")
            
            # Get SOL balance using RPC
            sol_balance = await self._get_sol_balance_rpc(address)
            if sol_balance is None:
                sol_balance = 0.0
            
            # Get SOL price
            sol_price = await self._get_token_price('solana')
            usd_balance = sol_balance * sol_price
            
            # Get SPL token balances (simplified for now)
            token_balances = {}
            
            balance = WalletBalance(
                address=address,
                chain='solana',
                native_balance=sol_balance,
                usd_balance=usd_balance,
                token_balances=token_balances,
                last_updated=datetime.now()
            )
            
            logger.info(f"✅ Solana: {sol_balance:.4f} SOL (~${usd_balance:,.2f})")
            return balance
            
        except Exception as e:
            logger.error(f"Error getting Solana balance: {e}")
            return WalletBalance(
                address=address,
                chain='solana',
                native_balance=0.0,
                usd_balance=0.0,
                token_balances={},
                last_updated=datetime.now()
            )
    
    async def _get_eth_balance_etherscan(self, address: str) -> Optional[float]:
        """Get ETH balance using Etherscan API"""
        try:
            url = f"{self.endpoints['ethereum']['etherscan']}"
            params = {
                'module': 'account',
                'action': 'balance',
                'address': address,
                'tag': 'latest',
                'apikey': 'YourApiKeyToken'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == '1':
                            balance_wei = int(data['result'])
                            balance_eth = balance_wei / 10**18
                            return balance_eth
            
            return None
            
        except Exception as e:
            logger.warning(f"Etherscan balance check failed: {e}")
            return None
    
    async def _get_sol_balance_rpc(self, address: str) -> Optional[float]:
        """Get SOL balance using Solana RPC"""
        try:
            url = self.endpoints['solana']['rpc']
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [address]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'result' in data and 'value' in data['result']:
                            balance_lamports = data['result']['value']
                            balance_sol = balance_lamports / 10**9  # Convert lamports to SOL
                            return balance_sol
            
            return None
            
        except Exception as e:
            logger.warning(f"Solana RPC balance check failed: {e}")
            return None
    
    async def _get_token_price(self, token_id: str) -> float:
        """Get current token price from CoinGecko"""
        try:
            if token_id in self.token_prices:
                return self.token_prices[token_id]
            
            url = f"{self.endpoints['ethereum']['coingecko']}/simple/price"
            params = {
                'ids': token_id,
                'vs_currencies': 'usd'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if token_id in data:
                            price = data[token_id]['usd']
                            self.token_prices[token_id] = price
                            return price
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get price for {token_id}: {e}")
            return 0.0
    
    async def get_all_balances(self) -> Dict[str, WalletBalance]:
        """Get balances across all configured chains"""
        try:
            logger.info(f"🔍 Checking multi-chain wallet balances...")
            
            balances = {}
            
            # Check Ethereum if configured
            if 'ethereum' in self.addresses:
                eth_balance = await self.get_ethereum_balance(self.addresses['ethereum'])
                balances['ethereum'] = eth_balance
            
            # Check Solana if configured
            if 'solana' in self.addresses:
                sol_balance = await self.get_solana_balance(self.addresses['solana'])
                balances['solana'] = sol_balance
            
            total_usd = sum(balance.usd_balance for balance in balances.values())
            
            logger.info(f"💰 Total portfolio value: ${total_usd:,.2f}")
            
            return balances
            
        except Exception as e:
            logger.error(f"Error getting all balances: {e}")
            return {}
    
    async def get_total_portfolio_value(self) -> float:
        """Get total portfolio value in USD"""
        try:
            balances = await self.get_all_balances()
            total_value = sum(balance.usd_balance for balance in balances.values())
            
            logger.info(f"📊 Total Portfolio Value: ${total_value:,.2f}")
            return total_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0

# Demo function for multi-chain detection
async def demo_multichain_detection():
    """Demo the enhanced multi-chain wallet detection"""
    
    print("🔍 ENHANCED MULTI-CHAIN WALLET DETECTION")
    print("=" * 50)
    
    # Test addresses (replace with actual addresses)
    test_addresses = {
        'ethereum': "0xb4add0df12df32981773ca25ee88bdab750bfa20",
        # If you have a Solana address, add it like:
        # 'solana': "YourSolanaAddressHere"
    }
    
    # Auto-detect mode (single address)
    print("🔧 Testing auto-detection mode...")
    eth_address = "0xb4add0df12df32981773ca25ee88bdab750bfa20"
    detector = MultiChainWalletDetector(eth_address)
    
    try:
        # Get all balances
        total_value = await detector.get_total_portfolio_value()
        
        print(f"\n💰 RESULTS:")
        print(f"   Total Portfolio: ${total_value:,.2f}")
        
        if total_value > 0:
            print(f"\n✅ SUCCESS: Detected wallet balance!")
            print(f"   System can now use ${total_value:,.2f} as trading capital")
        else:
            print(f"\n📝 INFO: No balance detected on configured networks")
            print(f"   This could mean:")
            print(f"   - Funds are on different network (BSC, Polygon, etc.)")
            print(f"   - Solana address needed if you added SOL")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(demo_multichain_detection())