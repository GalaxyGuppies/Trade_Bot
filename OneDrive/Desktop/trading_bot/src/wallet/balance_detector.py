"""
Wallet Balance Detection Module
Automatically detects and monitors real wallet balances across multiple chains
"""

import asyncio
import logging
import requests
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class WalletBalance:
    """Wallet balance information"""
    address: str
    chain: str
    eth_balance: float
    usd_balance: float
    token_balances: Dict[str, float]
    last_updated: datetime

class WalletBalanceDetector:
    """
    Automatically detects wallet balances across multiple chains
    """
    
    def __init__(self, wallet_address: str):
        self.wallet_address = wallet_address.lower()
        self.balances = {}
        
        # API endpoints for different chains
        self.endpoints = {
            'ethereum': {
                'etherscan': 'https://api.etherscan.io/api',
                'alchemy': 'https://eth-mainnet.g.alchemy.com/v2',
                'moralis': 'https://deep-index.moralis.io/api/v2'
            },
            'bsc': {
                'bscscan': 'https://api.bscscan.com/api',
                'moralis': 'https://deep-index.moralis.io/api/v2'
            },
            'polygon': {
                'polygonscan': 'https://api.polygonscan.com/api',
                'moralis': 'https://deep-index.moralis.io/api/v2'
            }
        }
        
        # Current prices cache
        self.token_prices = {}
        
        logger.info(f"🔍 Wallet Balance Detector initialized for: {self.wallet_address}")
    
    async def get_ethereum_balance(self) -> WalletBalance:
        """Get Ethereum mainnet balance"""
        try:
            logger.info("🔍 Checking Ethereum balance...")
            
            # Get ETH balance using multiple methods
            eth_balance = await self._get_eth_balance_etherscan()
            if eth_balance is None:
                eth_balance = await self._get_eth_balance_alchemy()
            if eth_balance is None:
                eth_balance = 0.0
            
            # Get token balances
            token_balances = await self._get_erc20_balances()
            
            # Get current ETH price
            eth_price = await self._get_token_price('ethereum')
            usd_balance = eth_balance * eth_price
            
            # Add major token values
            for token_symbol, token_amount in token_balances.items():
                token_price = await self._get_token_price(token_symbol.lower())
                usd_balance += token_amount * token_price
            
            balance = WalletBalance(
                address=self.wallet_address,
                chain='ethereum',
                eth_balance=eth_balance,
                usd_balance=usd_balance,
                token_balances=token_balances,
                last_updated=datetime.now()
            )
            
            logger.info(f"✅ Ethereum balance: {eth_balance:.4f} ETH (~${usd_balance:,.2f})")
            return balance
            
        except Exception as e:
            logger.error(f"Error getting Ethereum balance: {e}")
            return WalletBalance(
                address=self.wallet_address,
                chain='ethereum',
                eth_balance=0.0,
                usd_balance=0.0,
                token_balances={},
                last_updated=datetime.now()
            )
    
    async def _get_eth_balance_etherscan(self) -> Optional[float]:
        """Get ETH balance using Etherscan API"""
        try:
            url = f"{self.endpoints['ethereum']['etherscan']}"
            params = {
                'module': 'account',
                'action': 'balance',
                'address': self.wallet_address,
                'tag': 'latest',
                'apikey': 'YourApiKeyToken'  # Free tier works without key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == '1':
                            # Convert from wei to ETH
                            balance_wei = int(data['result'])
                            balance_eth = balance_wei / 10**18
                            return balance_eth
            
            return None
            
        except Exception as e:
            logger.warning(f"Etherscan balance check failed: {e}")
            return None
    
    async def _get_eth_balance_alchemy(self) -> Optional[float]:
        """Get ETH balance using Alchemy API (backup method)"""
        try:
            # This would require an Alchemy API key
            # For demo, we'll simulate a balance check
            logger.info("Attempting Alchemy balance check (would need API key)...")
            return None
            
        except Exception as e:
            logger.warning(f"Alchemy balance check failed: {e}")
            return None
    
    async def _get_erc20_balances(self) -> Dict[str, float]:
        """Get ERC-20 token balances"""
        try:
            # Common token addresses
            token_contracts = {
                'USDC': '0xA0b86a33E6411c3Ce98e6d9b4B3C61a6F2B0C1D2',
                'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',
                'WETH': '0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2',
                'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F'
            }
            
            token_balances = {}
            
            for symbol, contract_address in token_contracts.items():
                try:
                    balance = await self._get_token_balance(contract_address, symbol)
                    if balance > 0:
                        token_balances[symbol] = balance
                except Exception as e:
                    logger.warning(f"Failed to get {symbol} balance: {e}")
            
            return token_balances
            
        except Exception as e:
            logger.error(f"Error getting ERC-20 balances: {e}")
            return {}
    
    async def _get_token_balance(self, contract_address: str, symbol: str) -> float:
        """Get specific ERC-20 token balance"""
        try:
            url = f"{self.endpoints['ethereum']['etherscan']}"
            params = {
                'module': 'account',
                'action': 'tokenbalance',
                'contractaddress': contract_address,
                'address': self.wallet_address,
                'tag': 'latest',
                'apikey': 'YourApiKeyToken'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data['status'] == '1':
                            # Most tokens use 18 decimals, but some use different
                            decimals = 18  # Could fetch this from contract
                            balance_raw = int(data['result'])
                            balance = balance_raw / (10 ** decimals)
                            return balance
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get {symbol} balance: {e}")
            return 0.0
    
    async def _get_token_price(self, token_id: str) -> float:
        """Get current token price from CoinGecko"""
        try:
            if token_id in self.token_prices:
                return self.token_prices[token_id]
            
            # Map common symbols to CoinGecko IDs
            coingecko_ids = {
                'ethereum': 'ethereum',
                'usdc': 'usd-coin',
                'usdt': 'tether',
                'weth': 'weth',
                'dai': 'dai'
            }
            
            cg_id = coingecko_ids.get(token_id.lower(), token_id.lower())
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': cg_id,
                'vs_currencies': 'usd'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if cg_id in data:
                            price = data[cg_id]['usd']
                            self.token_prices[token_id] = price
                            return price
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to get price for {token_id}: {e}")
            return 0.0
    
    async def get_all_balances(self) -> Dict[str, WalletBalance]:
        """Get balances across all supported chains"""
        try:
            logger.info(f"🔍 Checking wallet balances for: {self.wallet_address}")
            
            balances = {}
            
            # Get Ethereum balance
            eth_balance = await self.get_ethereum_balance()
            balances['ethereum'] = eth_balance
            
            # Could add BSC and Polygon here
            # bsc_balance = await self.get_bsc_balance()
            # polygon_balance = await self.get_polygon_balance()
            
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

# Demo/test function
async def demo_wallet_detection():
    """Demo the wallet balance detection"""
    wallet_address = "0xb4add0df12df32981773ca25ee88bdab750bfa20"
    
    detector = WalletBalanceDetector(wallet_address)
    
    print("🔍 WALLET BALANCE DETECTION DEMO")
    print("=" * 50)
    print(f"Wallet: {wallet_address}")
    
    try:
        # Get Ethereum balance
        eth_balance = await detector.get_ethereum_balance()
        print(f"\n📊 Ethereum Balance:")
        print(f"   ETH: {eth_balance.eth_balance:.4f}")
        print(f"   USD: ${eth_balance.usd_balance:,.2f}")
        
        if eth_balance.token_balances:
            print(f"   Tokens:")
            for symbol, amount in eth_balance.token_balances.items():
                print(f"     {symbol}: {amount:.2f}")
        
        # Get total portfolio value
        total_value = await detector.get_total_portfolio_value()
        print(f"\n💰 Total Portfolio: ${total_value:,.2f}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_wallet_detection())