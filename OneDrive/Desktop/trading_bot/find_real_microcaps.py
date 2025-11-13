#!/usr/bin/env python3
"""
Find Real Solana Microcap Tokens
"""

import asyncio
import aiohttp
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def find_real_solana_microcaps():
    """Find real Solana microcap tokens from DexScreener"""
    
    base_url = "https://api.dexscreener.com/latest"
    
    # Search for lower volume Solana tokens
    endpoints = [
        "/dex/search?chainId=solana&q=solana",
        "/dex/search?q=solana",
        "/dex/search?chainId=solana&q=meme",
        "/dex/search?chainId=solana&q=coin",
    ]
    
    logger.info("🔍 Searching for real Solana microcap tokens...")
    
    all_tokens = []
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            try:
                url = f"{base_url}{endpoint}"
                logger.info(f"\nSearching: {endpoint}")
                
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'pairs' in data and data['pairs']:
                            pairs = data['pairs']
                            
                            # Filter for Solana tokens in microcap range
                            for pair in pairs:
                                chain_id = pair.get('chainId', '').lower()
                                if chain_id != 'solana':
                                    continue
                                
                                base_token = pair.get('baseToken', {})
                                market_cap = float(pair.get('marketCap', 0))
                                volume_24h = float(pair.get('volume', {}).get('h24', 0))
                                liquidity_usd = float(pair.get('liquidity', {}).get('usd', 0))
                                
                                # Target range: 100k - 10M market cap (broader range), minimal volume/liquidity
                                if (100000 <= market_cap <= 10000000 and 
                                    volume_24h > 1000 and 
                                    liquidity_usd > 1000):
                                    
                                    token_info = {
                                        'address': base_token.get('address', ''),
                                        'symbol': base_token.get('symbol', ''),
                                        'name': base_token.get('name', ''),
                                        'price_usd': float(pair.get('priceUsd', 0)),
                                        'volume_24h': volume_24h,
                                        'liquidity_usd': liquidity_usd,
                                        'price_change_24h': float(pair.get('priceChange', {}).get('h24', 0)),
                                        'market_cap': market_cap,
                                        'dex': pair.get('dexId', ''),
                                        'pair_address': pair.get('pairAddress', '')
                                    }
                                    
                                    # Avoid duplicates
                                    if not any(t['address'] == token_info['address'] for t in all_tokens):
                                        all_tokens.append(token_info)
                                        
                                        logger.info(f"✅ Found: {token_info['symbol']} - ${market_cap:,.0f} MC, ${volume_24h:,.0f} Vol")
                                        logger.info(f"   Address: {token_info['address']}")
                        
            except Exception as e:
                logger.error(f"Error searching {endpoint}: {e}")
    
    logger.info(f"\n🎯 Total microcap tokens found: {len(all_tokens)}")
    
    # Show top 5 candidates
    if all_tokens:
        sorted_tokens = sorted(all_tokens, key=lambda x: x['volume_24h'], reverse=True)
        logger.info("\n📊 Top 5 Solana Microcap Candidates:")
        
        for i, token in enumerate(sorted_tokens[:5], 1):
            logger.info(f"\n{i}. {token['symbol']} ({token['name']})")
            logger.info(f"   Address: {token['address']}")
            logger.info(f"   Market Cap: ${token['market_cap']:,.0f}")
            logger.info(f"   Volume 24h: ${token['volume_24h']:,.0f}")
            logger.info(f"   Liquidity: ${token['liquidity_usd']:,.0f}")
            logger.info(f"   Price: ${token['price_usd']:.6f}")
            logger.info(f"   DEX: {token['dex']}")
        
        return sorted_tokens[:3]  # Return top 3
    
    return []

if __name__ == "__main__":
    tokens = asyncio.run(find_real_solana_microcaps())
    
    if tokens:
        print("\n" + "="*50)
        print("READY TO USE IN BOT:")
        print("="*50)
        
        for i, token in enumerate(tokens):
            print(f"""
{{
    'address': '{token['address']}',
    'symbol': '{token['symbol']}',
    'name': '{token['name']}',
    'price_usd': {token['price_usd']},
    'volume_24h': {token['volume_24h']},
    'liquidity_usd': {token['liquidity_usd']},
    'price_change_24h': {token['price_change_24h']},
    'market_cap': {token['market_cap']},
    'dex': '{token['dex']}',
    'pair_address': '{token.get('pair_address', '')}',
    'discovery_source': 'curated',
    'network': 'solana'
}},""")