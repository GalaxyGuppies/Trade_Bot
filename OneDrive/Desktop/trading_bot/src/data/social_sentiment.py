"""
Social sentiment collector with third-party sentiment APIs
Enhanced with professional sentiment services: Santiment, StockGeist, OpenAI GPT-4, and social media APIs
Includes Reddit API integration for additional sentiment analysis
"""
import asyncio
import logging
import re
import time
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
import json

import tweepy
import requests
import httpx
import numpy as np
import praw  # Reddit API

logger = logging.getLogger(__name__)

class EnhancedSocialSentimentCollector:
    """
    Professional social sentiment collector with third-party APIs
    Integrates multiple sentiment services for comprehensive market sentiment analysis
    Now includes Reddit sentiment analysis
    """
    
    def __init__(self, 
                 # Twitter/X.com API
                 twitter_api_key: str = "Sj8ivlnfFe5feHLyLKysOJLyI",
                 twitter_api_secret: str = "vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y",
                 twitter_bearer_token: str = None,
                 twitter_access_token: str = None,
                 twitter_access_token_secret: str = None,
                 
                 # Reddit API
                 reddit_client_id: str = "VbckKzcb-yJPSTefnDFqDQ",
                 reddit_client_secret: str = None,
                 reddit_user_agent: str = "TradingBot:v1.0.0 (by /u/yourusername)",
                 
                 # Third-party sentiment APIs
                 santiment_api_key: str = None,
                 stockgeist_api_key: str = None,
                 openai_api_key: str = None,
                 lunarcrush_api_key: str = None,
                 messari_api_key: str = None,
                 worldnewsapi_key: str = None,
                 
                 # Blockchain analysis APIs
                 moralis_api_key: str = None):
        
        # Twitter/X.com API credentials
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_bearer_token = twitter_bearer_token
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret
        
        # Reddit API credentials
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.reddit_user_agent = reddit_user_agent
        
        # Third-party sentiment API credentials
        self.santiment_api_key = santiment_api_key
        self.stockgeist_api_key = stockgeist_api_key
        self.openai_api_key = openai_api_key
        self.lunarcrush_api_key = lunarcrush_api_key
        self.messari_api_key = messari_api_key
        self.worldnewsapi_key = worldnewsapi_key or os.getenv("WORLDNEWSAPI_KEY")
        
        # Blockchain analysis APIs
        self.moralis_api_key = moralis_api_key
        
        # Initialize Moralis provider if API key is provided
        self.moralis_provider = None
        if self.moralis_api_key:
            from .moralis_provider import MoralisProvider
            self.moralis_provider = MoralisProvider(self.moralis_api_key)
            logger.info("🔗 Moralis blockchain analysis initialized")
        
        # API clients
        self.twitter_client = None
        self.twitter_api_v1 = None
        self.reddit_client = None
        
        # Data storage
        self.sentiment_data = {}
        self.tweet_cache = {}
        self.running = False
        
        # Rate limiting
        self.last_twitter_request = 0
        self.twitter_request_interval = 1.1
        
        # API usage tracking for quota management
        self.worldnews_quota_used = 0
        self.worldnews_quota_left = 0
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = None
        self.init_sentiment_analyzer()
        
        # Initialize APIs
        self.init_twitter_api()
        self.init_reddit_api()
        
        logger.info("Enhanced Social Sentiment Collector initialized")
    
    def init_sentiment_analyzer(self):
        """Initialize sentiment analysis models"""
        try:
            # Try TextBlob first (simpler and more reliable)
            try:
                from textblob import TextBlob
                self.sentiment_analyzer = 'textblob'
                logger.info("✅ TextBlob sentiment analyzer initialized")
                return
            except ImportError:
                logger.warning("📦 TextBlob not available")
            
            # Fallback to basic sentiment
            self.sentiment_analyzer = 'basic'
            logger.info("✅ Basic sentiment analyzer initialized")
                    
        except Exception as e:
            logger.error(f"Failed to initialize sentiment analyzer: {e}")
            self.sentiment_analyzer = 'basic'
    
    def init_twitter_api(self):
        """Initialize X.com (Twitter) API with provided credentials"""
        try:
            if self.twitter_api_key and self.twitter_api_secret:
                # Initialize Twitter API v2 client
                if self.twitter_bearer_token:
                    self.twitter_client = tweepy.Client(
                        bearer_token=self.twitter_bearer_token,
                        consumer_key=self.twitter_api_key,
                        consumer_secret=self.twitter_api_secret,
                        access_token=self.twitter_access_token,
                        access_token_secret=self.twitter_access_token_secret,
                        wait_on_rate_limit=True
                    )
                else:
                    # For API key/secret only setup
                    self.twitter_client = tweepy.Client(
                        consumer_key=self.twitter_api_key,
                        consumer_secret=self.twitter_api_secret,
                        wait_on_rate_limit=True
                    )
                
                logger.info("✅ X.com (Twitter) API initialized successfully")
                return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Twitter API: {e}")
            return False
    
    def init_reddit_api(self):
        """Initialize Reddit API"""
        try:
            if self.reddit_client_id:
                try:
                    import praw
                    self.reddit_client = praw.Reddit(
                        client_id=self.reddit_client_id,
                        client_secret=self.reddit_client_secret or "",
                        user_agent=self.reddit_user_agent
                    )
                    logger.info("✅ Reddit API initialized successfully")
                except ImportError:
                    logger.warning("📦 praw not installed - Reddit functionality disabled")
                    self.reddit_client = None
                except Exception as e:
                    logger.error(f"Reddit API initialization error: {e}")
                    self.reddit_client = None
            else:
                logger.info("Reddit credentials not provided - Reddit functionality disabled")
                self.reddit_client = None
                
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit_client = None
    
    async def collect_twitter_sentiment(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Collect sentiment data from X.com (Twitter) for given symbols"""
        if not self.twitter_client:
            logger.warning("Twitter API not available")
            return {}
        
        sentiment_results = {}
        
        try:
            for symbol in symbols:
                logger.info(f"🐦 Collecting Twitter sentiment for {symbol}")
                
                # Rate limiting
                await self._rate_limit_twitter()
                
                # Build search query
                query = f"${symbol} OR #{symbol} -is:retweet lang:en"
                
                try:
                    # Search recent tweets
                    tweets = tweepy.Paginator(
                        self.twitter_client.search_recent_tweets,
                        query=query,
                        tweet_fields=['public_metrics', 'created_at', 'author_id'],
                        max_results=100
                    ).flatten(limit=200)
                    
                    tweet_list = []
                    for tweet in tweets:
                        if tweet.created_at > datetime.now(timezone.utc) - timedelta(hours=hours_back):
                            tweet_list.append(tweet)
                    
                    # Analyze sentiment
                    sentiment_data = await self._analyze_tweets_sentiment(tweet_list, symbol)
                    sentiment_results[symbol] = sentiment_data
                    
                except Exception as e:
                    logger.warning(f"Error collecting tweets for {symbol}: {e}")
                    sentiment_results[symbol] = self._get_empty_sentiment()
                
        except Exception as e:
            logger.error(f"Error in Twitter sentiment collection: {e}")
        
        return sentiment_results
    
    async def _analyze_tweets_sentiment(self, tweets: List, symbol: str) -> Dict[str, Any]:
        """Analyze sentiment of collected tweets"""
        if not tweets:
            return self._get_empty_sentiment()
        
        sentiments = []
        total_engagement = 0
        
        for tweet in tweets:
            try:
                # Get metrics
                metrics = tweet.public_metrics
                engagement = (
                    metrics.get('retweet_count', 0) * 2 +
                    metrics.get('like_count', 0) * 1.5 +
                    metrics.get('reply_count', 0)
                )
                
                # Analyze sentiment
                tweet_sentiment = await self._analyze_text_sentiment(tweet.text)
                
                # Weight by engagement
                weighted_sentiment = tweet_sentiment * (1 + np.log1p(engagement))
                sentiments.append(weighted_sentiment)
                total_engagement += engagement
                
            except Exception as e:
                logger.warning(f"Error analyzing tweet: {e}")
                continue
        
        # Calculate metrics
        if sentiments:
            avg_sentiment = np.mean(sentiments)
        else:
            avg_sentiment = 0.0
        
        sentiment_label = self._get_sentiment_label(avg_sentiment)
        
        return {
            'sentiment_score': float(avg_sentiment),
            'sentiment_label': sentiment_label,
            'tweet_count': len(tweets),
            'total_engagement': int(total_engagement),
            'avg_engagement': float(total_engagement / len(tweets)) if tweets else 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        try:
            if self.sentiment_analyzer == 'textblob':
                from textblob import TextBlob
                blob = TextBlob(text)
                return blob.sentiment.polarity
            else:
                return self._basic_sentiment_analysis(text)
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def _analyze_text_sentiment_sync(self, text: str) -> float:
        """Synchronous sentiment analysis for Reddit"""
        try:
            if self.sentiment_analyzer == 'textblob':
                from textblob import TextBlob
                blob = TextBlob(text)
                return blob.sentiment.polarity
            else:
                return self._basic_sentiment_analysis(text)
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {e}")
            return 0.0
    
    def _basic_sentiment_analysis(self, text: str) -> float:
        """Basic keyword-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_words = [
            'bullish', 'moon', 'pump', 'buy', 'hodl', 'diamond', 'hands',
            'rocket', 'gains', 'profit', 'up', 'rise', 'bull', 'green',
            'awesome', 'great', 'amazing', 'love', 'best', 'good'
        ]
        
        negative_words = [
            'bearish', 'dump', 'sell', 'crash', 'down', 'bear', 'red',
            'loss', 'lose', 'drop', 'fall', 'bad', 'worst', 'terrible',
            'hate', 'avoid', 'scam', 'rug', 'dead', 'rip'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        return max(-1.0, min(1.0, sentiment_score * 5))
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.1:
            return 'bullish'
        elif score < -0.1:
            return 'bearish'
        else:
            return 'neutral'
    
    def _get_empty_sentiment(self) -> Dict[str, Any]:
        """Return empty sentiment data structure"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'tweet_count': 0,
            'total_engagement': 0,
            'avg_engagement': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    async def _rate_limit_twitter(self):
        """Rate limiting for Twitter API"""
        current_time = time.time()
        time_since_last = current_time - self.last_twitter_request
        
        if time_since_last < self.twitter_request_interval:
            sleep_time = self.twitter_request_interval - time_since_last
            await asyncio.sleep(sleep_time)
        
        self.last_twitter_request = time.time()
    
    async def collect_reddit_sentiment(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Collect sentiment from Reddit cryptocurrency subreddits"""
        if not self.reddit_client:
            logger.warning("Reddit client not available")
            return {symbol: self._get_empty_sentiment() for symbol in symbols}
        
        sentiment_results = {}
        
        # Cryptocurrency-focused subreddits
        subreddits = ['CryptoCurrency', 'altcoins', 'CryptoMoonShots', 'SatoshiStreetBets', 
                     'defi', 'ethtrader', 'Bitcoin', 'CryptoMarkets']
        
        for symbol in symbols:
            try:
                posts_data = []
                comments_data = []
                
                # Search across multiple subreddits
                for subreddit_name in subreddits:
                    try:
                        subreddit = self.reddit_client.subreddit(subreddit_name)
                        
                        # Search for posts mentioning the symbol
                        for post in subreddit.search(symbol, time_filter='day', limit=10):
                            if post.created_utc > (time.time() - hours_back * 3600):
                                posts_data.append({
                                    'text': f"{post.title} {post.selftext}",
                                    'score': post.score,
                                    'num_comments': post.num_comments,
                                    'created': post.created_utc,
                                    'subreddit': subreddit_name
                                })
                                
                                # Get top comments
                                post.comments.replace_more(limit=0)
                                for comment in post.comments[:5]:  # Top 5 comments
                                    if hasattr(comment, 'body') and comment.body:
                                        comments_data.append({
                                            'text': comment.body,
                                            'score': comment.score,
                                            'created': comment.created_utc
                                        })
                    
                    except Exception as e:
                        logger.warning(f"Error accessing r/{subreddit_name}: {e}")
                        continue
                
                # Analyze sentiment
                sentiment_data = self._analyze_reddit_sentiment(posts_data, comments_data, symbol)
                sentiment_results[symbol] = sentiment_data
                
                # Rate limiting
                await asyncio.sleep(1)  # Reddit API rate limiting
                
            except Exception as e:
                logger.error(f"Error collecting Reddit sentiment for {symbol}: {e}")
                sentiment_results[symbol] = self._get_empty_sentiment()
        
        return sentiment_results
    
    def _analyze_reddit_sentiment(self, posts_data: List[Dict], comments_data: List[Dict], symbol: str) -> Dict[str, Any]:
        """Analyze sentiment from Reddit posts and comments"""
        try:
            all_texts = []
            weighted_scores = []
            
            # Process posts
            for post in posts_data:
                text = post['text']
                if text and len(text.strip()) > 10:
                    sentiment = self._analyze_text_sentiment_sync(text)
                    # Weight by Reddit score and comment count
                    weight = max(1, post['score']) * max(1, post['num_comments'] / 10)
                    weighted_scores.append((sentiment, weight))
                    all_texts.append(text)
            
            # Process comments
            for comment in comments_data:
                text = comment['text']
                if text and len(text.strip()) > 5:
                    sentiment = self._analyze_text_sentiment_sync(text)
                    # Weight by comment score
                    weight = max(1, comment['score'])
                    weighted_scores.append((sentiment, weight))
                    all_texts.append(text)
            
            if not weighted_scores:
                return self._get_empty_sentiment()
            
            # Calculate weighted average sentiment
            total_weight = sum(weight for _, weight in weighted_scores)
            weighted_sentiment = sum(sentiment * weight for sentiment, weight in weighted_scores) / total_weight
            
            # Calculate additional metrics
            post_count = len(posts_data)
            comment_count = len(comments_data)
            avg_score = np.mean([post['score'] for post in posts_data]) if posts_data else 0
            
            return {
                'sentiment_score': weighted_sentiment,
                'confidence': min(0.95, (post_count + comment_count) / 20),  # Higher confidence with more data
                'volume': post_count + comment_count,
                'source': 'reddit',
                'post_count': post_count,
                'comment_count': comment_count,
                'average_score': avg_score,
                'texts': all_texts[:10],  # Keep sample texts
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Reddit sentiment: {e}")
            return self._get_empty_sentiment()
    
    async def get_combined_sentiment(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """Get combined sentiment analysis from all sources"""
        logger.info(f"🔍 Collecting combined sentiment for {symbols}")
        
        # Collect from all sources
        twitter_results = await self.collect_twitter_sentiment(symbols, hours_back)
        reddit_results = await self.collect_reddit_sentiment(symbols, hours_back)
        
        combined_results = {}
        
        for symbol in symbols:
            twitter_data = twitter_results.get(symbol, {})
            reddit_data = reddit_results.get(symbol, {})
            
            # Combine sentiment scores
            twitter_weight = 0.8  # Twitter gets higher weight
            reddit_weight = 0.2
            
            twitter_sentiment = twitter_data.get('sentiment_score', 0.0)
            reddit_sentiment = reddit_data.get('sentiment_score', 0.0)
            
            combined_sentiment = (
                twitter_sentiment * twitter_weight +
                reddit_sentiment * reddit_weight
            )
            
            sentiment_label = self._get_sentiment_label(combined_sentiment)
            
            # Calculate confidence
            confidence = 0.0
            if twitter_data.get('tweet_count', 0) > 0:
                confidence += 0.8
            if reddit_data.get('post_count', 0) > 0:
                confidence += 0.2
            
            combined_results[symbol] = {
                'sentiment_score': float(combined_sentiment),
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'twitter_data': twitter_data,
                'reddit_data': reddit_data,
                'data_sources': {
                    'twitter_available': bool(twitter_data),
                    'reddit_available': bool(reddit_data)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return combined_results
    
    def start_monitoring(self, symbols: List[str], interval_minutes: int = 30):
        """Start continuous sentiment monitoring"""
        self.running = True
        logger.info(f"🚀 Starting sentiment monitoring for {symbols}")
        
        async def monitor_loop():
            while self.running:
                try:
                    sentiment_data = await self.get_combined_sentiment(symbols)
                    
                    # Store results
                    timestamp = datetime.now().isoformat()
                    for symbol, data in sentiment_data.items():
                        if symbol not in self.sentiment_data:
                            self.sentiment_data[symbol] = []
                        
                        self.sentiment_data[symbol].append({
                            'timestamp': timestamp,
                            'data': data
                        })
                        
                        # Keep only last 24 hours
                        cutoff = datetime.now() - timedelta(hours=24)
                        self.sentiment_data[symbol] = [
                            entry for entry in self.sentiment_data[symbol]
                            if datetime.fromisoformat(entry['timestamp']) > cutoff
                        ]
                    
                    await asyncio.sleep(interval_minutes * 60)
                    
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(monitor_loop())
    
    def stop_monitoring(self):
        """Stop sentiment monitoring"""
        self.running = False
        logger.info("🛑 Sentiment monitoring stopped")
    
    def get_sentiment_summary(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment summary for a symbol"""
        history = self.sentiment_data.get(symbol, [])
        
        if not history:
            return {
                'current_sentiment': 'neutral',
                'trend': 'stable',
                'confidence': 0.0,
                'data_points': 0
            }
        
        latest = history[-1]['data']
        
        return {
            'current_sentiment': latest['sentiment_label'],
            'sentiment_score': latest['sentiment_score'],
            'trend': 'stable',  # Simplified
            'confidence': latest['confidence'],
            'data_points': len(history),
            'last_updated': latest['timestamp']
        }
    
    # ==================== THIRD-PARTY SENTIMENT APIS ====================
    
    async def collect_santiment_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect sentiment data from Santiment API
        Santiment provides on-chain, social, and development activity metrics
        """
        if not self.santiment_api_key:
            logger.warning("Santiment API key not provided")
            return {}
        
        sentiment_results = {}
        
        try:
            headers = {
                'Authorization': f'Apikey {self.santiment_api_key}',
                'Content-Type': 'application/json'
            }
            
            for symbol in symbols:
                logger.info(f"📊 Collecting Santiment sentiment for {symbol}")
                
                # Santiment GraphQL query for social sentiment
                query = """
                {
                  getMetric(metric: "sentiment_positive_total") {
                    timeseriesData(
                      slug: "%s"
                      from: "%s"
                      to: "%s"
                      interval: "1h"
                    ) {
                      datetime
                      value
                    }
                  }
                }
                """ % (
                    symbol.lower(),
                    (datetime.now() - timedelta(hours=24)).isoformat(),
                    datetime.now().isoformat()
                )
                
                try:
                    response = await self._make_api_request(
                        'https://api.santiment.net/graphql',
                        method='POST',
                        headers=headers,
                        json={'query': query}
                    )
                    
                    if response and 'data' in response:
                        sentiment_data = self._process_santiment_data(response['data'], symbol)
                        sentiment_results[symbol] = sentiment_data
                    else:
                        sentiment_results[symbol] = self._get_empty_third_party_sentiment('santiment')
                        
                except Exception as e:
                    logger.warning(f"Error fetching Santiment data for {symbol}: {e}")
                    sentiment_results[symbol] = self._get_empty_third_party_sentiment('santiment')
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in Santiment sentiment collection: {e}")
        
        return sentiment_results
    
    async def collect_lunarcrush_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect sentiment data from LunarCrush API
        LunarCrush aggregates social sentiment from multiple platforms
        Note: Free API key provides demo data only - real data requires subscription
        """
        if not self.lunarcrush_api_key:
            logger.warning("LunarCrush API key not provided")
            return {}
        
        sentiment_results = {}
        
        try:
            # Use the working endpoint we discovered
            url = "https://lunarcrush.com/api4/public/coins/list/v1"
            headers = {
                'Authorization': f'Bearer {self.lunarcrush_api_key}',
                'User-Agent': 'TradingBot/1.0'
            }
            
            try:
                response = await self._make_api_request(url, headers=headers)
                
                if response and response.get('data'):
                    # Check if we're in demo mode
                    config = response.get('config', {})
                    is_demo_mode = 'DEMO DATA MODE' in config.get('notice', '')
                    
                    if is_demo_mode:
                        logger.warning("🌙 LunarCrush API in DEMO MODE - real data requires subscription")
                        logger.info("💡 Visit https://lunarcrush.com/pricing for subscription options")
                    
                    # Process available coins data
                    coins_data = response.get('data', [])
                    
                    for symbol in symbols:
                        # Find the symbol in the coins list
                        coin_data = None
                        for coin in coins_data:
                            if coin.get('symbol', '').upper() == symbol.upper():
                                coin_data = coin
                                break
                        
                        if coin_data:
                            sentiment_data = self._process_lunarcrush_data(coin_data, symbol)
                            # Mark as demo data if applicable
                            if is_demo_mode:
                                sentiment_data['is_demo_data'] = True
                                sentiment_data['note'] = 'Demo data - subscription required for real sentiment'
                            sentiment_results[symbol] = sentiment_data
                        else:
                            sentiment_results[symbol] = self._get_empty_third_party_sentiment('lunarcrush')
                            sentiment_results[symbol]['note'] = f'Symbol {symbol} not found in LunarCrush data'
                else:
                    for symbol in symbols:
                        sentiment_results[symbol] = self._get_empty_third_party_sentiment('lunarcrush')
                        
            except Exception as e:
                logger.warning(f"Error fetching LunarCrush data: {e}")
                for symbol in symbols:
                    sentiment_results[symbol] = self._get_empty_third_party_sentiment('lunarcrush')
            
            # Rate limiting for free tier
            await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in LunarCrush sentiment collection: {e}")
        
        return sentiment_results
    
    async def collect_openai_sentiment(self, symbols: List[str], news_texts: List[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI GPT-4 for advanced context understanding
        Can analyze news articles, social media posts, or market commentary
        """
        if not self.openai_api_key:
            logger.warning("OpenAI API key not provided")
            return {}
        
        sentiment_results = {}
        
        try:
            headers = {
                'Authorization': f'Bearer {self.openai_api_key}',
                'Content-Type': 'application/json'
            }
            
            for symbol in symbols:
                logger.info(f"🤖 Analyzing {symbol} sentiment with OpenAI GPT-4")
                
                # Create context-aware prompt
                prompt = self._create_openai_sentiment_prompt(symbol, news_texts)
                
                payload = {
                    'model': 'gpt-4',
                    'messages': [
                        {
                            'role': 'system',
                            'content': 'You are a professional cryptocurrency sentiment analyst. Analyze the provided text and return a sentiment score between -1 (very bearish) and 1 (very bullish), along with confidence level and key factors.'
                        },
                        {
                            'role': 'user',
                            'content': prompt
                        }
                    ],
                    'max_tokens': 200,
                    'temperature': 0.1
                }
                
                try:
                    response = await self._make_api_request(
                        'https://api.openai.com/v1/chat/completions',
                        method='POST',
                        headers=headers,
                        json=payload
                    )
                    
                    if response and 'choices' in response:
                        sentiment_data = self._process_openai_data(response['choices'][0]['message']['content'], symbol)
                        sentiment_results[symbol] = sentiment_data
                    else:
                        sentiment_results[symbol] = self._get_empty_third_party_sentiment('openai')
                        
                except Exception as e:
                    logger.warning(f"Error with OpenAI sentiment for {symbol}: {e}")
                    sentiment_results[symbol] = self._get_empty_third_party_sentiment('openai')
                
                # Rate limiting for OpenAI
                await asyncio.sleep(2.0)
                
        except Exception as e:
            logger.error(f"Error in OpenAI sentiment analysis: {e}")
        
        return sentiment_results
    
    async def collect_messari_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect market intelligence from Messari API
        Provides professional-grade market analysis and news sentiment
        """
        if not self.messari_api_key:
            logger.warning("Messari API key not provided")
            return {}
        
        sentiment_results = {}
        
        try:
            headers = {
                'x-messari-api-key': self.messari_api_key
            }
            
            for symbol in symbols:
                logger.info(f"📈 Collecting Messari intelligence for {symbol}")
                
                # Get news and market intel
                url = f"https://data.messari.io/api/v1/news/{symbol.lower()}"
                
                try:
                    response = await self._make_api_request(url, headers=headers)
                    
                    if response and 'data' in response:
                        sentiment_data = self._process_messari_data(response['data'], symbol)
                        sentiment_results[symbol] = sentiment_data
                    else:
                        sentiment_results[symbol] = self._get_empty_third_party_sentiment('messari')
                        
                except Exception as e:
                    logger.warning(f"Error fetching Messari data for {symbol}: {e}")
                    sentiment_results[symbol] = self._get_empty_third_party_sentiment('messari')
                
                # Rate limiting
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in Messari sentiment collection: {e}")
        
        return sentiment_results
    
    async def collect_worldnews_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Collect sentiment data from WorldNewsAPI
        Provides high-quality news sentiment analysis for cryptocurrencies
        """
        if not self.worldnewsapi_key:
            logger.warning("WorldNewsAPI key not provided")
            return {}
        
        sentiment_results = {}
        
        try:
            for symbol in symbols:
                logger.info(f"📰 Collecting WorldNews sentiment for {symbol}")
                
                # Search for news articles about the cryptocurrency
                # Use multiple search terms to get comprehensive coverage
                search_queries = [
                    f"{symbol} cryptocurrency price",
                    f"{symbol} crypto news",
                    f"Bitcoin {symbol}" if symbol != 'BTC' else "Bitcoin price analysis"
                ]
                
                all_articles = []
                
                for query in search_queries:
                    url = "https://api.worldnewsapi.com/search-news"
                    params = {
                        'text': query,
                        'language': 'en',
                        'sort': 'publish-time',
                        'sort-direction': 'DESC',
                        'number': 10,  # Get 10 articles per query
                        'earliest-publish-date': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                        'api-key': self.worldnewsapi_key
                    }
                    
                    try:
                        response = await self._make_worldnews_request(url, params)
                        
                        if response and response.get('news'):
                            articles = response['news']
                            all_articles.extend(articles)
                        
                        # Rate limiting between queries
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.warning(f"Error fetching WorldNews data for {symbol} query '{query}': {e}")
                        continue
                
                # Process collected articles for sentiment
                if all_articles:
                    sentiment_data = self._process_worldnews_data(all_articles, symbol)
                    sentiment_results[symbol] = sentiment_data
                else:
                    sentiment_results[symbol] = self._get_empty_third_party_sentiment('worldnews')
                
                # Rate limiting between symbols
                await asyncio.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in WorldNews sentiment collection: {e}")
        
        return sentiment_results
    
    async def _make_worldnews_request(self, url: str, params: Dict) -> Optional[Dict]:
        """Make WorldNewsAPI request with quota tracking"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30)
                
                # Track API quota usage BEFORE processing response
                self._update_worldnews_quota(response.headers)
                
                response.raise_for_status()
                result = response.json()
                
                return result
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("WorldNewsAPI rate limit exceeded")
            elif e.response.status_code == 401:
                logger.error("WorldNewsAPI authentication failed - check API key")
            else:
                logger.warning(f"WorldNewsAPI HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"WorldNewsAPI request failed: {e}")
            return None
    
    def _update_worldnews_quota(self, headers: Dict):
        """Update WorldNewsAPI quota tracking from response headers"""
        try:
            # WorldNewsAPI uses lowercase headers with hyphens
            if 'x-api-quota-used' in headers:
                self.worldnews_quota_used = float(headers['x-api-quota-used'])
            if 'x-api-quota-left' in headers:
                self.worldnews_quota_left = float(headers['x-api-quota-left'])
                
            if self.worldnews_quota_left < 10:
                logger.warning(f"⚠️  WorldNewsAPI quota running low: {self.worldnews_quota_left:.1f} requests left today")
            elif self.worldnews_quota_left < 2:
                logger.error(f"🚨 WorldNewsAPI quota critical: {self.worldnews_quota_left:.1f} requests left today")
                
        except (ValueError, KeyError) as e:
            logger.debug(f"Could not parse WorldNewsAPI quota headers: {e}")
    
    def _process_worldnews_data(self, articles: List[Dict], symbol: str) -> Dict[str, Any]:
        """Process WorldNewsAPI response data and analyze sentiment"""
        try:
            if not articles:
                return self._get_empty_third_party_sentiment('worldnews')
            
            # Analyze sentiment of news articles
            sentiments = []
            relevant_articles = 0
            
            for article in articles[:20]:  # Analyze up to 20 most recent articles
                title = article.get('title', '')
                text = article.get('text', '')
                summary = article.get('summary', '')
                
                # Combine title, summary, and text for analysis
                combined_text = f"{title} {summary} {text}"
                
                # Check if article is relevant to the cryptocurrency
                if self._is_crypto_relevant(combined_text, symbol):
                    sentiment = self._basic_sentiment_analysis(combined_text)
                    
                    # Weight more recent articles higher
                    publish_date = article.get('publish_date', '')
                    weight = self._calculate_article_weight(publish_date)
                    
                    sentiments.append(sentiment * weight)
                    relevant_articles += 1
            
            if sentiments:
                # Calculate weighted average sentiment
                avg_sentiment = np.mean(sentiments)
                sentiment_label = self._get_sentiment_label(avg_sentiment)
                
                # Confidence based on number of relevant articles and recency
                confidence = min(0.95, (relevant_articles / 10.0) * 0.8 + 0.2)
            else:
                avg_sentiment = 0.0
                sentiment_label = 'neutral'
                confidence = 0.0
            
            return {
                'sentiment_score': float(avg_sentiment),
                'sentiment_label': sentiment_label,
                'source': 'worldnews',
                'confidence': float(confidence),
                'articles_analyzed': len(articles),
                'relevant_articles': relevant_articles,
                'data_quality': self._assess_worldnews_quality(articles, relevant_articles),
                'api_quota_used': self.worldnews_quota_used,
                'api_quota_left': self.worldnews_quota_left,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing WorldNews data for {symbol}: {e}")
            return self._get_empty_third_party_sentiment('worldnews')
    
    def _is_crypto_relevant(self, text: str, symbol: str) -> bool:
        """Check if news article text is relevant to the cryptocurrency"""
        text_lower = text.lower()
        symbol_lower = symbol.lower()
        
        # Common cryptocurrency terms
        crypto_terms = [
            symbol_lower, f"${symbol_lower}", 
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'defi', 'nft', 'altcoin', 'token', 'coin',
            'trading', 'exchange', 'wallet', 'mining', 'staking'
        ]
        
        # Symbol-specific terms
        symbol_terms = {
            'btc': ['bitcoin', 'btc'],
            'eth': ['ethereum', 'eth', 'ether'],
            'sol': ['solana', 'sol'],
            'ada': ['cardano', 'ada'],
            'dot': ['polkadot', 'dot'],
            'matic': ['polygon', 'matic'],
            'avax': ['avalanche', 'avax'],
            'link': ['chainlink', 'link']
        }
        
        # Add symbol-specific terms
        if symbol_lower in symbol_terms:
            crypto_terms.extend(symbol_terms[symbol_lower])
        
        # Check for relevance
        relevant_count = sum(1 for term in crypto_terms if term in text_lower)
        return relevant_count >= 2  # At least 2 crypto-related terms
    
    def _calculate_article_weight(self, publish_date: str) -> float:
        """Calculate weight for article based on publish date (more recent = higher weight)"""
        try:
            if not publish_date:
                return 0.5  # Default weight for articles without date
            
            # Parse the date (assuming ISO format)
            article_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
            current_date = datetime.now(article_date.tzinfo)
            
            # Calculate hours since publication
            hours_old = (current_date - article_date).total_seconds() / 3600
            
            # Weight decreases with age: 1.0 for 0-6 hours, 0.8 for 6-24 hours, etc.
            if hours_old <= 6:
                return 1.0
            elif hours_old <= 24:
                return 0.8
            elif hours_old <= 72:
                return 0.6
            elif hours_old <= 168:  # 1 week
                return 0.4
            else:
                return 0.2
                
        except Exception as e:
            logger.debug(f"Error calculating article weight: {e}")
            return 0.5  # Default weight
    
    def _assess_worldnews_quality(self, articles: List[Dict], relevant_articles: int) -> str:
        """Assess the quality of WorldNews data"""
        if not articles:
            return 'no_data'
        
        total_articles = len(articles)
        relevance_ratio = relevant_articles / total_articles if total_articles > 0 else 0
        
        if relevance_ratio >= 0.7 and relevant_articles >= 5:
            return 'excellent'
        elif relevance_ratio >= 0.5 and relevant_articles >= 3:
            return 'good'
        elif relevance_ratio >= 0.3 and relevant_articles >= 2:
            return 'fair'
        else:
            return 'poor'
    
    # ==================== THIRD-PARTY DATA PROCESSORS ====================
    
    def _process_santiment_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Process Santiment API response data"""
        try:
            timeseries = data.get('getMetric', {}).get('timeseriesData', [])
            
            if not timeseries:
                return self._get_empty_third_party_sentiment('santiment')
            
            # Calculate sentiment from positive sentiment data
            values = [float(point['value']) for point in timeseries if point['value'] is not None]
            
            if values:
                avg_sentiment = np.mean(values)
                # Normalize to -1 to 1 scale (Santiment provides 0-100 scale)
                normalized_sentiment = (avg_sentiment / 50.0) - 1.0
                sentiment_label = self._get_sentiment_label(normalized_sentiment)
            else:
                normalized_sentiment = 0.0
                sentiment_label = 'neutral'
            
            return {
                'sentiment_score': float(normalized_sentiment),
                'sentiment_label': sentiment_label,
                'source': 'santiment',
                'data_points': len(values),
                'confidence': min(0.9, len(values) / 24.0),  # Higher confidence with more data points
                'raw_data': timeseries[-5:],  # Last 5 data points for reference
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Santiment data for {symbol}: {e}")
            return self._get_empty_third_party_sentiment('santiment')
    
    def _process_lunarcrush_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Process LunarCrush API v4 response data"""
        try:
            # Handle both topic data and coins list data formats
            if 'topic' in data:
                # Topic endpoint format
                topic_data = data.get('topic', {})
                social_score = topic_data.get('social_score', 0)
                social_volume = topic_data.get('social_volume_24h', 0)
                social_engagement = topic_data.get('social_engagement_24h', 0)
                social_contributors = topic_data.get('social_contributors_24h', 0)
            else:
                # Coins list endpoint format
                social_score = data.get('galaxy_score', 0) or 0
                social_volume = data.get('social_volume_24h', 0) or 0
                social_engagement = data.get('interactions_24h', 0) or 0
                social_contributors = 0  # Not available in coins list
                
                # Additional market data from coins endpoint
                price = data.get('price', 0)
                market_cap = data.get('market_cap', 0)
                percent_change_24h = data.get('percent_change_24h', 0)
            
            # Calculate sentiment score from available metrics
            if social_volume > 0 and social_engagement > 0:
                # Create sentiment score based on engagement ratio and volume
                engagement_ratio = social_engagement / max(social_volume, 1)
                volume_score = min(1.0, social_volume / 1000.0)  # Normalize volume
                
                # Combine metrics into sentiment (-1 to 1 scale)
                raw_sentiment = (engagement_ratio * volume_score * 2) - 1
                sentiment_score = max(-1.0, min(1.0, raw_sentiment))
            elif social_score and social_score > 0:
                # Use galaxy score if available (0-100 scale)
                sentiment_score = (social_score / 50.0) - 1.0  # Convert to -1 to 1
            else:
                sentiment_score = 0.0
            
            sentiment_label = self._get_sentiment_label(sentiment_score)
            
            # Calculate confidence based on data availability
            confidence_factors = []
            if social_volume > 0:
                confidence_factors.append(min(1.0, social_volume / 100.0))
            if social_engagement > 0:
                confidence_factors.append(min(1.0, social_engagement / 500.0))
            if social_score and social_score > 0:
                confidence_factors.append(min(1.0, social_score / 100.0))
            
            confidence = min(0.95, sum(confidence_factors) / max(len(confidence_factors), 1)) if confidence_factors else 0.0
            
            result = {
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label,
                'source': 'lunarcrush',
                'social_volume': social_volume,
                'social_engagement': social_engagement,
                'social_contributors': social_contributors,
                'social_score': social_score,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add market data if available
            if 'price' in locals():
                result.update({
                    'price': price,
                    'market_cap': market_cap,
                    'percent_change_24h': percent_change_24h
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing LunarCrush data for {symbol}: {e}")
            return self._get_empty_third_party_sentiment('lunarcrush')
    
    def _process_openai_data(self, response_text: str, symbol: str) -> Dict[str, Any]:
        """Process OpenAI GPT-4 sentiment analysis response"""
        try:
            # Parse GPT-4 response (expecting structured format)
            # GPT-4 should return something like: "Sentiment: 0.7, Confidence: 0.8, Factors: [positive news, strong fundamentals]"
            
            import re
            
            # Extract sentiment score
            sentiment_match = re.search(r'sentiment[:\s]+(-?\d*\.?\d+)', response_text.lower())
            sentiment_score = float(sentiment_match.group(1)) if sentiment_match else 0.0
            
            # Extract confidence
            confidence_match = re.search(r'confidence[:\s]+(\d*\.?\d+)', response_text.lower())
            confidence = float(confidence_match.group(1)) if confidence_match else 0.5
            
            # Extract key factors
            factors_match = re.search(r'factors[:\s]+\[(.*?)\]', response_text.lower())
            factors = factors_match.group(1).split(',') if factors_match else []
            
            sentiment_label = self._get_sentiment_label(sentiment_score)
            
            return {
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label,
                'source': 'openai_gpt4',
                'confidence': confidence,
                'key_factors': [f.strip() for f in factors],
                'full_analysis': response_text,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing OpenAI data for {symbol}: {e}")
            return self._get_empty_third_party_sentiment('openai')
    
    def _process_messari_data(self, data: Dict, symbol: str) -> Dict[str, Any]:
        """Process Messari API response data"""
        try:
            news_items = data.get('news', [])
            
            if not news_items:
                return self._get_empty_third_party_sentiment('messari')
            
            # Analyze sentiment of news headlines and content
            sentiments = []
            for item in news_items[:10]:  # Analyze last 10 news items
                title = item.get('title', '')
                content = item.get('content', '')
                text = f"{title} {content}"
                
                if text.strip():
                    sentiment = self._basic_sentiment_analysis(text)
                    sentiments.append(sentiment)
            
            if sentiments:
                avg_sentiment = np.mean(sentiments)
                sentiment_label = self._get_sentiment_label(avg_sentiment)
                confidence = min(0.9, len(sentiments) / 10.0)
            else:
                avg_sentiment = 0.0
                sentiment_label = 'neutral'
                confidence = 0.0
            
            return {
                'sentiment_score': float(avg_sentiment),
                'sentiment_label': sentiment_label,
                'source': 'messari',
                'news_count': len(news_items),
                'confidence': confidence,
                'latest_headlines': [item.get('title', '') for item in news_items[:3]],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing Messari data for {symbol}: {e}")
            return self._get_empty_third_party_sentiment('messari')
    
    # ==================== HELPER METHODS ====================
    
    async def _make_api_request(self, url: str, method: str = 'GET', headers: Dict = None, json: Dict = None) -> Optional[Dict]:
        """Make async API request with error handling"""
        try:
            # Use httpx for proper async support
            async with httpx.AsyncClient() as client:
                if method.upper() == 'POST':
                    response = await client.post(url, headers=headers, json=json, timeout=30)
                else:
                    response = await client.get(url, headers=headers, timeout=30)
                
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            logger.warning(f"API request failed for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def _create_openai_sentiment_prompt(self, symbol: str, news_texts: List[str] = None) -> str:
        """Create context-aware prompt for OpenAI sentiment analysis"""
        prompt = f"""
        Analyze the current market sentiment for {symbol} cryptocurrency.
        
        Consider the following context:
        - Recent market trends and price movements
        - Social media sentiment and discussions
        - News and announcements
        - Technical analysis indicators
        - Overall crypto market conditions
        """
        
        if news_texts:
            prompt += f"\n\nRecent news and social media content:\n"
            for i, text in enumerate(news_texts[:5]):  # Limit to 5 texts
                prompt += f"{i+1}. {text}\n"
        
        prompt += f"""
        
        Please provide:
        1. Sentiment score between -1 (very bearish) and 1 (very bullish)
        2. Confidence level between 0 and 1
        3. Key factors influencing the sentiment
        
        Format: Sentiment: [score], Confidence: [confidence], Factors: [factor1, factor2, factor3]
        """
        
        return prompt
    
    def _get_empty_third_party_sentiment(self, source: str) -> Dict[str, Any]:
        """Return empty sentiment data structure for third-party APIs"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'source': source,
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'error': 'No data available'
        }
    
    # ==================== ENHANCED COMBINED SENTIMENT ====================
    
    async def get_professional_combined_sentiment(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Any]:
        """
        Get professional-grade combined sentiment from all sources including third-party APIs
        """
        logger.info(f"🔍 Collecting professional sentiment analysis for {symbols}")
        
        # Collect from all sources concurrently
        results = await asyncio.gather(
            self.collect_twitter_sentiment(symbols, hours_back),
            self.collect_reddit_sentiment(symbols, hours_back),
            self.collect_santiment_sentiment(symbols),
            self.collect_lunarcrush_sentiment(symbols),
            self.collect_messari_sentiment(symbols),
            self.collect_worldnews_sentiment(symbols),
            return_exceptions=True
        )
        
        twitter_results, reddit_results, santiment_results, lunarcrush_results, messari_results, worldnews_results = results
        
        # Handle any exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Error in sentiment collection {i}: {result}")
                results[i] = {}
        
        combined_results = {}
        
        for symbol in symbols:
            # Gather all sentiment data
            twitter_data = twitter_results.get(symbol, {}) if isinstance(twitter_results, dict) else {}
            reddit_data = reddit_results.get(symbol, {}) if isinstance(reddit_results, dict) else {}
            santiment_data = santiment_results.get(symbol, {}) if isinstance(santiment_results, dict) else {}
            lunarcrush_data = lunarcrush_results.get(symbol, {}) if isinstance(lunarcrush_results, dict) else {}
            messari_data = messari_results.get(symbol, {}) if isinstance(messari_results, dict) else {}
            worldnews_data = worldnews_results.get(symbol, {}) if isinstance(worldnews_results, dict) else {}
            
            # Weight different sources based on reliability and data quality
            weights = {
                'twitter': 0.20,      # Real-time but noisy
                'reddit': 0.15,       # Community sentiment
                'santiment': 0.20,    # Professional on-chain data
                'lunarcrush': 0.15,   # Social metrics (demo data only)
                'messari': 0.15,      # Fundamental analysis
                'worldnews': 0.15     # News sentiment analysis
            }
            
            # Calculate weighted sentiment
            weighted_sentiments = []
            available_sources = []
            total_confidence = 0.0
            
            if twitter_data.get('tweet_count', 0) > 0:
                twitter_sentiment = twitter_data.get('sentiment_score', 0.0)
                weighted_sentiments.append(twitter_sentiment * weights['twitter'])
                available_sources.append('twitter')
                total_confidence += weights['twitter']
            
            if reddit_data.get('post_count', 0) > 0:
                reddit_sentiment = reddit_data.get('sentiment_score', 0.0)
                weighted_sentiments.append(reddit_sentiment * weights['reddit'])
                available_sources.append('reddit')
                total_confidence += weights['reddit']
            
            if santiment_data.get('source') == 'santiment':
                santiment_sentiment = santiment_data.get('sentiment_score', 0.0)
                santiment_confidence = santiment_data.get('confidence', 0.5)
                weighted_sentiments.append(santiment_sentiment * weights['santiment'] * santiment_confidence)
                available_sources.append('santiment')
                total_confidence += weights['santiment'] * santiment_confidence
            
            if lunarcrush_data.get('source') == 'lunarcrush':
                lunarcrush_sentiment = lunarcrush_data.get('sentiment_score', 0.0)
                lunarcrush_confidence = lunarcrush_data.get('confidence', 0.5)
                weighted_sentiments.append(lunarcrush_sentiment * weights['lunarcrush'] * lunarcrush_confidence)
                available_sources.append('lunarcrush')
                total_confidence += weights['lunarcrush'] * lunarcrush_confidence
            
            if messari_data.get('source') == 'messari':
                messari_sentiment = messari_data.get('sentiment_score', 0.0)
                messari_confidence = messari_data.get('confidence', 0.5)
                weighted_sentiments.append(messari_sentiment * weights['messari'] * messari_confidence)
                available_sources.append('messari')
                total_confidence += weights['messari'] * messari_confidence
            
            if worldnews_data.get('source') == 'worldnews':
                worldnews_sentiment = worldnews_data.get('sentiment_score', 0.0)
                worldnews_confidence = worldnews_data.get('confidence', 0.5)
                weighted_sentiments.append(worldnews_sentiment * weights['worldnews'] * worldnews_confidence)
                available_sources.append('worldnews')
                total_confidence += weights['worldnews'] * worldnews_confidence
            
            # Calculate final sentiment
            if weighted_sentiments and total_confidence > 0:
                combined_sentiment = sum(weighted_sentiments) / total_confidence
            else:
                combined_sentiment = 0.0
            
            sentiment_label = self._get_sentiment_label(combined_sentiment)
            
            # Calculate overall confidence based on source diversity and individual confidences
            source_diversity_bonus = len(available_sources) / 5.0  # More sources = higher confidence
            final_confidence = min(0.95, total_confidence * source_diversity_bonus)
            
            combined_results[symbol] = {
                'sentiment_score': float(combined_sentiment),
                'sentiment_label': sentiment_label,
                'confidence': final_confidence,
                'available_sources': available_sources,
                'source_count': len(available_sources),
                'detailed_data': {
                    'twitter': twitter_data,
                    'reddit': reddit_data,
                    'santiment': santiment_data,
                    'lunarcrush': lunarcrush_data,
                    'messari': messari_data,
                    'worldnews': worldnews_data
                },
                'weights_used': {k: v for k, v in weights.items() if k.replace('_', '') in [s.replace('_', '') for s in available_sources]},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"📊 {symbol}: {sentiment_label} sentiment ({combined_sentiment:.3f}) "
                       f"from {len(available_sources)} sources with {final_confidence:.1%} confidence")
        
        return combined_results

# Maintain backward compatibility
SocialSentimentCollector = EnhancedSocialSentimentCollector