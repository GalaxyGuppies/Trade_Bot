"""
Social sentiment collector for X.com (Twitter), Reddit, and other sources
Enhanced with real X.com API integration
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
import numpy as np

logger = logging.getLogger(__name__)

class SocialSentimentCollector:
    """
    Enhanced social sentiment collector with real X.com API integration
    Collects and analyzes sentiment from multiple social media sources
    """
    
    def __init__(self, 
                 twitter_api_key: str = "Sj8ivlnfFe5feHLyLKysOJLyI",
                 twitter_api_secret: str = "vTfAWSayK2jkMt40kyczU0QgZE8Z7qEx5GQFjFPQpQgyZgj31y",
                 twitter_bearer_token: str = None,
                 twitter_access_token: str = None,
                 twitter_access_token_secret: str = None):
        
        # X.com (Twitter) API credentials
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret = twitter_api_secret
        self.twitter_bearer_token = twitter_bearer_token
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret
        
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
        """Initialize Reddit API (optional)"""
        try:
            reddit_client_id = os.getenv("REDDIT_CLIENT_ID", "")
            reddit_client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
            
            if reddit_client_id and reddit_client_secret:
                try:
                    import praw
                    self.reddit_client = praw.Reddit(
                        client_id=reddit_client_id,
                        client_secret=reddit_client_secret,
                        user_agent="TradingBot_SentimentAnalyzer_v1.0"
                    )
                    logger.info("✅ Reddit API initialized successfully")
                except ImportError:
                    logger.warning("📦 praw not installed - Reddit functionality disabled")
                    self.reddit_client = None
            else:
                logger.info("Reddit credentials not found - Reddit functionality disabled")
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
        """Collect sentiment from Reddit (placeholder - requires praw)"""
        if not self.reddit_client:
            return {}
        
        # Placeholder implementation
        sentiment_results = {}
        for symbol in symbols:
            sentiment_results[symbol] = self._get_empty_sentiment()
        
        return sentiment_results
    
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