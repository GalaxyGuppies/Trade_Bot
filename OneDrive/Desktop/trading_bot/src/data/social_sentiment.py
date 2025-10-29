"""
Social sentiment collector for Twitter, Reddit, and other sources
"""
import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

import tweepy
import praw
from transformers import pipeline

logger = logging.getLogger(__name__)

class SocialSentimentCollector:
    def __init__(self):
        self.twitter_client = None
        self.reddit_client = None
        self.sentiment_analyzer = None
        self.sentiment_data = {}
        self.running = False
        self.keywords = [
            'bitcoin', 'btc', 'ethereum', 'eth', 'solana', 'sol',
            'crypto', 'cryptocurrency', 'defi', 'nft'
        ]
        
        # Initialize APIs and sentiment analyzer
        self.init_apis()
        self.init_sentiment_analyzer()
    
    def init_apis(self):
        """Initialize social media APIs"""
        try:
            # Twitter API v2 (you'll need to add your credentials)
            # Bearer token from config
            bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
            if bearer_token and bearer_token != "YOUR_TWITTER_BEARER_TOKEN":
                self.twitter_client = tweepy.Client(bearer_token=bearer_token)
                logger.info("Twitter API initialized")
            
            # Reddit API
            reddit_client_id = "YOUR_REDDIT_CLIENT_ID"
            reddit_client_secret = "YOUR_REDDIT_CLIENT_SECRET"
            
            if reddit_client_id and reddit_client_id != "YOUR_REDDIT_CLIENT_ID":
                self.reddit_client = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent="TradingBot/1.0"
                )
                logger.info("Reddit API initialized")
                
        except Exception as e:
            logger.error(f"Error initializing social media APIs: {e}")
    
    def init_sentiment_analyzer(self):
        """Initialize sentiment analysis model with hardware optimization"""
        try:
            # Import hardware optimizer
            from src.hardware_optimizer import hardware_optimizer
            
            # Get optimized configuration
            model_config = hardware_optimizer.optimization_config['model_config']
            device = model_config.get('device', 'cpu')
            model_name = model_config.get('sentiment_model', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
            
            logger.info(f"Initializing sentiment analyzer on {device} with model: {model_name}")
            
            # Apply hardware optimizations
            hardware_optimizer.optimize_torch_settings()
            
            # Initialize model with optimized settings
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0 if device == 'cuda' and hardware_optimizer.hardware_info['gpu_available'] else -1,
                return_all_scores=True,
                batch_size=model_config.get('batch_size', 8),
                max_length=model_config.get('max_length', 256)
            )
            
            logger.info(f"Sentiment analyzer initialized successfully on {device}")
            logger.info(f"Hardware strategy: {hardware_optimizer.optimization_config['strategy']}")
            
        except Exception as e:
            logger.error(f"Error initializing optimized sentiment analyzer: {e}")
            # Fallback to basic model
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=-1,  # Force CPU for fallback
                    return_all_scores=True
                )
                logger.info("Fallback sentiment analyzer initialized")
            except Exception as e2:
                logger.error(f"Error initializing fallback sentiment analyzer: {e2}")
    
    async def start(self):
        """Start sentiment collection"""
        if self.running:
            return
        
        self.running = True
        logger.info("Starting social sentiment collection...")
        
        # Start collection tasks
        if self.twitter_client:
            asyncio.create_task(self.collect_twitter_sentiment())
        
        if self.reddit_client:
            asyncio.create_task(self.collect_reddit_sentiment())
        
        # Start periodic cleanup
        asyncio.create_task(self.cleanup_old_data())
    
    async def stop(self):
        """Stop sentiment collection"""
        self.running = False
        logger.info("Stopping social sentiment collection...")
    
    async def collect_twitter_sentiment(self):
        """Collect sentiment from Twitter"""
        while self.running:
            try:
                for keyword in self.keywords:
                    # Search for recent tweets
                    tweets = self.twitter_client.search_recent_tweets(
                        query=f"{keyword} -is:retweet",
                        max_results=100,
                        tweet_fields=['created_at', 'public_metrics']
                    )
                    
                    if tweets.data:
                        await self.process_tweets(tweets.data, keyword)
                
                await asyncio.sleep(300)  # Wait 5 minutes between collections
                
            except Exception as e:
                logger.error(f"Error collecting Twitter sentiment: {e}")
                await asyncio.sleep(300)
    
    async def process_tweets(self, tweets: List[Any], keyword: str):
        """Process tweets for sentiment analysis"""
        sentiments = []
        
        for tweet in tweets:
            try:
                # Clean tweet text
                text = self.clean_text(tweet.text)
                
                # Analyze sentiment
                sentiment_result = await self.analyze_sentiment(text)
                
                if sentiment_result:
                    sentiment_data = {
                        'text': text[:100],  # Store first 100 chars
                        'sentiment': sentiment_result,
                        'timestamp': tweet.created_at,
                        'source': 'twitter',
                        'keyword': keyword,
                        'engagement': getattr(tweet, 'public_metrics', {})
                    }
                    sentiments.append(sentiment_data)
                    
            except Exception as e:
                logger.error(f"Error processing tweet: {e}")
        
        # Update sentiment data
        await self.update_sentiment_data(keyword, sentiments)
    
    async def collect_reddit_sentiment(self):
        """Collect sentiment from Reddit"""
        while self.running:
            try:
                # Monitor crypto-related subreddits
                subreddits = ['cryptocurrency', 'bitcoin', 'ethereum', 'solana', 'defi']
                
                for subreddit_name in subreddits:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts
                    hot_posts = subreddit.hot(limit=50)
                    
                    for post in hot_posts:
                        try:
                            # Check if post mentions our keywords
                            text = f"{post.title} {post.selftext}"
                            
                            for keyword in self.keywords:
                                if keyword.lower() in text.lower():
                                    # Analyze sentiment
                                    sentiment_result = await self.analyze_sentiment(text)
                                    
                                    if sentiment_result:
                                        sentiment_data = {
                                            'text': text[:200],
                                            'sentiment': sentiment_result,
                                            'timestamp': datetime.fromtimestamp(post.created_utc),
                                            'source': 'reddit',
                                            'keyword': keyword,
                                            'engagement': {
                                                'score': post.score,
                                                'num_comments': post.num_comments
                                            }
                                        }
                                        
                                        await self.update_sentiment_data(keyword, [sentiment_data])
                                        
                        except Exception as e:
                            logger.error(f"Error processing Reddit post: {e}")
                
                await asyncio.sleep(600)  # Wait 10 minutes between collections
                
            except Exception as e:
                logger.error(f"Error collecting Reddit sentiment: {e}")
                await asyncio.sleep(600)
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions and hashtags for better sentiment analysis
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    async def analyze_sentiment(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of text"""
        if not self.sentiment_analyzer or not text:
            return None
        
        try:
            # Run sentiment analysis
            result = self.sentiment_analyzer(text[:512])  # Limit text length
            
            if result:
                # Convert to standardized format
                if isinstance(result[0], list):
                    # Model returns all scores
                    scores = {item['label'].lower(): item['score'] for item in result[0]}
                else:
                    # Model returns single prediction
                    scores = {result[0]['label'].lower(): result[0]['score']}
                
                # Calculate compound sentiment score
                compound_score = self.calculate_compound_score(scores)
                
                return {
                    'scores': scores,
                    'compound': compound_score,
                    'label': self.get_sentiment_label(compound_score)
                }
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
        
        return None
    
    def calculate_compound_score(self, scores: Dict[str, float]) -> float:
        """Calculate compound sentiment score (-1 to 1)"""
        # Map different label formats to standard scores
        positive = scores.get('positive', scores.get('pos', 0))
        negative = scores.get('negative', scores.get('neg', 0))
        neutral = scores.get('neutral', 0)
        
        # Calculate compound score
        compound = positive - negative
        return max(-1, min(1, compound))
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """Get sentiment label from compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    async def update_sentiment_data(self, keyword: str, sentiments: List[Dict[str, Any]]):
        """Update sentiment data for a keyword"""
        if keyword not in self.sentiment_data:
            self.sentiment_data[keyword] = {
                'sentiments': [],
                'summary': {
                    'total_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0,
                    'average_sentiment': 0.0,
                    'last_update': datetime.now()
                }
            }
        
        # Add new sentiments
        self.sentiment_data[keyword]['sentiments'].extend(sentiments)
        
        # Keep only recent sentiments (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.sentiment_data[keyword]['sentiments'] = [
            s for s in self.sentiment_data[keyword]['sentiments']
            if s['timestamp'] > cutoff_time
        ]
        
        # Update summary
        await self.update_sentiment_summary(keyword)
    
    async def update_sentiment_summary(self, keyword: str):
        """Update sentiment summary for a keyword"""
        sentiments = self.sentiment_data[keyword]['sentiments']
        
        if not sentiments:
            return
        
        total_count = len(sentiments)
        positive_count = sum(1 for s in sentiments if s['sentiment']['label'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment']['label'] == 'negative')
        neutral_count = total_count - positive_count - negative_count
        
        # Calculate average sentiment
        avg_sentiment = sum(s['sentiment']['compound'] for s in sentiments) / total_count
        
        self.sentiment_data[keyword]['summary'] = {
            'total_count': total_count,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'average_sentiment': avg_sentiment,
            'positive_ratio': positive_count / total_count,
            'negative_ratio': negative_count / total_count,
            'last_update': datetime.now()
        }
    
    async def cleanup_old_data(self):
        """Periodically clean up old sentiment data"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(hours=48)
                
                for keyword in self.sentiment_data:
                    self.sentiment_data[keyword]['sentiments'] = [
                        s for s in self.sentiment_data[keyword]['sentiments']
                        if s['timestamp'] > cutoff_time
                    ]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error cleaning up sentiment data: {e}")
                await asyncio.sleep(3600)
    
    async def get_symbol_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get sentiment data for a symbol"""
        # Map symbol to keyword
        symbol_lower = symbol.lower()
        keyword_mapping = {
            'btc': 'bitcoin',
            'eth': 'ethereum',
            'sol': 'solana'
        }
        
        keyword = keyword_mapping.get(symbol_lower[:3], symbol_lower)
        
        return self.sentiment_data.get(keyword)
    
    async def get_overall_sentiment(self) -> Dict[str, Any]:
        """Get overall market sentiment"""
        all_sentiments = []
        
        for keyword_data in self.sentiment_data.values():
            all_sentiments.extend(keyword_data['sentiments'])
        
        if not all_sentiments:
            return {
                'total_count': 0,
                'average_sentiment': 0.0,
                'dominant_sentiment': 'neutral'
            }
        
        total_count = len(all_sentiments)
        avg_sentiment = sum(s['sentiment']['compound'] for s in all_sentiments) / total_count
        
        positive_count = sum(1 for s in all_sentiments if s['sentiment']['label'] == 'positive')
        negative_count = sum(1 for s in all_sentiments if s['sentiment']['label'] == 'negative')
        
        if positive_count > negative_count:
            dominant_sentiment = 'positive'
        elif negative_count > positive_count:
            dominant_sentiment = 'negative'
        else:
            dominant_sentiment = 'neutral'
        
        return {
            'total_count': total_count,
            'average_sentiment': avg_sentiment,
            'dominant_sentiment': dominant_sentiment,
            'positive_ratio': positive_count / total_count,
            'negative_ratio': negative_count / total_count,
            'last_update': datetime.now().isoformat()
        }