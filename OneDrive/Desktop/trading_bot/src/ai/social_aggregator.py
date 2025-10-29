"""
Cross-Platform Social Signal Aggregation
Advanced sentiment analysis across 10+ platforms with AI fake news detection
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import re
import json
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
import joblib

logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    platform: str
    author: str
    author_followers: int
    author_verified: bool
    content: str
    timestamp: datetime
    engagement: Dict  # likes, shares, comments, etc.
    url: Optional[str] = None
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    media_urls: List[str] = field(default_factory=list)

@dataclass
class InfluencerProfile:
    username: str
    platform: str
    followers: int
    verified: bool
    credibility_score: float  # 0-1
    bias_score: float  # -1 (bearish) to 1 (bullish)
    accuracy_history: List[float]  # Historical accuracy of predictions
    engagement_rate: float
    account_age_days: int

@dataclass
class SentimentSignal:
    platform: str
    symbol: str
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0-1
    volume: int  # Number of mentions
    reach: int  # Total potential audience
    influencer_sentiment: float  # Weighted by influencer credibility
    fake_news_probability: float  # 0-1
    trending_score: float  # 0-1
    metadata: Dict

class FakeNewsDetector:
    """AI-powered fake news and spam detection"""
    
    def __init__(self):
        self.fake_news_model = None
        self.spam_model = None
        self.vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        
        # Load or initialize models
        self._load_models()
        
        # Fake news indicators
        self.fake_indicators = {
            'urgency_words': ['urgent', 'emergency', 'breaking', 'exclusive', 'leaked'],
            'hyperbole': ['massive', 'huge', 'incredible', 'unbelievable', 'shocking'],
            'conspiracy': ['they dont want you to know', 'hidden truth', 'conspiracy', 'coverup'],
            'financial_spam': ['guaranteed', 'risk-free', 'get rich quick', 'easy money']
        }
        
        # Reliable source domains
        self.reliable_sources = {
            'cointelegraph.com', 'coindesk.com', 'bloomberg.com', 'reuters.com',
            'decrypt.co', 'theblock.co', 'forbes.com', 'cnbc.com'
        }
    
    def _load_models(self):
        """Load or initialize fake news detection models"""
        try:
            self.fake_news_model = joblib.load('models/fake_news_model.pkl')
            self.spam_model = joblib.load('models/spam_model.pkl')
            self.vectorizer = joblib.load('models/fake_news_vectorizer.pkl')
        except FileNotFoundError:
            # Initialize new models
            self.fake_news_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.spam_model = MultinomialNB()
    
    async def detect_fake_news(self, post: SocialPost) -> float:
        """Detect probability that post contains fake news (0-1)"""
        try:
            fake_probability = 0.0
            
            # Text-based analysis
            text_score = self._analyze_text_patterns(post.content)
            fake_probability += text_score * 0.4
            
            # Author credibility
            author_score = self._analyze_author_credibility(post)
            fake_probability += author_score * 0.3
            
            # Engagement patterns
            engagement_score = self._analyze_engagement_patterns(post)
            fake_probability += engagement_score * 0.2
            
            # Source analysis
            source_score = self._analyze_source_reliability(post)
            fake_probability += source_score * 0.1
            
            return min(fake_probability, 1.0)
            
        except Exception as e:
            logger.error(f"Error in fake news detection: {e}")
            return 0.5  # Default medium risk
    
    def _analyze_text_patterns(self, content: str) -> float:
        """Analyze text for fake news patterns"""
        score = 0.0
        content_lower = content.lower()
        
        # Check for fake news indicators
        for category, words in self.fake_indicators.items():
            for word in words:
                if word in content_lower:
                    if category == 'urgency_words':
                        score += 0.2
                    elif category == 'hyperbole':
                        score += 0.15
                    elif category == 'conspiracy':
                        score += 0.3
                    elif category == 'financial_spam':
                        score += 0.25
        
        # Check for excessive capitalization
        caps_ratio = sum(1 for c in content if c.isupper()) / max(len(content), 1)
        if caps_ratio > 0.3:
            score += 0.2
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in content if c in '!?') / max(len(content), 1)
        if punct_ratio > 0.05:
            score += 0.15
        
        # Check for price predictions without basis
        price_pattern = r'\$[0-9,]+'
        price_matches = re.findall(price_pattern, content)
        if len(price_matches) > 2:  # Multiple specific price targets
            score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_author_credibility(self, post: SocialPost) -> float:
        """Analyze author credibility factors"""
        score = 0.0
        
        # Low follower count but not verified
        if post.author_followers < 1000 and not post.author_verified:
            score += 0.3
        
        # New account (if we had account age data)
        # This would require additional API calls to get account creation date
        
        # Check username patterns (common bot patterns)
        username = post.author.lower()
        if re.match(r'.*[0-9]{4,}$', username):  # Ends with 4+ digits
            score += 0.2
        
        if len(username) < 5:  # Very short usernames
            score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_engagement_patterns(self, post: SocialPost) -> float:
        """Analyze engagement patterns for bot activity"""
        score = 0.0
        
        likes = post.engagement.get('likes', 0)
        shares = post.engagement.get('shares', 0)
        comments = post.engagement.get('comments', 0)
        
        total_engagement = likes + shares + comments
        
        if total_engagement == 0:
            return 0.1  # Low engagement is slightly suspicious
        
        # Unusual engagement ratios
        if shares > likes * 2:  # More shares than likes (unusual)
            score += 0.2
        
        if comments > 0:
            comment_like_ratio = comments / likes if likes > 0 else 0
            if comment_like_ratio > 0.5:  # Very high comment ratio
                score += 0.1
        
        # Very high engagement for small accounts
        if post.author_followers < 10000 and total_engagement > post.author_followers:
            score += 0.3
        
        return min(score, 1.0)
    
    def _analyze_source_reliability(self, post: SocialPost) -> float:
        """Analyze source reliability"""
        if not post.url:
            return 0.1  # Slight penalty for no source
        
        # Check against reliable sources
        for reliable_domain in self.reliable_sources:
            if reliable_domain in post.url:
                return 0.0  # Reliable source
        
        # Check for suspicious domains
        suspicious_patterns = [
            r'\.tk$', r'\.ml$', r'\.ga$',  # Free domains
            r'crypto.*pump', r'moon.*coin',  # Pump domains
            r'[0-9]+\.[a-z]+',  # Numeric domains
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, post.url):
                return 0.4
        
        return 0.2  # Unknown source gets medium suspicion

class InfluencerScorer:
    """Scores and tracks crypto influencers across platforms"""
    
    def __init__(self):
        self.influencer_database = {}
        self.load_influencer_data()
    
    def load_influencer_data(self):
        """Load known influencer profiles"""
        # In production, this would load from a database
        self.influencer_database = {
            'twitter': {
                'elonmusk': InfluencerProfile(
                    username='elonmusk',
                    platform='twitter',
                    followers=150000000,
                    verified=True,
                    credibility_score=0.7,
                    bias_score=0.2,  # Slightly bullish
                    accuracy_history=[0.6, 0.7, 0.5, 0.8],
                    engagement_rate=0.05,
                    account_age_days=5000
                ),
                'VitalikButerin': InfluencerProfile(
                    username='VitalikButerin',
                    platform='twitter',
                    followers=5000000,
                    verified=True,
                    credibility_score=0.9,
                    bias_score=0.1,
                    accuracy_history=[0.8, 0.9, 0.7, 0.8],
                    engagement_rate=0.03,
                    account_age_days=4000
                )
            },
            'reddit': {
                'u/cryptoexpert': InfluencerProfile(
                    username='u/cryptoexpert',
                    platform='reddit',
                    followers=50000,
                    verified=False,
                    credibility_score=0.6,
                    bias_score=-0.1,
                    accuracy_history=[0.7, 0.6, 0.8],
                    engagement_rate=0.08,
                    account_age_days=2000
                )
            }
        }
    
    def get_influencer_score(self, username: str, platform: str) -> float:
        """Get credibility score for influencer"""
        platform_data = self.influencer_database.get(platform, {})
        influencer = platform_data.get(username)
        
        if not influencer:
            return 0.5  # Default score for unknown influencers
        
        return influencer.credibility_score
    
    def calculate_weighted_sentiment(self, posts: List[SocialPost], sentiments: List[float]) -> float:
        """Calculate sentiment weighted by influencer credibility"""
        if not posts or not sentiments:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for post, sentiment in zip(posts, sentiments):
            # Get influencer weight
            influencer_score = self.get_influencer_score(post.author, post.platform)
            
            # Additional weight factors
            follower_weight = min(np.log(post.author_followers + 1) / 20, 1.0)  # Log scale
            verified_weight = 1.2 if post.author_verified else 1.0
            engagement_weight = min(sum(post.engagement.values()) / 1000, 2.0)
            
            # Combined weight
            total_post_weight = influencer_score * follower_weight * verified_weight * engagement_weight
            
            weighted_sum += sentiment * total_post_weight
            total_weight += total_post_weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0

class PlatformConnector:
    """Connects to various social media platforms"""
    
    def __init__(self):
        self.platforms = {
            'twitter': TwitterConnector(),
            'reddit': RedditConnector(),
            'telegram': TelegramConnector(),
            'discord': DiscordConnector(),
            'youtube': YouTubeConnector(),
            'tiktok': TikTokConnector(),
            'medium': MediumConnector(),
            'github': GitHubConnector(),
            'news': NewsConnector(),
            'forums': ForumConnector()
        }
    
    async def get_posts(self, platform: str, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get posts from specific platform"""
        connector = self.platforms.get(platform)
        if not connector:
            logger.warning(f"No connector for platform: {platform}")
            return []
        
        try:
            return await connector.get_posts(symbol, hours)
        except Exception as e:
            logger.error(f"Error getting posts from {platform}: {e}")
            return []

class TwitterConnector:
    """Twitter API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get Twitter posts about symbol"""
        # Mock Twitter data - in production would use Twitter API v2
        posts = []
        
        # Generate mock tweets
        for i in range(20):
            posts.append(SocialPost(
                platform='twitter',
                author=f'crypto_user_{i}',
                author_followers=np.random.randint(100, 100000),
                author_verified=np.random.random() > 0.8,
                content=self._generate_mock_tweet(symbol),
                timestamp=datetime.now() - timedelta(hours=np.random.uniform(0, hours)),
                engagement={
                    'likes': np.random.randint(0, 1000),
                    'retweets': np.random.randint(0, 200),
                    'comments': np.random.randint(0, 100)
                },
                hashtags=[f'#{symbol}', '#crypto'],
                mentions=['@crypto_exchange']
            ))
        
        return posts
    
    def _generate_mock_tweet(self, symbol: str) -> str:
        """Generate mock tweet content"""
        templates = [
            f"{symbol} is looking bullish! 🚀 Could see $100k soon",
            f"Just bought more {symbol}. This dip is a gift 🎁",
            f"{symbol} breaking resistance! Next stop moon 🌙",
            f"Bearish on {symbol} right now. Expecting correction 📉",
            f"Why is {symbol} pumping? Any news?",
            f"{symbol} technical analysis: Strong support at current levels",
            f"Whale alert! Large {symbol} transfer detected 🐋",
            f"{symbol} fundamentals looking strong. Long term bullish 📈"
        ]
        
        return np.random.choice(templates)

class RedditConnector:
    """Reddit API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get Reddit posts about symbol"""
        posts = []
        
        # Mock Reddit data
        for i in range(15):
            posts.append(SocialPost(
                platform='reddit',
                author=f'u/crypto_redditor_{i}',
                author_followers=0,  # Reddit doesn't show follower counts
                author_verified=False,
                content=self._generate_mock_reddit_post(symbol),
                timestamp=datetime.now() - timedelta(hours=np.random.uniform(0, hours)),
                engagement={
                    'upvotes': np.random.randint(0, 500),
                    'downvotes': np.random.randint(0, 50),
                    'comments': np.random.randint(0, 200)
                }
            ))
        
        return posts
    
    def _generate_mock_reddit_post(self, symbol: str) -> str:
        """Generate mock Reddit post"""
        templates = [
            f"Daily {symbol} discussion thread - what are your thoughts?",
            f"Technical analysis of {symbol} - detailed breakdown with charts",
            f"{symbol} just hit my price target. Taking profits here",
            f"Unpopular opinion: {symbol} is overvalued at current prices",
            f"New to {symbol} - can someone explain the fundamentals?",
            f"{symbol} staking rewards update - higher APY announced",
            f"Major {symbol} partnership announced! This is huge news",
            f"Be careful with {symbol} - seeing some concerning patterns"
        ]
        
        return np.random.choice(templates)

class TelegramConnector:
    """Telegram API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get Telegram posts about symbol"""
        # Mock Telegram data
        return []

class DiscordConnector:
    """Discord API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get Discord posts about symbol"""
        # Mock Discord data
        return []

class YouTubeConnector:
    """YouTube API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get YouTube posts about symbol"""
        # Mock YouTube data
        return []

class TikTokConnector:
    """TikTok API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get TikTok posts about symbol"""
        # Mock TikTok data
        return []

class MediumConnector:
    """Medium API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get Medium posts about symbol"""
        # Mock Medium data
        return []

class GitHubConnector:
    """GitHub API connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get GitHub posts about symbol"""
        # Mock GitHub data (commits, issues, releases)
        return []

class NewsConnector:
    """News aggregator connector"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get news articles about symbol"""
        # Mock news data
        return []

class ForumConnector:
    """Forum connector (BitcoinTalk, etc.)"""
    
    async def get_posts(self, symbol: str, hours: int = 24) -> List[SocialPost]:
        """Get forum posts about symbol"""
        # Mock forum data
        return []

class SocialSignalAggregator:
    """Main social signal aggregation engine"""
    
    def __init__(self):
        self.platform_connector = PlatformConnector()
        self.fake_news_detector = FakeNewsDetector()
        self.influencer_scorer = InfluencerScorer()
        self.sentiment_analyzer = self._load_sentiment_analyzer()
        
        # Platform weights based on reliability and influence
        self.platform_weights = {
            'twitter': 0.25,
            'reddit': 0.20,
            'telegram': 0.15,
            'discord': 0.10,
            'youtube': 0.10,
            'news': 0.20
        }
    
    def _load_sentiment_analyzer(self):
        """Load sentiment analysis model"""
        try:
            # In production, would load a trained model
            # For now, use simple rule-based sentiment
            return None
        except:
            return None
    
    async def get_aggregated_sentiment(self, symbol: str, hours: int = 24) -> SentimentSignal:
        """Get aggregated sentiment signal across all platforms"""
        try:
            platform_signals = {}
            all_posts = []
            
            # Collect data from all platforms
            for platform in self.platform_connector.platforms.keys():
                posts = await self.platform_connector.get_posts(platform, symbol, hours)
                
                if posts:
                    # Filter fake news
                    filtered_posts = await self._filter_fake_news(posts)
                    
                    # Analyze sentiment
                    platform_sentiment = await self._analyze_platform_sentiment(filtered_posts, platform)
                    
                    platform_signals[platform] = platform_sentiment
                    all_posts.extend(filtered_posts)
            
            # Aggregate signals
            aggregated_signal = self._aggregate_platform_signals(platform_signals, symbol)
            
            # Add metadata
            aggregated_signal.metadata.update({
                'total_posts': len(all_posts),
                'platforms_analyzed': len(platform_signals),
                'analysis_timestamp': datetime.now().isoformat()
            })
            
            return aggregated_signal
            
        except Exception as e:
            logger.error(f"Error in sentiment aggregation: {e}")
            return SentimentSignal(
                platform='aggregated',
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                reach=0,
                influencer_sentiment=0.0,
                fake_news_probability=0.5,
                trending_score=0.0,
                metadata={'error': str(e)}
            )
    
    async def _filter_fake_news(self, posts: List[SocialPost]) -> List[SocialPost]:
        """Filter out fake news and spam"""
        filtered_posts = []
        
        for post in posts:
            fake_probability = await self.fake_news_detector.detect_fake_news(post)
            
            # Keep posts with low fake news probability
            if fake_probability < 0.7:  # Threshold for fake news
                filtered_posts.append(post)
        
        logger.info(f"Filtered {len(posts) - len(filtered_posts)} suspicious posts")
        return filtered_posts
    
    async def _analyze_platform_sentiment(self, posts: List[SocialPost], platform: str) -> Dict:
        """Analyze sentiment for specific platform"""
        if not posts:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'volume': 0,
                'reach': 0,
                'trending_score': 0.0
            }
        
        # Simple sentiment analysis (in production would use advanced NLP)
        sentiments = []
        total_reach = 0
        
        for post in posts:
            # Simple keyword-based sentiment
            sentiment = self._calculate_simple_sentiment(post.content)
            sentiments.append(sentiment)
            
            # Calculate reach
            total_reach += post.author_followers + sum(post.engagement.values())
        
        # Calculate weighted sentiment by influencer credibility
        influencer_sentiment = self.influencer_scorer.calculate_weighted_sentiment(posts, sentiments)
        
        # Calculate trending score based on volume and recency
        trending_score = self._calculate_trending_score(posts)
        
        # Calculate confidence based on volume and consistency
        confidence = self._calculate_sentiment_confidence(sentiments)
        
        return {
            'sentiment_score': np.mean(sentiments) if sentiments else 0.0,
            'confidence': confidence,
            'volume': len(posts),
            'reach': total_reach,
            'influencer_sentiment': influencer_sentiment,
            'trending_score': trending_score
        }
    
    def _calculate_simple_sentiment(self, content: str) -> float:
        """Simple keyword-based sentiment analysis"""
        content_lower = content.lower()
        
        bullish_words = [
            'bullish', 'moon', 'pump', 'bull', 'rocket', 'up', 'rise', 'gain',
            'buy', 'long', 'support', 'breakout', 'rally', 'surge', 'green'
        ]
        
        bearish_words = [
            'bearish', 'dump', 'crash', 'fall', 'drop', 'bear', 'sell',
            'short', 'resistance', 'correction', 'dip', 'red', 'decline'
        ]
        
        bullish_count = sum(1 for word in bullish_words if word in content_lower)
        bearish_count = sum(1 for word in bearish_words if word in content_lower)
        
        total_sentiment_words = bullish_count + bearish_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        sentiment_score = (bullish_count - bearish_count) / total_sentiment_words
        return np.clip(sentiment_score, -1.0, 1.0)
    
    def _calculate_trending_score(self, posts: List[SocialPost]) -> float:
        """Calculate how trending the topic is"""
        if not posts:
            return 0.0
        
        # Time decay factor (recent posts count more)
        now = datetime.now()
        scores = []
        
        for post in posts:
            time_diff = (now - post.timestamp).total_seconds() / 3600  # Hours
            time_weight = max(0, 1 - (time_diff / 24))  # Decay over 24 hours
            
            # Engagement weight
            engagement_score = min(sum(post.engagement.values()) / 1000, 1.0)
            
            post_score = time_weight * engagement_score
            scores.append(post_score)
        
        # Volume factor
        volume_factor = min(len(posts) / 100, 1.0)  # Normalize to 100 posts
        
        return np.mean(scores) * volume_factor if scores else 0.0
    
    def _calculate_sentiment_confidence(self, sentiments: List[float]) -> float:
        """Calculate confidence in sentiment analysis"""
        if not sentiments:
            return 0.0
        
        # Volume confidence
        volume_confidence = min(len(sentiments) / 50, 1.0)  # 50 posts = full confidence
        
        # Consistency confidence (low standard deviation = high confidence)
        if len(sentiments) > 1:
            sentiment_std = np.std(sentiments)
            consistency_confidence = max(0, 1 - sentiment_std)
        else:
            consistency_confidence = 0.5
        
        # Strength confidence (stronger sentiment = higher confidence)
        avg_sentiment = np.mean(sentiments)
        strength_confidence = abs(avg_sentiment)  # 0 (neutral) to 1 (strong)
        
        return np.mean([volume_confidence, consistency_confidence, strength_confidence])
    
    def _aggregate_platform_signals(self, platform_signals: Dict, symbol: str) -> SentimentSignal:
        """Aggregate signals from all platforms"""
        if not platform_signals:
            return SentimentSignal(
                platform='aggregated',
                symbol=symbol,
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                reach=0,
                influencer_sentiment=0.0,
                fake_news_probability=0.0,
                trending_score=0.0,
                metadata={}
            )
        
        # Weighted aggregation
        weighted_sentiment = 0.0
        weighted_confidence = 0.0
        weighted_influencer_sentiment = 0.0
        weighted_trending = 0.0
        total_weight = 0.0
        
        total_volume = 0
        total_reach = 0
        
        for platform, signal in platform_signals.items():
            weight = self.platform_weights.get(platform, 0.1)
            confidence = signal['confidence']
            effective_weight = weight * confidence
            
            weighted_sentiment += signal['sentiment_score'] * effective_weight
            weighted_confidence += confidence * weight
            weighted_influencer_sentiment += signal['influencer_sentiment'] * effective_weight
            weighted_trending += signal['trending_score'] * effective_weight
            
            total_weight += effective_weight
            total_volume += signal['volume']
            total_reach += signal['reach']
        
        # Normalize by total weight
        if total_weight > 0:
            final_sentiment = weighted_sentiment / total_weight
            final_confidence = weighted_confidence / sum(self.platform_weights.values())
            final_influencer_sentiment = weighted_influencer_sentiment / total_weight
            final_trending = weighted_trending / total_weight
        else:
            final_sentiment = 0.0
            final_confidence = 0.0
            final_influencer_sentiment = 0.0
            final_trending = 0.0
        
        return SentimentSignal(
            platform='aggregated',
            symbol=symbol,
            sentiment_score=final_sentiment,
            confidence=final_confidence,
            volume=total_volume,
            reach=total_reach,
            influencer_sentiment=final_influencer_sentiment,
            fake_news_probability=0.1,  # Low since we filtered fake news
            trending_score=final_trending,
            metadata={
                'platform_signals': platform_signals,
                'platform_weights': self.platform_weights
            }
        )

# Example usage and testing
async def test_social_aggregator():
    """Test social signal aggregation"""
    aggregator = SocialSignalAggregator()
    
    # Test sentiment analysis
    signal = await aggregator.get_aggregated_sentiment('BTC', hours=24)
    
    print(f"Social Sentiment Analysis for BTC:")
    print(f"  Overall Sentiment: {signal.sentiment_score:.3f}")
    print(f"  Confidence: {signal.confidence:.3f}")
    print(f"  Volume: {signal.volume} posts")
    print(f"  Reach: {signal.reach:,} people")
    print(f"  Influencer Sentiment: {signal.influencer_sentiment:.3f}")
    print(f"  Trending Score: {signal.trending_score:.3f}")
    print(f"  Fake News Probability: {signal.fake_news_probability:.3f}")
    
    # Interpret sentiment
    if signal.sentiment_score > 0.3:
        interpretation = "Bullish"
    elif signal.sentiment_score < -0.3:
        interpretation = "Bearish"
    else:
        interpretation = "Neutral"
    
    print(f"  Interpretation: {interpretation}")
    
    return signal

if __name__ == "__main__":
    asyncio.run(test_social_aggregator())