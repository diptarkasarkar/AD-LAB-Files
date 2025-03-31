from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tweepy
from textblob import TextBlob
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="API for fetching tweets and performing sentiment analysis",
    version="1.0.0"
)

# Twitter API credentials
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")

# Initialize Twitter client
client = tweepy.Client(
    bearer_token=TWITTER_BEARER_TOKEN,
    consumer_key=TWITTER_API_KEY,
    consumer_secret=TWITTER_API_SECRET,
    access_token=TWITTER_ACCESS_TOKEN,
    access_token_secret=TWITTER_ACCESS_TOKEN_SECRET
)

# Request model
class TweetRequest(BaseModel):
    keyword: str
    count: int = 10  # Default to 10 tweets if not specified

# Response models
class SentimentResult(BaseModel):
    text: str
    polarity: float
    sentiment: str  # "positive", "negative", or "neutral"

class TweetResponse(BaseModel):
    keyword: str
    count: int
    tweets: List[SentimentResult]
    positive_count: int
    negative_count: int
    neutral_count: int

# Helper function to analyze sentiment
def analyze_sentiment(text: str) -> Dict:
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0:
        sentiment = "positive"
    elif polarity < 0:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    return {
        "text": text,
        "polarity": polarity,
        "sentiment": sentiment
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to Twitter Sentiment Analysis API",
        "status": "running",
        "endpoints": {
            "fetch_tweets": {
                "method": "POST",
                "path": "/fetch_tweets",
                "description": "Fetch tweets and analyze sentiment",
                "required_params": {
                    "keyword": "string",
                    "count": "integer (optional, default=10)"
                }
            }
        }
    }

# Fetch tweets endpoint
@app.post("/fetch_tweets", response_model=TweetResponse, tags=["Tweets"])
async def fetch_tweets(request: TweetRequest):
    try:
        # Fetch recent tweets
        tweets = client.search_recent_tweets(
            query=request.keyword,
            max_results=request.count,
            tweet_fields=["text"]
        )
        
        if not tweets.data:
            return {
                "keyword": request.keyword,
                "count": 0,
                "tweets": [],
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0
            }
        
        # Analyze sentiment for each tweet
        analyzed_tweets = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for tweet in tweets.data:
            result = analyze_sentiment(tweet.text)
            analyzed_tweets.append(result)
            
            # Count sentiments
            if result["sentiment"] == "positive":
                positive_count += 1
            elif result["sentiment"] == "negative":
                negative_count += 1
            else:
                neutral_count += 1
        
        return {
            "keyword": request.keyword,
            "count": len(analyzed_tweets),
            "tweets": analyzed_tweets,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count
        }
        
    except tweepy.TweepyException as e:
        raise HTTPException(status_code=400, detail=f"Twitter API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)