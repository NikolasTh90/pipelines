import os
import sys
import subprocess
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime

# Try to import TextBlob, install if not available
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob not found. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "textblob"])
        subprocess.check_call([sys.executable, "-m", "textblob.download_corpora"])
        from textblob import TextBlob
        TEXTBLOB_AVAILABLE = True
        print("TextBlob installed successfully!")
    except Exception as e:
        print(f"Failed to install TextBlob: {str(e)}")
        print("The filter will run in limited mode.")

class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to
        pipelines: List[str] = ["*"]
        
        # Assign a priority level to the filter pipeline
        priority: int = 0
        
        # Custom configuration for sentiment analysis
        min_sentiment_threshold: float = -1.0  # Values from -1.0 (most negative) to 1.0 (most positive)
        log_level: str = "info"
        block_negative_messages: bool = False
        track_subjectivity: bool = True

    def __init__(self):
        # This filter analyzes sentiment of messages and can optionally block very negative content
        self.type = "filter"
        self.name = "Sentiment Analysis Filter"

        # Initialize with default configuration
        self.valves = self.Valves()
        
        # For tracking sentiment statistics
        self.total_messages = 0
        self.sentiment_sum = 0.0
        self.subjectivity_sum = 0.0
        self.most_negative_message = {"text": "", "score": 0.0, "timestamp": None}
        self.most_positive_message = {"text": "", "score": 0.0, "timestamp": None}
        
        # Flag to track if we're in limited mode
        self.limited_mode = not TEXTBLOB_AVAILABLE

    async def on_startup(self):
        # This function is called when the server is started
        print(f"Sentiment Analysis Filter starting up")
        
        if self.limited_mode:
            print("WARNING: Running in limited mode - TextBlob is not available")
            print("Sentiment analysis will not be performed")
            print("Please manually install TextBlob with: pip install textblob")
            print("and download corpora with: python -m textblob.download_corpora")
        else:
            print(f"Minimum sentiment threshold: {self.valves.min_sentiment_threshold}")
            print(f"Blocking negative messages: {self.valves.block_negative_messages}")
            
            # Verify TextBlob is working
            try:
                test_blob = TextBlob("Test message")
                test_sentiment = test_blob.sentiment.polarity
                print(f"TextBlob initialized successfully (test sentiment: {test_sentiment})")
            except Exception as e:
                print(f"TextBlob initialization error: {str(e)}")
                self.limited_mode = True

    async def on_shutdown(self):
        # This function is called when the server is stopped
        print("Sentiment Analysis Filter shutting down")
        if not self.limited_mode and self.total_messages > 0:
            avg_sentiment = self.sentiment_sum / self.total_messages
            avg_subjectivity = self.subjectivity_sum / self.total_messages
            print(f"Statistics for session:")
            print(f"  Total messages analyzed: {self.total_messages}")
            print(f"  Average sentiment: {avg_sentiment:.2f}")
            print(f"  Average subjectivity: {avg_subjectivity:.2f}")
            if self.most_negative_message["text"]:
                print(f"  Most negative message: \"{self.most_negative_message['text']}\" (score: {self.most_negative_message['score']:.2f})")
            if self.most_positive_message["text"]:
                print(f"  Most positive message: \"{self.most_positive_message['text']}\" (score: {self.most_positive_message['score']:.2f})")

    async def on_valves_updated(self):
        # This function is called when the valves are updated
        if not self.limited_mode:
            print(f"Valves updated: min_sentiment_threshold = {self.valves.min_sentiment_threshold}")
            print(f"Blocking negative messages: {self.valves.block_negative_messages}")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """
        Analyzes the sentiment of incoming messages and optionally blocks very negative content.
        
        Args:
            body: The request body containing the messages
            user: Optional user information
            
        Returns:
            The original body if the message passes the sentiment check
            
        Raises:
            Exception: If the message sentiment is below the configured threshold and blocking is enabled
        """
        if "messages" not in body or not body["messages"]:
            return body
            
        user_message = body["messages"][-1]["content"]
        username = user.get("username", "unknown") if user else "unknown"
        
        # If we're in limited mode, just log the message and pass it through
        if self.limited_mode:
            print(f"Message from {username} (sentiment analysis disabled): {user_message[:30]}...")
            return body
        
        # Use TextBlob to analyze sentiment
        try:
            blob = TextBlob(user_message)
            sentiment_score = blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)
            subjectivity_score = blob.sentiment.subjectivity  # Range from 0 (objective) to 1 (subjective)
            
            # Update statistics
            self.total_messages += 1
            self.sentiment_sum += sentiment_score
            self.subjectivity_sum += subjectivity_score
            
            # Track most negative/positive messages
            if sentiment_score < self.most_negative_message["score"] or not self.most_negative_message["text"]:
                self.most_negative_message = {
                    "text": user_message,
                    "score": sentiment_score,
                    "timestamp": datetime.now()
                }
            
            if sentiment_score > self.most_positive_message["score"] or not self.most_positive_message["text"]:
                self.most_positive_message = {
                    "text": user_message,
                    "score": sentiment_score,
                    "timestamp": datetime.now()
                }
            
            # Log sentiment analysis results
            sentiment_label = "positive" if sentiment_score > 0 else "negative" if sentiment_score < 0 else "neutral"
            print(f"Message from {username}: sentiment={sentiment_score:.2f} ({sentiment_label}), subjectivity={subjectivity_score:.2f}")
            
            # Check if message should be blocked based on sentiment
            if self.valves.block_negative_messages and sentiment_score < self.valves.min_sentiment_threshold:
                raise Exception(f"Message blocked due to very negative sentiment (score: {sentiment_score:.2f})")
        
        except Exception as e:
            if "blocked due to very negative sentiment" in str(e):
                # Re-raise the intentional block
                raise
            else:
                # Log other errors but let the message through
                print(f"Error analyzing sentiment: {str(e)}")
                
        # Pass the message through if it wasn't blocked
        return body