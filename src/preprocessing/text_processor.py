# Standard library imports
import re  # For regular expressions and text pattern matching

# Natural Language Processing imports
import nltk  # Core NLP library
from nltk.tokenize import word_tokenize  # For splitting text into words
from nltk.corpus import stopwords  # For removing common words
from nltk.stem import WordNetLemmatizer  # For word normalization

# Sentiment Analysis imports
from textblob import TextBlob  # For additional sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # For social media sentiment


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for social media content.
    
    This class handles all text processing steps including:
    - Text cleaning and normalization
    - Tokenization
    - Stopword removal
    - Lemmatization
    - Sentiment scoring using both VADER and TextBlob
    """
    
    def __init__(self):
        """
        Initialize the text processor and ensure all required NLTK data is available.
        Downloads required NLTK data silently if not already present.
        """
        # First, check if required NLTK data is available
        try:
            nltk.data.find("tokenizers/punkt")  # For sentence tokenization
            nltk.data.find("corpora/stopwords")  # For stopword removal
            nltk.data.find("corpora/wordnet")  # For lemmatization
        except LookupError:
            # If any data is missing, download it silently
            import io
            from contextlib import redirect_stdout
            with io.StringIO() as buf, redirect_stdout(buf):
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)

        # Initialize NLP tools
        self.lemmatizer = WordNetLemmatizer()  # For converting words to base form
        self.stop_words = set(stopwords.words("english"))  # Cache stopwords for efficiency
        self.vader = SentimentIntensityAnalyzer()  # Initialize VADER sentiment analyzer

    def clean_text(self, text):
        """
        Clean and normalize social media text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned and normalized text
            
        Processing steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove @mentions
        4. Remove hashtag symbols (keep text)
        5. Remove special characters and numbers
        """
        # Handle non-string input
        if not isinstance(text, str):
            return ""

        # Convert to lowercase for consistency
        text = text.lower()

        # Remove URLs (http, https, www)
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove @username mentions (common in social media)
        text = re.sub(r"@\w+", "", text)

        # Remove hashtag symbols but keep the text content
        text = re.sub(r"#", "", text)

        # Remove all special characters and numbers, keep only letters and spaces
        text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Remove extra whitespace and return
        return text.strip()

    def tokenize(self, text):
        """
        Split text into individual words while preserving sentence structure.
        
        Args:
            text (str): Cleaned text to tokenize
            
        Returns:
            list: List of individual words/tokens
        """
        return word_tokenize(text, preserve_line=True)  # Keep sentence boundaries

    def remove_stopwords(self, tokens):
        """
        Remove common English words that don't carry significant meaning.
        
        Args:
            tokens (list): List of word tokens
            
        Returns:
            list: Filtered tokens with stopwords removed
            
        Example stopwords: 'the', 'is', 'at', 'which', etc.
        """
        return [token for token in tokens if token not in self.stop_words]

    def lemmatize(self, tokens):
        """
        Convert words to their base/dictionary form.
        
        Args:
            tokens (list): List of word tokens
            
        Returns:
            list: Lemmatized tokens
            
        Examples:
        - 'running' -> 'run'
        - 'better' -> 'good'
        - 'cities' -> 'city'
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def get_sentiment_scores(self, text):
        """
        Analyze text sentiment using both VADER and TextBlob.
        
        VADER is optimized for social media content, while TextBlob
        provides additional polarity and subjectivity scores.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Combined sentiment scores:
                - vader_compound: Overall sentiment (-1 to 1)
                - vader_pos: Positive sentiment strength
                - vader_neu: Neutral sentiment strength
                - vader_neg: Negative sentiment strength
                - textblob_polarity: TextBlob sentiment (-1 to 1)
                - textblob_subjectivity: Opinion strength (0 to 1)
        """
        # Get VADER scores (specialized for social media)
        vader_scores = self.vader.polarity_scores(text)

        # Get TextBlob scores (traditional sentiment analysis)
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment

        # Combine both sentiment scores
        return {
            "vader_compound": vader_scores["compound"],  # Overall score
            "vader_pos": vader_scores["pos"],  # Positive component
            "vader_neu": vader_scores["neu"],  # Neutral component
            "vader_neg": vader_scores["neg"],  # Negative component
            "textblob_polarity": textblob_sentiment.polarity,  # TextBlob sentiment
            "textblob_subjectivity": textblob_sentiment.subjectivity,  # Opinion strength
        }

    def process_text(self, text):
        """
        Execute complete text processing pipeline.
        
        This method orchestrates all text processing steps in the correct order:
        1. Clean and normalize text
        2. Get sentiment scores (before tokenization)
        3. Tokenize into words
        4. Remove stopwords
        5. Lemmatize tokens
        
        Args:
            text (str): Raw input text
            
        Returns:
            dict: Processed results containing:
                - processed_text: Final cleaned and normalized text
                - tokens: List of processed tokens
                - sentiment_scores: VADER and TextBlob scores
        """
        # Step 1: Clean and normalize the text
        cleaned_text = self.clean_text(text)

        # Step 2: Get sentiment scores before breaking into tokens
        # (sentiment analyzers work better on complete sentences)
        sentiment_scores = self.get_sentiment_scores(cleaned_text)

        # Step 3: Break text into tokens
        tokens = self.tokenize(cleaned_text)

        # Step 4: Remove common words that don't add meaning
        tokens = self.remove_stopwords(tokens)

        # Step 5: Convert words to their base form
        tokens = self.lemmatize(tokens)

        # Return all processed data
        return {
            "processed_text": " ".join(tokens),  # Recombine tokens into text
            "tokens": tokens,  # Keep tokens for word clouds/analysis
            "sentiment_scores": sentiment_scores,  # Include sentiment analysis
        }
