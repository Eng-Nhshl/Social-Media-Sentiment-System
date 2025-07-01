# Visualization Libraries
import matplotlib.pyplot as plt  # For static plots
import seaborn as sns  # For statistical visualizations
import plotly.express as px  # For interactive plots
import plotly.graph_objects as go  # For custom interactive plots
from wordcloud import WordCloud  # For word cloud generation

# Data manipulation
import pandas as pd  # For data handling
import numpy as np  # For numerical operations


class SentimentVisualizer:
    """
    A comprehensive visualization class for sentiment analysis results.
    
    This class provides various visualization methods including:
    - Sentiment distribution plots
    - Word clouds
    - Time series analysis
    - Confusion matrices
    - Feature importance plots
    - Category-based analysis
    - Interactive dashboards
    """
    
    def __init__(self):
        """
        Initialize the visualizer with consistent styling.
        Sets up a clean, modern look for all static plots.
        """
        # Start with matplotlib's default style for clean base
        plt.style.use("default")
        
        # Configure plot aesthetics
        plt.rcParams["figure.figsize"] = [10, 6]  # Standard figure size
        plt.rcParams["axes.grid"] = True  # Add grids
        plt.rcParams["grid.alpha"] = 0.3  # Subtle grid lines
        
        # Define color scheme for sentiment categories
        self.colors = {
            -1: "#e74c3c",  # Red for negative
            0: "#f1c40f",   # Yellow for neutral
            1: "#2ecc71"    # Green for positive
        }

    def plot_sentiment_distribution(self, sentiments, title="Sentiment Distribution"):
        """
        Create a bar plot showing the distribution of sentiment labels.
        
        Args:
            sentiments (array-like): Array of sentiment values (-1, 0, 1)
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Bar plot of sentiment distribution
        """
        # Create new figure
        plt.figure(figsize=(10, 6))
        
        # Convert to DataFrame for seaborn
        df = pd.DataFrame({"predicted_sentiment": sentiments})
        
        # Create color-coded bar plot
        sns.countplot(
            data=df,
            x="predicted_sentiment",
            hue="predicted_sentiment",  # Color bars by sentiment
            palette=self.colors,       # Use our color scheme
            legend=False              # Hide redundant legend
        )
        
        # Add labels
        plt.title(title)
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        
        return plt.gcf()

    def create_wordcloud(self, texts, title="Word Cloud"):
        """
        Generate a word cloud visualization from processed text.
        
        Args:
            texts (list): List of processed text strings
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Word cloud visualization
            
        The size of each word represents its frequency in the texts.
        Colors are randomly assigned for visual appeal.
        """
        # Generate word cloud
        wordcloud = WordCloud(
            width=800,                # Wide format
            height=400,               # Standard height
            background_color="white", # Clean background
            max_words=200            # Limit to most frequent words
        ).generate(" ".join(texts))  # Combine all texts

        # Create figure and display word cloud
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")  # Hide axes
        plt.title(title)
        
        return plt.gcf()

    def plot_sentiment_over_time(self, dates, sentiments, title="Sentiment Trends"):
        """
        Create an interactive line plot showing sentiment trends over time.
        
        Args:
            dates (array-like): Dates/timestamps for each sentiment
            sentiments (array-like): Sentiment values (-1, 0, 1)
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Interactive time series plot
        """
        # Combine data into DataFrame
        df = pd.DataFrame({"date": dates, "predicted_sentiment": sentiments})
        
        # Calculate daily average sentiment
        df = df.groupby("date")["predicted_sentiment"].mean().reset_index()

        # Create interactive line plot
        fig = px.line(
            df,
            x="date",
            y="predicted_sentiment",
            title=title,
            labels={"predicted_sentiment": "Average Sentiment", "date": "Date"},
        )
        return fig

    def plot_confusion_matrix(self, conf_matrix, labels, title="Confusion Matrix"):
        """
        Create a heatmap visualization of the model's confusion matrix.
        
        Args:
            conf_matrix (array-like): 2D confusion matrix array
            labels (list): Category labels for axes
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix heatmap
        """
        # Create new figure
        plt.figure(figsize=(10, 8))
        
        # Create annotated heatmap
        sns.heatmap(
            conf_matrix,
            annot=True,         # Show values in cells
            fmt="d",           # Display as integers
            cmap="Blues",      # Blue color scheme
            xticklabels=labels, # X-axis labels
            yticklabels=labels, # Y-axis labels
        )
        
        # Add labels
        plt.title(title)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        return plt.gcf()

    def plot_feature_importance(self, feature_names, importance_scores, top_n=20):
        """
        Create a bar plot of the most important features in the model.
        
        Args:
            feature_names (array-like): Names of all features
            importance_scores (array-like): Importance scores for each feature
            top_n (int): Number of top features to display
            
        Returns:
            matplotlib.figure.Figure: Feature importance bar plot
        """
        # Get indices of top N most important features
        indices = np.argsort(importance_scores)[-top_n:]  # Sort and get top N
        top_features = feature_names[indices]
        top_scores = importance_scores[indices]

        # Create bar plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_scores, y=top_features)
        plt.title(f"Top {top_n} Most Important Features")
        plt.xlabel("Importance Score")
        plt.ylabel("Feature")
        return plt.gcf()

    def plot_sentiment_by_category(
        self, categories, sentiments, title="Sentiment by Category"
    ):
        """
        Create a bar plot showing average sentiment for each category.
        
        Args:
            categories (array-like): Category labels for each data point
            sentiments (array-like): Sentiment values (-1, 0, 1)
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Category-wise sentiment bar plot
        """
        # Combine data and calculate averages
        df = pd.DataFrame({"category": categories, "predicted_sentiment": sentiments})
        avg_sentiment = (
            df.groupby("category")["predicted_sentiment"].mean().sort_values()
        )

        # Create color-coded bar plot
        plt.figure(figsize=(12, 6))
        avg_sentiment.plot(kind="bar", color=self.colors)
        
        # Add labels and adjust layout
        plt.title(title)
        plt.xlabel("Category")
        plt.ylabel("Average Sentiment Score")
        plt.xticks(rotation=45)  # Rotate category labels for readability
        plt.tight_layout()       # Adjust spacing
        return plt.gcf()

    def create_interactive_dashboard(self, data):
        """
        Create an interactive dashboard combining multiple visualizations.
        
        Args:
            data (pandas.DataFrame): DataFrame containing:
                - predicted_sentiment: Sentiment values
                - date: Timestamps
                
        Returns:
            plotly.graph_objects.Figure: Interactive dashboard
        """
        # Create sentiment distribution histogram
        fig1 = px.histogram(
            data,
            x="predicted_sentiment",
            title="Sentiment Distribution",
            color="predicted_sentiment",  # Color by sentiment
        )

        # Create time series plot
        fig2 = px.line(
            data,
            x="date",
            y="predicted_sentiment",
            title="Sentiment Trends Over Time"
        )

        # Initialize dashboard layout
        dashboard = go.Figure()

        # Combine plots into dashboard
        for trace in fig1.data:  # Add distribution plot
            dashboard.add_trace(trace)
        for trace in fig2.data:  # Add time series plot
            dashboard.add_trace(trace)

        # Configure dashboard layout
        dashboard.update_layout(
            title="Sentiment Analysis Dashboard",
            height=800,        # Tall enough for both plots
            showlegend=True   # Show plot legends
        )

        return dashboard
