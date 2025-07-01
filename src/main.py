# Required libraries
import pandas as pd  # For data manipulation and analysis
import os  # For file and directory operations

# Add the project root directory to Python path for imports
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import custom modules for sentiment analysis pipeline
from src.preprocessing.text_processor import (
    TextPreprocessor,
)  # Handles text cleaning and initial sentiment scoring
from src.modeling.sentiment_model import SentimentModel  # ML model for sentiment prediction
from src.visualization.visualizer import SentimentVisualizer  # Creates plots and dashboards
from src.utils.report_generator import ReportGenerator  # Generates analysis reports


class SentimentAnalysisSystem:
    """
    Main class that orchestrates the entire sentiment analysis pipeline.
    Coordinates data loading, preprocessing, model training, visualization, and reporting.
    """

    def __init__(self, data_path=None, model_path=None):
        """
        Initialize the sentiment analysis system.

        Args:
            data_path (str, optional): Path to input CSV file. If None, uses default path.
            model_path (str, optional): Path to save/load ML models. If None, uses default path.
        """
        # Get absolute path to project root directory for consistent file operations
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set up file paths, using provided paths or defaults
        self.data_path = data_path or os.path.join(
            self.project_root,
            "data",
            "social_media_data.csv",  # Default data file location
        )
        self.model_path = model_path or os.path.join(
            self.project_root,
            "models",
            "sentiment_model",  # Default model save location
        )

        # Initialize all pipeline components
        self.preprocessor = (
            TextPreprocessor()
        )  # For text cleaning and initial sentiment
        self.model = SentimentModel(model_path=self.model_path)  # ML model component
        self.visualizer = SentimentVisualizer()  # For creating visualizations

        # Set up report generator with proper output directory
        reports_dir = os.path.join(self.project_root, "reports")
        self.report_generator = ReportGenerator(
            output_dir=reports_dir
        )  # For generating reports

    def load_data(self, file_path=None):
        """
        Load and prepare the social media dataset.

        Args:
            file_path (str, optional): Path to CSV file. If None, uses default path.

        Returns:
            pandas.DataFrame: Loaded dataset with proper date formatting
        """
        # Use provided path or default
        file_path = file_path or self.data_path

        # Read CSV file into pandas DataFrame
        df = pd.read_csv(file_path)

        # If date column exists, convert it to datetime for time-based analysis
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def preprocess_data(self, data):
        """
        Clean and prepare text data for sentiment analysis.

        Args:
            data (pandas.DataFrame): Raw input data with 'text' column

        Returns:
            pandas.DataFrame: Processed data with additional columns for cleaned text and sentiment scores
        """
        # List to store processed results for each text
        processed_data = []

        # Process each text entry individually
        for text in data["text"]:
            result = self.preprocessor.process_text(
                text
            )  # Clean and get initial sentiment
            processed_data.append(result)

        # Convert processed results to DataFrame for easier handling
        processed_df = pd.DataFrame(processed_data)

        # Extract VADER and TextBlob sentiment scores into separate DataFrame
        sentiment_scores = pd.DataFrame([d["sentiment_scores"] for d in processed_data])

        # Combine original data, processed text, and sentiment scores
        return pd.concat([data, processed_df, sentiment_scores], axis=1)

    def train_model(self, data, labels):
        """
        Train the ML model on processed text data.

        Args:
            data (array-like): Processed text data for training
            labels (array-like): Sentiment labels for training

        Returns:
            dict: Model evaluation metrics
        """
        # Train model and get evaluation metrics
        evaluation = self.model.train(data, labels)

        # Save trained model for future use
        self.model.save_model()
        return evaluation

    def analyze_sentiments(self, data):
        """
        Perform complete sentiment analysis on input data.

        Args:
            data (pandas.DataFrame): Input data with 'text' column

        Returns:
            pandas.DataFrame: Analyzed data with sentiment predictions and probabilities
        """
        # First step: clean and prepare the text data
        processed_data = self.preprocess_data(data)

        def get_sentiment_label(scores):
            """
            Convert VADER compound score to sentiment label.
            1 = positive, 0 = neutral, -1 = negative
            """
            compound = scores["vader_compound"]
            if compound >= 0.05:
                return 1  # positive sentiment
            elif compound <= -0.05:
                return -1  # negative sentiment
            else:
                return 0  # neutral sentiment

        # Initial sentiment analysis using VADER scores
        processed_data["predicted_sentiment"] = processed_data.apply(
            lambda row: get_sentiment_label(
                {
                    "vader_compound": row["vader_compound"],
                    "vader_pos": row["vader_pos"],
                    "vader_neg": row["vader_neg"],
                    "vader_neu": row["vader_neu"],
                }
            ),
            axis=1,
        )

        # Use ML model for prediction if we have enough data
        if len(processed_data) >= 6:  # Minimum size for train/test split
            # Get ML model predictions and probabilities
            predictions = self.model.predict(processed_data["processed_text"])
            probabilities = self.model.predict_proba(processed_data["processed_text"])
            # Update sentiment predictions with ML model results
            processed_data["predicted_sentiment"] = predictions
            # Store prediction confidence scores
            processed_data["sentiment_probability"] = [max(p) for p in probabilities]
        else:
            # For small datasets, use VADER sentiment strength as probability
            processed_data["sentiment_probability"] = processed_data[
                "vader_compound"
            ].abs()

        return processed_data

    def generate_visualizations(self, analyzed_data):
        """
        Create a suite of visualizations for the analyzed data.

        Args:
            analyzed_data (pandas.DataFrame): Data with sentiment predictions

        Returns:
            dict: Collection of visualization objects (matplotlib figures and plotly graphs)
        """
        # Dictionary to store all visualizations
        visualizations = {}

        # Create distribution plot of sentiment predictions
        visualizations["sentiment_dist"] = self.visualizer.plot_sentiment_distribution(
            analyzed_data["predicted_sentiment"]
        )

        # Generate word cloud from processed text
        visualizations["wordcloud"] = self.visualizer.create_wordcloud(
            analyzed_data["processed_text"]
        )

        # If we have date information, create time-based analysis
        if "date" in analyzed_data.columns:
            visualizations["sentiment_trends"] = (
                self.visualizer.plot_sentiment_over_time(
                    analyzed_data["date"], analyzed_data["predicted_sentiment"]
                )
            )

        # Create interactive dashboard with all metrics
        visualizations["dashboard"] = self.visualizer.create_interactive_dashboard(
            analyzed_data
        )

        return visualizations

    def run_analysis(self, data_path=None, generate_report=True):
        """
        Execute the complete sentiment analysis pipeline.

        This is the main method that orchestrates the entire analysis process:
        1. Loads and preprocesses the data
        2. Analyzes sentiments using VADER and ML
        3. Generates visualizations
        4. Trains/updates the ML model
        5. Creates comprehensive reports

        Args:
            data_path (str, optional): Path to input data file
            generate_report (bool): Whether to create HTML and Excel reports

        Returns:
            dict: Complete analysis results including:
                - analyzed_data: DataFrame with predictions
                - visualizations: All generated plots
                - evaluation: Model performance metrics
                - report_paths: Paths to generated reports
        """
        # Step 1: Load the dataset
        data = self.load_data(data_path)

        # Step 2: Process text and analyze sentiments
        analyzed_data = self.analyze_sentiments(data)

        # Step 3: Create all visualizations
        visualizations = self.generate_visualizations(analyzed_data)

        # Step 4: Train/update ML model if we have sufficient data
        sentiment_counts = analyzed_data["predicted_sentiment"].value_counts()
        min_samples_per_class = 3  # Minimum samples needed per sentiment class

        # Check if we have enough data for proper model training
        if len(sentiment_counts) > 1 and all(
            count >= min_samples_per_class for count in sentiment_counts
        ):
            # Full training with hyperparameter tuning
            evaluation_results = self.model.train(
                analyzed_data["processed_text"],
                analyzed_data["predicted_sentiment"],
                tune_hyperparameters=True,
            )
            self.model.save_model()  # Save the optimized model
        else:
            # Simple training for small datasets
            evaluation_results = self.model.train(
                analyzed_data["processed_text"],
                analyzed_data["predicted_sentiment"],
                tune_hyperparameters=False,
            )
            self.model.save_model()  # Save the basic model

        # Compile all results
        results = {
            "analyzed_data": analyzed_data,
            "visualizations": visualizations,
            "evaluation": evaluation_results,
        }

        # Step 5: Generate reports if requested
        if generate_report:
            # Create detailed HTML report with interactive visualizations
            report_path = self.report_generator.generate_html_report(
                analyzed_data, evaluation_results, visualizations
            )
            # Export data to Excel for further analysis
            excel_path = self.report_generator.save_to_excel(analyzed_data)
            # Add report paths to results
            results["report_paths"] = {
                "html_report": report_path,
                "excel_report": excel_path,
            }

        return results


# This section runs when the script is executed directly (not imported as a module)
if __name__ == "__main__":
    # Create an instance of the sentiment analysis system
    # Uses default paths for data and model files
    sentiment_system = SentimentAnalysisSystem()

    # Execute the complete analysis pipeline
    # This will load data, analyze sentiments, create visualizations, and generate reports
    results = sentiment_system.run_analysis()

    # Ensure the reports directory exists
    reports_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"
    )
    os.makedirs(reports_dir, exist_ok=True)

    # Save the analyzed data to a CSV file for external use
    results["analyzed_data"].to_csv(
        os.path.join(reports_dir, "sentiment_analysis_results.csv"),
        index=False,  # Don't include pandas index in output
    )

    # Save all visualizations in appropriate formats
    for name, fig in results["visualizations"].items():
        if hasattr(fig, "savefig"):  # For matplotlib/seaborn figures
            fig.savefig(os.path.join(reports_dir, f"{name}.png"))
        elif hasattr(fig, "write_html"):  # For plotly figures
            fig.write_html(os.path.join(reports_dir, f"{name}.html"))
