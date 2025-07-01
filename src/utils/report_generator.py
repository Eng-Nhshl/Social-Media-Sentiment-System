# Data manipulation and analysis
import pandas as pd  # For DataFrame operations

# System utilities
import os          # For file/directory operations
from datetime import datetime  # For timestamps
import json        # For JSON data handling

# Visualization
import plotly.io as pio  # For saving interactive plots


class ReportGenerator:
    """
    A comprehensive report generator for sentiment analysis results.
    
    This class handles the creation of detailed reports including:
    - Summary statistics
    - Data visualizations
    - Model evaluation metrics
    - Feature analysis
    - Excel exports
    
    Reports are saved with timestamps and can include both static
    and interactive visualizations.
    """
    
    def __init__(self, output_dir='reports'):
        """
        Initialize the report generator.
        
        Args:
            output_dir (str): Name of directory to store reports,
                              relative to project root
        
        The constructor:
        1. Determines the project root directory
        2. Creates absolute path for output directory
        3. Ensures output directory exists
        """
        # Find project root by walking up from current file
        self.project_root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
        
        # Create output directory path and ensure it exists
        self.output_dir = os.path.join(self.project_root, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_summary_stats(self, data):
        """
        Generate key statistics from the analyzed sentiment data.
        
        Args:
            data (pandas.DataFrame): DataFrame containing:
                - date: Timestamp of each post
                - predicted_sentiment: Sentiment label
                - sentiment_probability: Confidence score
                
        Returns:
            dict: Summary statistics including:
                - total_posts: Total number of analyzed posts
                - sentiment_distribution: Count of each sentiment
                - average_sentiment_score: Mean sentiment score
                - sentiment_trends: Daily sentiment counts
        """
        # Calculate daily sentiment trends
        sentiment_trends = {}
        grouped_data = data.groupby(data['date'].dt.date)['predicted_sentiment'].value_counts()
        
        # Convert trends to JSON-friendly format
        for (date, sentiment), count in grouped_data.items():
            date_str = date.strftime('%Y-%m-%d')  # Format date as string
            if date_str not in sentiment_trends:
                sentiment_trends[date_str] = {}
            sentiment_trends[date_str][sentiment] = int(count)

        # Compile all summary statistics
        summary = {
            'total_posts': len(data),  # Total analyzed posts
            'sentiment_distribution': data['predicted_sentiment'].value_counts().to_dict(),
            'average_sentiment_score': float(data['sentiment_probability'].mean()),
            'sentiment_trends': sentiment_trends  # Daily breakdown
        }
        return summary

    def generate_feature_analysis(self, data):
        """
        Analyze important textual features in the sentiment data.
        
        This method extracts key insights about the text content:
        1. Most frequent words in positive sentiment texts
        2. Most frequent words in negative sentiment texts
        3. Average text length for each sentiment category
        
        Args:
            data (pandas.DataFrame): DataFrame containing:
                - predicted_sentiment: Sentiment labels
                - processed_text: Preprocessed text content
                
        Returns:
            dict: Feature analysis results including:
                - top_positive_words: 20 most common words in positive texts
                - top_negative_words: 20 most common words in negative texts
                - sentiment_by_length: Average text length per sentiment
        """
        feature_analysis = {
            # Find most common words in positive texts
            'top_positive_words': data[
                data['predicted_sentiment'] == 'positive'
            ]['processed_text'].str.split()  # Split into words
            .explode()  # One row per word
            .value_counts()  # Count frequencies
            [:20]  # Top 20 words
            .to_dict(),
            
            # Find most common words in negative texts
            'top_negative_words': data[
                data['predicted_sentiment'] == 'negative'
            ]['processed_text'].str.split()
            .explode()
            .value_counts()
            [:20]
            .to_dict(),
            
            # Calculate average text length by sentiment
            'sentiment_by_length': data.groupby('predicted_sentiment')[
                'processed_text'
            ].apply(lambda x: x.str.len().mean()).to_dict()
        }
        return feature_analysis

    def save_visualizations(self, visualizations):
        """
        Save all visualizations to files with appropriate formats.
        
        This method handles both static (matplotlib) and interactive
        (plotly) visualizations, saving them as PNG or HTML files
        respectively.
        
        Args:
            visualizations (dict): Dictionary mapping visualization
                                  names to figure objects
                
        Returns:
            dict: Mapping of visualization names to saved file paths
        """
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_paths = {}

        # Process each visualization
        for name, fig in visualizations.items():
            if hasattr(fig, 'savefig'):  # Matplotlib figure
                # Save as high-quality PNG
                path = os.path.join(self.output_dir, f'{name}_{timestamp}.png')
                fig.savefig(
                    path,
                    dpi=300,          # High resolution
                    bbox_inches='tight'  # No extra whitespace
                )
                viz_paths[name] = path
                
            elif hasattr(fig, 'write_html'):  # Plotly figure
                # Save as interactive HTML
                path = os.path.join(self.output_dir, f'{name}_{timestamp}.html')
                fig.write_html(path)
                viz_paths[name] = path

        return viz_paths

    def generate_html_report(self, analyzed_data, evaluation_results, visualizations):
        """
        Generate a comprehensive HTML report of sentiment analysis results.
        
        This method creates a well-formatted HTML report that includes:
        1. Summary statistics of sentiment distribution
        2. Model evaluation metrics and performance
        3. Feature analysis and insights
        4. Interactive and static visualizations
        
        Args:
            analyzed_data (pandas.DataFrame): Processed data with sentiments
            evaluation_results (dict): Model performance metrics
            visualizations (dict): Dictionary of visualization figures
            
        Returns:
            str: Path to the generated HTML report file
        """
        # Create unique timestamp for the report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(
            self.output_dir,
            f'sentiment_analysis_report_{timestamp}.html'
        )

        # Generate all report components
        summary_stats = self.generate_summary_stats(analyzed_data)
        feature_analysis = self.generate_feature_analysis(analyzed_data)
        viz_paths = self.save_visualizations(visualizations)

        # Create HTML content with modern styling
        html_content = f"""
        <html>
        <head>
            <title>Sentiment Analysis Report - {timestamp}</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    line-height: 1.6;
                    color: #333;
                }}
                .section {{ 
                    margin-bottom: 30px;
                    padding: 20px;
                    background: #fff;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .visualization {{ 
                    margin: 20px 0;
                    padding: 15px;
                    background: #f8f9fa;
                    border-radius: 4px;
                }}
                table {{ 
                    border-collapse: collapse;
                    width: 100%;
                    margin: 15px 0;
                }}
                th, td {{ 
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{ 
                    background-color: #f8f9fa;
                    font-weight: 600;
                }}
                h1, h2 {{ 
                    color: #2c3e50;
                    margin-top: 0;
                }}
                pre {{ 
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 4px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            <h1>Sentiment Analysis Report</h1>
            
            <div class="section">
                <h2>Summary Statistics</h2>
                <pre>{json.dumps(summary_stats, indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Model Evaluation</h2>
                <pre>{json.dumps(evaluation_results, indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Feature Analysis</h2>
                <pre>{json.dumps(feature_analysis, indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                {self._generate_visualization_html(viz_paths)}
            </div>
        </body>
        </html>
        """

        # Save the report with UTF-8 encoding
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_path

    def _generate_visualization_html(self, viz_paths):
        """
        Generate HTML code for embedding visualizations.
        
        This internal method handles both types of visualizations:
        1. Interactive Plotly figures (embedded HTML)
        2. Static matplotlib figures (image tags)
        
        Args:
            viz_paths (dict): Mapping of visualization names to file paths
            
        Returns:
            str: HTML code containing all visualizations
        """
        html = ""
        for name, path in viz_paths.items():
            if path.endswith('.html'):  # Interactive Plotly figure
                # Read and embed the interactive visualization
                with open(path, 'r', encoding='utf-8') as f:
                    viz_content = f.read()
                html += f'''
                <div class="visualization">
                    <h3>{name}</h3>
                    {viz_content}
                </div>'''
            else:  # Static matplotlib figure
                # Include as responsive image
                html += f'''
                <div class="visualization">
                    <h3>{name}</h3>
                    <img src="{path}" alt="{name}" style="max-width:100%;height:auto;">
                </div>'''
        return html

    def save_to_excel(self, analyzed_data, sheet_name='Sentiment Analysis'):
        """
        Save detailed analysis results to an Excel file.
        
        This method exports the complete analyzed dataset to Excel
        for further analysis or reporting purposes.
        
        Args:
            analyzed_data (pandas.DataFrame): Complete analysis results
            sheet_name (str): Name for the Excel worksheet
            
        Returns:
            str: Path to the generated Excel file
        """
        # Create unique timestamp for the file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        excel_path = os.path.join(
            self.output_dir,
            f'sentiment_analysis_results_{timestamp}.xlsx'
        )
        
        # Save to Excel without row indices
        with pd.ExcelWriter(excel_path) as writer:
            analyzed_data.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False  # Don't include row numbers
            )
            
        return excel_path
