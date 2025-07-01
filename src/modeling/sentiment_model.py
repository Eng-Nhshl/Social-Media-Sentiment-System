# Feature extraction and text processing
from sklearn.feature_extraction.text import TfidfVectorizer  # For converting text to numerical features

# Machine learning models
from sklearn.linear_model import LogisticRegression  # Main classifier
from sklearn.model_selection import train_test_split, GridSearchCV  # For model training and tuning
from sklearn.ensemble import RandomForestClassifier  # Alternative classifier option

# Model evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

# Utilities
import numpy as np  # For numerical operations
import joblib      # For model persistence
import os         # For file operations


class SentimentModel:
    """
    A sentiment analysis model using machine learning.
    
    This class implements a complete sentiment analysis pipeline including:
    - Text vectorization using TF-IDF
    - Model training with hyperparameter tuning
    - Prediction with probability estimates
    - Model persistence (save/load)
    
    The model uses a logistic regression classifier and is initialized
    with a small set of labeled examples to provide basic functionality
    even before training on actual data.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the sentiment model with basic components and seed data.
        
        Args:
            model_path (str, optional): Path for model persistence
            
        The initialization:
        1. Creates a TF-IDF vectorizer (max 5000 features)
        2. Creates a logistic regression classifier
        3. Trains on seed data to provide basic functionality
        """
        # Initialize text vectorizer with reasonable defaults
        self.vectorizer = TfidfVectorizer(
            max_features=5000  # Limit vocabulary size for efficiency
        )
        
        # Initialize classifier
        self.model = LogisticRegression()
        
        # Create balanced seed dataset with example sentiments
        training_texts = [
            # Positive examples
            "This is amazing! Love it!",
            "Great product, highly recommend",
            "Best experience ever!",
            "Excellent quality and service",
            "Really happy with my purchase",
            # Negative examples
            "This is terrible, don't buy",
            "Worst product ever",
            "Very disappointed",
            "Poor quality and service",
            "Complete waste of money",
            # Neutral examples
            "It's okay, nothing special",
            "Could be better",
            "Not bad, not great",
            "Average product",
            "Neutral experience"
        ]
        
        # Labels: 1 (positive), -1 (negative), 0 (neutral)
        training_labels = [
            1, 1, 1, 1, 1,      # Positive examples
            -1, -1, -1, -1, -1,  # Negative examples
            0, 0, 0, 0, 0        # Neutral examples
        ]
        
        # Convert seed texts to TF-IDF features
        X = self.vectorizer.fit_transform(training_texts)
        
        # Train model on seed data
        self.model.fit(X, training_labels)
        
        # Store model path for persistence
        self.model_path = model_path

    def prepare_features(self, texts, fit=False):
        """
        Convert text data into TF-IDF feature vectors.
        
        Args:
            texts (list): List of preprocessed text strings
            fit (bool): If True, fit the vectorizer on this data
                       If False, use previously fitted vectorizer
                       
        Returns:
            scipy.sparse.csr_matrix: Sparse matrix of TF-IDF features
            
        The TF-IDF vectorizer converts text into numerical features by:
        1. Counting word frequencies (TF - Term Frequency)
        2. Weighting by inverse document frequency (IDF)
        3. Normalizing the vectors
        """
        if fit:
            return self.vectorizer.fit_transform(texts)  # Learn vocabulary and transform
        return self.vectorizer.transform(texts)  # Use existing vocabulary

    def train(self, texts, labels, tune_hyperparameters=True):
        """
        Train the sentiment analysis model with optional hyperparameter tuning.
        
        Args:
            texts (list): List of preprocessed text strings
            labels (array-like): Target sentiment labels (-1, 0, 1)
            tune_hyperparameters (bool): Whether to perform grid search
            
        The training process includes:
        1. Feature extraction using TF-IDF
        2. Train-validation split (80-20)
        3. Optional hyperparameter tuning via grid search
        4. Model training and evaluation
        """
        # Convert texts to TF-IDF features
        X = self.prepare_features(texts, fit=True)  # Learn vocabulary
        y = labels

        # Create train-validation split for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,     # 20% validation set
            random_state=42    # For reproducibility
        )

        # Check if we should perform hyperparameter tuning
        if tune_hyperparameters and len(set(y_train)) > 1 and all(sum(y_train == label) >= 3 for label in set(y_train)):
            # Define hyperparameter search space
            param_grid = {
                'C': [0.1, 1.0, 10.0],        # Regularization strength
                'penalty': ['l2'],              # L2 regularization
                'solver': ['liblinear']         # Efficient for small datasets
            }

            try:
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    self.model,           # Base model
                    param_grid,           # Parameter space
                    cv=2,                 # 2-fold CV for efficiency
                    n_jobs=-1,            # Use all CPU cores
                    scoring='f1_weighted'  # Use weighted F1 for imbalanced classes
                )
                # Train the model with grid search
                grid_search.fit(X_train, y_train)
                self.model = grid_search.best_estimator_  # Use best model found
            except Exception as e:
                # Handle any exceptions during hyperparameter tuning
                print(f"Warning: Hyperparameter tuning failed: {e}")
                print("Training with default parameters...")
                self.model.fit(X_train, y_train)  # Fallback to default params
        else:
            # Train with default parameters when tuning is not needed/possible
            self.model.fit(X_train, y_train)

        # Evaluate model performance on validation set
        val_predictions = self.model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)  # Simple accuracy
        val_f1 = f1_score(y_val, val_predictions, average='weighted')  # F1 for imbalanced data

        # Report validation metrics
        print(f"Validation Accuracy: {val_accuracy:.2f}")
        print(f"Validation F1 Score: {val_f1:.2f}")

        # Train final model on complete dataset
        self.model.fit(X, y)  # Use all data for final model

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        evaluation = {
            'classification_report': classification_report(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_val, self.model.predict_proba(X_val), multi_class='ovr')
        }

        return evaluation

    def predict(self, texts):
        """
        Predict sentiment labels for new texts.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            numpy.ndarray: Predicted sentiment labels (-1, 0, 1)
        """
        # Convert texts to TF-IDF features
        X = self.prepare_features(texts)
        return self.model.predict(X)

    def predict_proba(self, texts):
        """
        Get probability estimates for each sentiment class.
        
        Args:
            texts (list): List of preprocessed text strings
            
        Returns:
            numpy.ndarray: Probability matrix (n_samples, n_classes)
                         Each row sums to 1, giving P(class) for each input
        """
        X = self.prepare_features(texts)
        return self.model.predict_proba(X)

    def save_model(self, path=None):
        """
        Save the trained model and vectorizer to disk.
        
        This method saves both the trained classifier and the fitted
        TF-IDF vectorizer to ensure consistent preprocessing for
        future predictions.
        
        Args:
            path (str): Path to save the model (optional)
        
        Raises:
            ValueError: If model_path is not specified
        """
        if path is None:
            path = self.model_path
        
        if path is None:
            raise ValueError("No path specified for saving the model")

        # Ensure save directory exists
        os.makedirs(path, exist_ok=True)
        
        model_file = os.path.join(path, 'sentiment_model.joblib')
        vectorizer_file = os.path.join(path, 'vectorizer.joblib')
        
        joblib.dump(self.model, model_file)
        joblib.dump(self.vectorizer, vectorizer_file)

    def load_model(self, path=None):
        """
        Load a previously saved model from disk.
        
        This method restores both the trained classifier and the
        fitted TF-IDF vectorizer to ensure consistent preprocessing.
        
        Args:
            path (str): Path to load the model from (optional)
        
        Raises:
            FileNotFoundError: If no model file exists at model_path
        """
        if path is None:
            path = self.model_path
            
        if path is None:
            raise ValueError("No path specified for loading the model")

        model_file = os.path.join(path, 'sentiment_model.joblib')
        vectorizer_file = os.path.join(path, 'vectorizer.joblib')
        
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(vectorizer_file)
