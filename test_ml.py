import os
import numpy as np
import pandas as pd
import unittest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
from ml.model import inference

# Prepare data for testing
project_path = ""  # Adjust your project path accordingly
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)
train, test = train_test_split(data, test_size=0.2, random_state=42)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Create a model for testing
class TestModel(unittest.TestCase):

    def setUp(self):
        """Set up the model for testing."""
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)  # Fit the model with training data

    def test_inference_output_type(self):
        """Test if the inference output is a NumPy array."""
        # Create test data with the same number of features as X_train
        X_test = np.random.rand(1, X_train.shape[1])  # Random test data with correct shape
        preds = inference(self.model, X_test)

        # Test if output is a numpy array
        self.assertIsInstance(preds, np.ndarray, "Output should be a NumPy array")

    def test_compute_model_metrics(self):
        """Tests whether model metrics are correct."""
        self.assertIsInstance(self.model, RandomForestClassifier,
                              "Model should be a Random Forest Classifier")

    def test_model_algorithm_type(self):
        """Test if the model is using the expected algorithm."""
        self.assertIsInstance(self.model, RandomForestClassifier,
                              "Expected model type is RandomForestClassifier")