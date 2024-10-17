import pytest
import numpy as np
import unittest
# TODO: implement the first test. Change the function name and input as needed
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .ml.data import  process_data
from .ml.model import inference
from sklearn.ensemble import RandomForestClassifier

#first lets create the data for testing
project_path = ""
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)
train, test = train_test_split(data, test_size=.2, random_state=42) #using defaults
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


class TestModelInference(unittest.TestCase):
    def test_inference_output_type(self):
        # Example model and test data
        model = RandomForestClassifier()
        X_test = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example test data
        preds = inference(model, X_test)

        # Test if output is a numpy array
        self.assertIsInstance(preds, np.ndarray, "Output should be a NumPy array")


# TODO: implement the second test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Tests whether model metrics are correct
    Parameters
    ----------

    Returns
    -------

    """

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    assertIsInstance(model, RandomForestClassifier, "Model should be a Random Forest Classifier")
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_feature_data_types():
    """
    Tests whether feature data types are correct
    Parameters
    ----------

    Returns
    -------

    """

    expected_dtypes = {'age': 'int64', 'education_num': 'int64', 'hours_per_week': 'int64'}
    for feature, dtype in expected_dtypes.items():
        assertEqual(X_train[feature].dtype.name, dtype, f"Feature '{feature}' should be of type '{dtype}'")
    pass
