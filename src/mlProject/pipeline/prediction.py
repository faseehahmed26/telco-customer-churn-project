import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        # Load trained model
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

        # Load the preprocessor for feature transformation
        self.preprocessor = joblib.load(Path('artifacts/data_transformation/preprocessor.pkl'))

        # Load the label encoder if it exists (for classification problems)
        label_encoder_path = Path('artifacts/data_transformation/label_encoder.pkl')
        self.label_encoder = joblib.load(label_encoder_path) if label_encoder_path.exists() else None

    def preprocess_input(self, input_data):
        """
        Convert input dictionary to DataFrame and apply preprocessing.
        """
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])

        # # Drop customerID as it's not used for prediction
        # if 'customerID' in input_df.columns:
        #     input_df.drop(columns=['customerID'], inplace=True)

        # Apply the same preprocessing as during training
        transformed_data = self.preprocessor.transform(input_df)

        return transformed_data

    def predict(self, input_data):
        """
        Preprocess input, make predictions, and return human-readable output.
        """
        # Preprocess input data
        processed_data = self.preprocess_input(input_data)

        # Make prediction
        prediction = self.model.predict(processed_data)

        # Convert prediction back to original labels if label encoder exists
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform(prediction.astype(int))

        return prediction
