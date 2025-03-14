import os
from mlProject import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from mlProject.entity.config_entity import DataTransformationConfig
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    
class DataTransformation:
    def __init__(self, config, target_column):
        self.config = config
        self.target_column = target_column

    def train_test_splitting(self):
        # Load the dataset
        data = pd.read_csv(self.config.data_path)

        # Identify numerical and categorical columns
        num_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = data.select_dtypes(include=['object']).columns.tolist()

        # Remove target column from features
        if self.target_column in num_features:
            num_features.remove(self.target_column)
            target_is_numeric = True
        elif self.target_column in cat_features:
            cat_features.remove(self.target_column)
            target_is_numeric = False
        else:
            raise ValueError(f"Target column {self.target_column} not found in dataset.")

        # Define transformation pipelines
        num_pipeline = Pipeline([
            ('scaler', StandardScaler())  # Standardize numerical features
        ])

        cat_pipeline = Pipeline([
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encode categorical features
        ])

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features)
        ])

        # Apply transformations to features
        transformed_features = preprocessor.fit_transform(data.drop(columns=[self.target_column]))

        # Extract correct feature names
        num_feature_names = num_features
        cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_features).tolist()
        feature_names = num_feature_names + cat_feature_names

        # Transform target variable
        if target_is_numeric:
            transformed_target = data[self.target_column].values  # Keep as-is for regression
        else:
            label_encoder = LabelEncoder()
            transformed_target = label_encoder.fit_transform(data[self.target_column])  # Convert to numeric labels

            # Save LabelEncoder to use in inference
            label_encoder_path = os.path.join(self.config.root_dir, "label_encoder.pkl")
            joblib.dump(label_encoder, label_encoder_path)
            logger.info(f"Saved LabelEncoder at {label_encoder_path}")

        # Convert to DataFrame
        transformed_df = pd.DataFrame(transformed_features, columns=feature_names)
        transformed_df[self.target_column] = transformed_target  # Append transformed target column

        # Split data
        train, test = train_test_split(transformed_df, test_size=0.25, random_state=42)

        # Save transformed datasets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info(f"Data transformed and split into training ({train.shape}) and test ({test.shape}) sets.")

        # return train, test
        