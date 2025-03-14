import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from mlProject import logger
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred, task_type):
        """Evaluate metrics based on problem type"""
        if task_type == "Classification":
            # Ensure predictions are in integer format for classification
            pred = np.round(pred).astype(int)
            actual = actual.astype(int)

            accuracy = accuracy_score(actual, pred)
            precision = precision_score(actual, pred, average='weighted', zero_division=1)
            recall = recall_score(actual, pred, average='weighted', zero_division=1)
            f1 = f1_score(actual, pred, average='weighted', zero_division=1)

            return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

        else:  # Regression case
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            r2 = r2_score(actual, pred)

            return {"rmse": rmse, "mae": mae, "r2": r2}

    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]].values.ravel()  # Flatten target

        # **Auto-detect Task Type**: If target has ≤ 10 unique values, classify as "Classification"
        task_type = "Classification" 
        # if test_y.nunique() <= 10 else "Regression"

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)

            # Compute correct metrics based on task type
            metrics = self.eval_metrics(test_y, predicted_qualities, task_type)

            # Save metrics to JSON
            save_json(path=Path(self.config.metric_file_name), data=metrics)

            # Log all calculated metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")