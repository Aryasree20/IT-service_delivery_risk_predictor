import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from urllib.parse import urlparse
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize
from risk_predictor import logger
from risk_predictor.entity.config_entity import ModelEvaluationConfig
from risk_predictor.utils.common import  save_json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.metrics = None
        self.roc_path = None

    def eval_metrics(self, y_true, y_pred, y_pred_proba):
        """Compute classification metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted"),
            "recall": recall_score(y_true, y_pred, average="weighted"),
            "f1_score": f1_score(y_true, y_pred, average="weighted"),
        }
        return metrics

    def _save_roc_curve(self, name, y_true, y_pred_proba, classes):
        """Save ROC curve plot for multiclass"""
        y_true_bin = label_binarize(y_true, classes=classes)
        if y_true_bin.shape[1] == 1:
            return None

        plt.figure()
        for i in range(len(classes)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"Class {classes[i]} (AUC={roc_auc:.2f})")

        plt.plot([0, 1], [0, 1], "k--")
        plt.title(f"ROC Curve - {name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")

        roc_path = Path(self.config.plots_dir) / f"{name}_roc.png"
        plt.savefig(roc_path)
        plt.close()
        logger.info(f" Saved ROC curve plot: {roc_path}")
        return roc_path

    def eval_and_save(self):
        """Run evaluation, save metrics and ROC plot"""
        # Load test data
        X_test = joblib.load(self.config.X_test_path)
        y_test = joblib.load(self.config.y_test_path)

        # Load trained model
        model = joblib.load(self.config.model_path)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        self.metrics = self.eval_metrics(y_test, y_pred, y_pred_proba)
        save_json(path=Path(self.config.metric_file_name), data=self.metrics)

        # ROC curve
        self.roc_path = self._save_roc_curve("RandomForestClassifier", y_test, y_pred_proba, classes=model.classes_)

        logger.info("Metrics and ROC curve saved")

    def log_into_mlflow(self):
        """Log results into MLflow"""
        if self.metrics is None or self.roc_path is None:
            raise ValueError("Run eval_and_save() before log_into_mlflow()")

        # Load trained model
        model = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log params
            mlflow.log_params(self.config.all_params)

            # Log metrics
            for k, v in self.metrics.items():
                mlflow.log_metric(k, v)

            # Log ROC curve plot
            mlflow.log_artifact(str(self.roc_path), artifact_path="plots")

            # Register the model in MLflow Model Registry
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassifier_MultiClassifier")
            else:
                mlflow.sklearn.log_model(model, "model")

        
        logger.info("Model evaluation (RandomForestClassifier_MultiClassifier) completed and logged to MLflow")