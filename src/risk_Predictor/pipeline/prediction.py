import os
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PredictionPipeline:
    def __init__(self):
        try:
            model_path = os.path.join("artifacts", "model_trainer", "best_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            self.model = joblib.load(model_path)
            logger.info("✅ Model loaded successfully from %s", model_path)
        except Exception as e:
            logger.error("❌ Error loading model: %s", str(e))
            raise e

    def predict(self, data: np.ndarray) -> str:
        try:
            prediction = self.model.predict(data)
            mapping = {0: "High", 1: "Low", 2: "Medium"}
            return mapping.get(int(prediction[0]), "Unknown")
        except Exception as e:
            logger.error("❌ Error during prediction: %s", str(e))
            raise e