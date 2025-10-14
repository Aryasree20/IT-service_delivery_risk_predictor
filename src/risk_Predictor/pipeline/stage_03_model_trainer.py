from risk_predictor.config.configuration import ConfigurationManager
from risk_predictor.components.model_trainer import ModelTrainer
from risk_predictor import logger
import os, json, joblib
import numpy as np
import pandas as pd

STAGE_NAME = "Data model training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):   # ✅ Correctly indented inside the class now
        logger.info("Starting Model Training stage")

        # Load configuration
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Initialize ModelTrainer
        model_trainer = ModelTrainer(config=model_trainer_config)

        # Load preprocessed datasets
        X_train = joblib.load("artifacts/data_preprocessing/pickle/X_train.joblib")
        X_test = joblib.load("artifacts/data_preprocessing/pickle/X_test.joblib")
        y_train = joblib.load("artifacts/data_preprocessing/pickle/y_train.joblib")
        y_test = joblib.load("artifacts/data_preprocessing/pickle/y_test.joblib")

        # Execute model training + evaluation
        best_model, best_name = model_trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test
        )

        if best_model:
            logger.info(f"Model training pipeline completed successfully. Best model: {best_name}")
        else:
            logger.warning("Model training finished but no suitable model found.")

# ✅ Keep this part outside the class
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()  # ✅ This will now work fine
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

















































'''from risk_predictor.config.configuration import ConfigurationManager
from risk_predictor.components.model_trainer import ModelTrainer
from risk_predictor import logger
import os, json, joblib
import numpy as np
import pandas as pd



STAGE_NAME = "Data model training Stage"

class ModelTrainingPipeline:
    def __init__(self):
     pass
def main(self):
   
        logger.info("Starting Model Training stage")

        # Load configuration
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()

        # Initialize ModelTrainer
        model_trainer = ModelTrainer(config=model_trainer_config)

        # Load preprocessed datasets
        X_train = joblib.load("artifacts/data_preprocessing/pickle/X_train.joblib")
        X_test = joblib.load("artifacts/data_preprocessing/pickle/X_test.joblib")
        y_train = joblib.load("artifacts/data_preprocessing/pickle/y_train.joblib")
        y_test = joblib.load("artifacts/data_preprocessing/pickle/y_test.joblib")

        # Execute model training + evaluation (this internally calls _save_report_json, 
        # _save_report_csv, _save_confusion_matrix, and _save_roc_curve for each model)
        best_model, best_name = model_trainer.train_and_evaluate(
            X_train, y_train, X_test, y_test
        )

        if best_model:
            logger.info(f"Model training pipeline completed successfully. Best model: {best_name}")
        else:
            logger.warning("Model training finished but no suitable model found.")
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e '''