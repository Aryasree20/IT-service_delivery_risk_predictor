from risk_predictor import logger
from risk_predictor.pipeline.stage_01_dataingestion import DataIngestionTrainingPipeline
from risk_predictor.pipeline.stage_02_data_preprocessing import DataPreprocessingPipeline
from risk_predictor.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from risk_predictor.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


STAGE_NAME = "Data Ingestion stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Data preprocessing stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e




STAGE_NAME = "Model Training stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Evaluation stage"

try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx=================x")
except Exception as e:
        logger.exception(e)
        raise e
