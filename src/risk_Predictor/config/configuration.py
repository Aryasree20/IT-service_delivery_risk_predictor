import os
from risk_predictor.constants import *
from risk_predictor.utils.common import read_yaml, create_directories
from risk_predictor.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelTrainerConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])

    # ---------------- Stage 01: Data Ingestion ----------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )
        return data_ingestion_config

    # ---------------- Stage 02: Data Preprocessing ----------------
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            raw_data_path=config.raw_data_path,
            pickle_save=config.pickle_save
        )
        return data_preprocessing_config

    # ---------------- Stage 03: Model Trainer ----------------
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir, config.reports_dir, config.plots_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            processed_data=config.processed_data,
            reports_dir=config.reports_dir,
            plots_dir=config.plots_dir,
            best_model_path=config.best_model_path
        )
        return model_trainer_config



























































"""import os
from risk_predictor.constants import *
from risk_predictor.utils.common import read_yaml,create_directories
from risk_predictor.entity.config_entity import DataIngestionConfig , ModelTrainerConfig
from risk_predictor.entity.config_entity import DataPreprocessingConfig

from risk_predictor.entity.config_entity import DataIngestionConfig
from risk_predictor.constants import *
from risk_predictor.utils.common import read_yaml, create_directories
import os

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
        )

        return data_ingestion_config

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        return DataPreprocessingConfig(
            root_dir=config.root_dir,
            raw_data_path=config.raw_data_path,
            pickle_save=config.pickle_save
            
        )    
    
class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH):
        self.config = read_yaml(config_filepath)
        create_directories([self.config.artifacts_root])

    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        create_directories([config.root_dir, config.reports_dir, config.plots_dir])
        return ModelTrainerConfig(
            root_dir=config.root_dir,
            processed_data=config.processed_data,
            reports_dir=config.reports_dir,
            plots_dir=config.plots_dir,
            best_model_path=config.best_model_path
        )
    






    """