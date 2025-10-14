from risk_predictor.config.configuration import ConfigurationManager 
from risk_predictor.components.data_preprocessing import DataPreprocessing
from risk_predictor import logger

STAGE_NAME = "Data Preprocessing Stage"

class DataPreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocessing_config = config.get_data_preprocessing_config()

        # Initialize class
        data_preprocessing = DataPreprocessing(config=data_preprocessing_config)

        df = data_preprocessing.load_data(data_preprocessing_config.raw_data_path)
        df = data_preprocessing.validate_duration(df)
        df = data_preprocessing.encode_target(df)
        df = data_preprocessing.drop_columns(df)
        X_train, X_test, y_train, y_test = data_preprocessing.split_data(df)
        X_train, y_train = data_preprocessing.apply_smote(X_train, y_train)
        X_train, X_test = data_preprocessing.scale_data(X_train, X_test)
        data_preprocessing.save_dataset(X_train, X_test, y_train, y_test)

        logger.info("Data Preprocessing pipeline executed successfully")


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
