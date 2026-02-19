import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from risk_predictor import logger
from risk_predictor.entity.config_entity import DataPreprocessingConfig

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.le = LabelEncoder()

    def load_data(self, data_path: str):
        """Load raw CSV data"""
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        logger.info(f"Shape: {df.shape}")
        logger.info(f"info: \n{str(df.info())}")
        logger.info(f"describe: \n{df.describe().to_dict()}")
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        logger.info(f"Target distribution:\n{df['predicted_risk'].value_counts().to_dict()}")
        return df

    def validate_duration(self, df: pd.DataFrame):
        """Validate duration calculation"""
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])
        df['actual_duration'] = (df['end_date'] - df['start_date']).dt.days
        discrepancy = (df['actual_duration'] - df['actual_duration_days']).abs().sum()
        if discrepancy != 0:
            logger.warning(f"Discrepancy in actual_duration_days: {discrepancy}")



        """cross validate expected delivery_delay_days"""  
        df["calculated_delay"] = df["actual_duration_days"] - df["planned_duration_days"]
        if (df["delivery_delay_days"] != df["calculated_delay"]).all():
            logger.warning(f"mismatch found")
      
        return df
    

    def encode_target(self, df: pd.DataFrame):
        """Encode target labels"""
        df['predicted_risk'] = self.le.fit_transform(df['predicted_risk'])
        return df

    def drop_columns(self, df: pd.DataFrame):
        """Drop unnecessary columns"""
        df.drop(['project_id', 'start_date', 'end_date', 'delivery_delay_days','actual_duration','calculated_delay'], axis=1, inplace=True)
        return df

    def split_data(self, df: pd.DataFrame):
        """Split into train and test"""
        X = df.drop('predicted_risk', axis=1)
        y = df['predicted_risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE oversampling"""
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(f"Class distribution after SMOTE: {pd.Series(y_res).value_counts().to_dict()}")
        return X_res, y_res

    def scale_data(self, X_train, X_test):
        """Scale features"""
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test
    
    
    def save_dataset(self, X_train, X_test, y_train, y_test):
        """
        Saves the processed train and test datasets as joblib files
        """
        logger.info("Saving train and test datasets as joblib files")
        pickle_save = self.config.pickle_save
        os.makedirs(pickle_save, exist_ok=True)

        # Paths
        x_train_path = os.path.join(pickle_save, 'X_train.joblib')
        x_test_path = os.path.join(pickle_save, 'X_test.joblib')
        y_train_path = os.path.join(pickle_save, 'y_train.joblib')
        y_test_path = os.path.join(pickle_save, 'y_test.joblib')

        # Save
        joblib.dump(X_train, x_train_path)
        joblib.dump(X_test, x_test_path)
        joblib.dump(y_train, y_train_path)
        joblib.dump(y_test, y_test_path)

        logger.info(f" Datasets saved to {pickle_save}")
        return X_train, X_test, y_train, y_test
