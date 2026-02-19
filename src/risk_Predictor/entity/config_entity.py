from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path


@dataclass(frozen=True)
class EDAConfig:
    root_dir: Path
    data_file: Path
    reports_dir: Path
    plots_dir: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: str
    raw_data_path: str
    pickle_save: str  


@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    processed_data: Path
    reports_dir: Path
    plots_dir: Path
    best_model_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    plots_dir: Path
    metric_file_name: Path
    model_path: Path
    X_test_path: Path
    y_test_path: Path
    mlflow_uri: str
    all_params: dict
     