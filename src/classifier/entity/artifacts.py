from dataclasses import dataclass


@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str


@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str


@dataclass
class ModelTrainingArtifacts:
    trained_model_path: str
    x_test_path: list
    y_test_path: list


@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool


@dataclass
class ModelPushingArtifacts:
    bucket_name: str
