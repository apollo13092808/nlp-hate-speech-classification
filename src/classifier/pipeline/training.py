import sys
from classifier.components.data_ingestion import DataIngestion
from classifier.components.data_transformation import DataTransformation
from classifier.components.model_evaluation import ModelEvaluation
from classifier.components.model_pushing import ModelPushing
from classifier.components.model_training import ModelTraining
from classifier.entity.artifacts import DataIngestionArtifacts, DataTransformationArtifacts, ModelEvaluationArtifacts, ModelPushingArtifacts, ModelTrainingArtifacts
from classifier.entity.configs import DataIngestionConfig, DataTransformationConfig, ModelEvaluationConfig, ModelPushingConfig, ModelTrainingConfig
from classifier.exception import CustomException
from classifier.logger import logging


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainingConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPushingConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info(
            "Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from GCLoud Storage bucket")
            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config)

            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the train and valid from GCLoud Storage")
            logging.info(
                "Exited the start_data_ingestion method of TrainingPipeline class"
            )
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifacts=DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info(
            "Entered the start_data_transformation method of TrainingPipeline class"
        )
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts=data_ingestion_artifacts,
                data_transformation_config=self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info(
                "Exited the start_data_transformation method of TrainingPipeline class"
            )
            return data_transformation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_training(self, data_transformation_artifacts: DataTransformationArtifacts) -> ModelTrainingArtifacts:
        logging.info(
            "Entered the start_model_training method of TrainPipeline class"
        )
        try:
            model_training = ModelTraining(
                data_transformation_artifacts=data_transformation_artifacts,
                model_trainer_config=self.model_trainer_config
            )
            model_training_artifacts = model_training.initiate_model_trainer()
            logging.info(
                "Exited the start_model_training method of TrainingPipeline class"
            )
            return model_training_artifacts

        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self, model_trainer_artifacts: ModelTrainingArtifacts, data_transformation_artifacts: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        logging.info(
            "Entered the start_model_evaluation method of TrainPipeline class")
        try:
            model_evaluation = ModelEvaluation(data_transformation_artifacts=data_transformation_artifacts,
                                               model_evaluation_config=self.model_evaluation_config,
                                               model_trainer_artifacts=model_trainer_artifacts)

            model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
            logging.info(
                "Exited the start_model_evaluation method of TrainingPipeline class"
            )
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_model_pushing(self,) -> ModelPushingArtifacts:
        logging.info(
            "Entered the start_model_pushing method of TrainPipeline class")
        try:
            model_pushing = ModelPushing(
                model_pushing_config=self.model_pushing_config,
            )
            model_pushing_artifact = model_pushing.initiate_model_pushing()
            logging.info("Initiated the model pusher")
            logging.info(
                "Exited the start_model_pushing method of TrainingPipeline class")
            return model_pushing_artifact

        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts=data_ingestion_artifacts
            )

            model_training_artifacts = self.start_model_training(
                data_transformation_artifacts=data_transformation_artifacts
            )

            model_evaluation_artifacts = self.start_model_evaluation(
                model_training_artifacts=model_training_artifacts,
                data_transformation_artifacts=data_transformation_artifacts
            )

            if not model_evaluation_artifacts.is_model_accepted:
                raise Exception(
                    "Trained model is not better than the best model")

            model_pusher_artifacts = self.start_model_pusher()

            logging.info(
                "Exited the run_pipeline method of TrainingPipeline class")

        except Exception as e:
            raise CustomException(e, sys) from e
