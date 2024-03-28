import sys
from classifier.configurations.syncer import GoogleCloudSync
from classifier.entity.artifacts import ModelPushingArtifacts
from classifier.entity.configs import ModelPushingConfig
from classifier.exception import CustomException
from classifier.logger import logging


class ModelPushing:
    def __init__(self, model_pushing_config: ModelPushingConfig):
        """
        :param model_pushing_config: Configuration for model pushing
        """
        self.model_pushing_config = model_pushing_config
        self.gcloud = GoogleCloudSync()

    def initiate_model_pushing(self) -> ModelPushingArtifacts:
        """
            Method Name :   initiate_model_pushing
            Description :   This method initiates model pushing.

            Output      :    Model pushing artifact
        """
        logging.info(
            "Entered initiate_model_pushing method of ModelPushing class"
        )
        try:
            # Uploading the model to gcloud storage

            self.gcloud.sync_folder_to_cloud(self.model_pushing_config.BUCKET_NAME,
                                             self.model_pushing_config.TRAINED_MODEL_PATH,
                                             self.model_pushing_config.MODEL_NAME)

            logging.info("Uploaded best model to gcloud storage")

            # Saving the model pusher artifacts
            model_pushing_artifact = ModelPushingArtifacts(
                bucket_name=self.model_pushing_config.BUCKET_NAME
            )
            logging.info(
                "Exited the initiate_model_pusher method of ModelTrainer class"
            )
            return model_pushing_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
