import os
import pickle
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from classifier.constants import LABEL, TWEET
from classifier.entity.artifacts import DataTransformationArtifacts, ModelTrainingArtifacts
from classifier.entity.configs import ModelTrainingConfig
from classifier.exception import CustomException

from keras.utils import pad_sequences
from classifier.logger import logging
from keras_preprocessing.text import Tokenizer


class ModelTraining:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                 model_training_config: ModelTrainingConfig):

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_training_config = model_training_config

    def spliting_data(self, csv_path):
        try:
            logging.info("Entered the spliting_data function")
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into X and y")
            X = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train_test_split on the data")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            print(len(X_train), len(y_train))
            print(len(X_test), len(y_test))
            print(type(X_train), type(y_train))
            logging.info("Exited the spliting the data function")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, X_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(
                num_words=self.model_training_config.MAX_WORDS
            )
            tokenizer.fit_on_texts(X_train)
            sequences = tokenizer.texts_to_sequences(X_train)
            logging.info(f"converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(
                sequences, maxlen=self.model_training_config.MAX_LEN)
            logging.info(f" The sequence matrix is: {sequences_matrix}")
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_training(self,) -> ModelTrainingArtifacts:
        logging.info(
            "Entered initiate_model_training method of ModelTraining class")

        """
        Method Name :   initiate_model_training
        Description :   This function initiates a model training steps
        
        Output      :   Returns model training artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            logging.info("Entered the initiate_model_training function ")
            X_train, X_test, y_train, y_test = self.spliting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()

            model = model_architecture.get_model()

            logging.info(f"Xtrain size is : {X_train.shape}")

            logging.info(f"Xtest size is : {X_test.shape}")

            sequences_matrix, tokenizer = self.tokenizing(X_train)

            logging.info("Entered into model training")
            model.fit(sequences_matrix, y_train,
                      batch_size=self.model_training_config.BATCH_SIZE,
                      epochs=self.model_training_config.EPOCH,
                      validation_split=self.model_training_config.VALIDATION_SPLIT,
                      )
            logging.info("Model training finished")

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer,
                            handle,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(
                self.model_training_config.TRAINED_MODEL_DIR, exist_ok=True
            )

            logging.info("saving the model")
            model.save(self.model_training_config.TRAINED_MODEL_PATH)
            X_test.to_csv(self.model_training_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_training_config.Y_TEST_DATA_PATH)

            X_train.to_csv(self.model_training_config.X_TRAIN_DATA_PATH)

            model_training_artifacts = ModelTrainingArtifacts(
                trained_model_path=self.model_training_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_training_config.X_TEST_DATA_PATH,
                y_test_path=self.model_training_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainingArtifacts")
            return model_training_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
