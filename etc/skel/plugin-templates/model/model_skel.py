#!/usr/bin/env python3

from typing import Any, Tuple

import jespipe.plugin.save as save  # Provides routines for saving data created during the training, attack, and clean stages.
from jespipe.plugin.start import start  # Reads in parameters sent by Jespipe when your plugin is called.
from jespipe.plugin.train.build import Build  # Defines abstract skeleton for building models in Jespipe.
from jespipe.plugin.train.evaluate import Evaluate  # Defines abstract skeleton for evaluating models in Jespipe.
from jespipe.plugin.train.fit import Fit  # Defines abstract skeleton for fitting models in Jespipe.
from jespipe.plugin.train.predict import Predict  # Defines abstract skeleton for making predictions on data with Jespipe.


class BuildModel(Build):
    """Class to build model before fitting to training data."""
    def __init__(self, parameters: dict) -> None:
        """
        Create an instance of the BuildModel class.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe when plugin is called.

        #### Notes:
        - You do not need to use all the data sent by Jespipe.
        You only need to use the data your model needs.
        """
        pass

    def build_model(self) -> Tuple[Any, Tuple[Any, Any, Any, Any]]:
        """
        Abstract method for building models.
        
        ### Returns:
        :return: Method should return a tuple with two stored values.
        - index 0: The model you are going to train.
        - index 1: Tuple with the following values:
          - 0: Training Features.
          - 1: Training Labels.
          - 2: Test Features.
          - 3: Test Labels.

        #### Notes:
        - This method can be modified to return validation data as well.
        """
        pass


class FitModel(Fit):
    """Class to fit model to training data before evaluation and/or predictions."""
    def __init__(self, model: Any, feature_train: Any, label_train: Any, parameters: dict) -> None:
        """
        Create an instance of the FitModel class.

        ### Parameters:
        :param model: Model to train.
        :param feature_train: Features to use for model training.
        :param label_train: Labels to use for model training.
        :param parameters: Parameter dictionary sent by Jespipe when plugin is called.

        #### Notes:
        - You do not need to use all the data sent by Jespipe.
        You only need to use the data your model needs for training.
        """
        pass

    def model_fit(self) -> None:
        """
        Abstract method for fitting models to training data.

        ### Returns:
        - There is no return, but this will fit your model to the training data.
        You can call FitModel.model to get a reference to your trained model.
        """
        pass


class PredictModel(Predict):
    """Class to make predictions on data."""
    def __init__(self, model: Any, predictee: Any) -> None:
        """
        Create instance of the PredictModel class.

        ### Parameters:
        :param model: Trained model to make prediction with.
        :param predictee: Data to make prediction on.

        #### Notes:
        - Your model should be fitted to its training data before
        making predictions.
        """
        pass

    def model_predict(self) -> Any:
        """
        Abstract method for making predictions on data with a trained model.

        ### Returns:
        :return: Model prediction.
        """
        pass


class EvaluateModel(Evaluate):
    """Class to evaluate model's performance on testing data and/or adversarial examples."""
    def __init__(self, feature_test: Any, label_test: Any, model_to_eval: Any) -> None:
        """
        Create instance of the EvaluateModel class.

        ### Parameters:
        :param feature_test: Test features to evaluate model on.
        :param label_test: Test labels to evaluate model on.
        :param model_to_eval: Trained model to evaluate.

        #### Notes:
        - You can add internal methods to perform your desired evaluations.
        - You can modify __init__ if you are only evaluating certain aspects
        of the model.
        """
        pass

    def model_evaluate(self) -> Any:
        """
        Abstract method for evaluating trained models.

        ### Returns:
        :return: Data produced by model evaluation.
        """
        pass


if __name__ == "__main__":
    # Read in stage and parameter dictionary sent by Jespipe
    stage, parameters = start()

    # Execute code block based on the passed stage from Jespipe
    if stage == "train":
        """
        This block will be executed during the training stage of Jespipe.

        ### Parameter dictionary content:
        - `dataset_name`: Name of the data set being used to train the model.
        - `original_dataset`: File to the original, unmanipulated dataset.
        - `model_name`: Name of the model.
        - `dataframe`: The dataframe containing the data the model is being trained on.
        - `model_params`: Hyperparameters to use for your model.
        - `manip_params`: Parameters used for your data manipulation.
        - `save_path`: The file path to use for saving data created by your model.
        - `log_path`: The file path to use for saving your evaluation data. (Important to your adversarial analysis).
        - `manip_info`: Tuple containing info on your manipulation (name, tag).
        """
        pass

    elif stage == "attack":
        """
        This block will be executed during the model evaluation sub-stage during the attack stage of Jespipe.

        ### Parameter dictionary content:
        - `adver_features`: List containing file paths to adversarial examples for the model.
        - `attack_name`: Name of the attack that the model is being evaluated on.
        - `model_labels`: Test labels to use for evaluation of model's adversarial robustness.
        - `log_path`: The file path to use for saving your evaluation data. (Important to your adversarial analysis).
        - `model_path`: Path to your saved model file in HDF5 format. (Loaded back into memory during the evaluation).
        """
        pass

    else:
        """You will typically never encounter this case, but it is here just in case."""
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
