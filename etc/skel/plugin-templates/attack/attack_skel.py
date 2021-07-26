#!/usr/bin/env python3

from typing import Any

import jespipe.plugin.save as save  # Provides routines for saving data created during the training, attack, and clean stages.
from jespipe.plugin.start import start  # Reads in parameters sent by Jespipe when your plugin is called.
from jespipe.plugin.attack.attack import Attack  # Defines abstract skeleton for attacks.


class Attack(Attack):
    """Class to facilitate attacks aganist a trained model."""
    def __init__(self, model: str, features: Any, parameters: dict) -> None:
        """
        Create an instance of the Attack class.

        ### Parameters:
        :param model: File path to trained model being attacked. Model will need to be loaded back into memory.
        :param features: Features to use for adversarial examples.
        :param parameters: Parameter dictionary for the attack.

        #### Notes:
        - You can add internal methods to help perform your attack.
        - You can modify your attack to not use a model. (File path to trained model is still sent by Jespipe)
        """
        pass

    def attack(self) -> Any:
        """
        Abstract method for generating adversarial examples.

        ### Returns:
        :return: Generated adversarial example.
        """
        pass


if __name__ == "__main__":
    # Read in stage and parameter dictionary sent by Jespipe
    stage, parameters = start()

    # Execute code block based on the passed stage from Jespipe
    if stage == "attack":
        """
        This block will be executed during the adversarial example generation sub-stage during the attack stage of Jespipe.

        ### Parameter dictionary content:
        - `name`: Name of the attack being used.
        - `model_path`: File path to model being attacked.
        - `model_test_features`: Test features to use for adversarial example generation.
        - `attack_params`: Parameters to use for the attack.
        - `save_path`: Where to save the adversarial example.
        """
        pass

    else:
        """You will typically never encounter this case, but it is here just in case."""
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
