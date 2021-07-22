#!/usr/bin/env python3

import jespipe.plugin.save as save # Provides routines for saving data created during the training, attack, and clean stages.
from jespipe.plugin.manip.manip import Manipulation  # Defines abstract skeleton for writing your own data manipulations
from jespipe.plugin.start import start # Reads in parameters sent by Jespipe when your plugin is called.

class Manip(Manipulation):
    """Class to manipulate data before training a model."""
    def __init__(self, parameters: dict) -> None:
        """
        Create instance of the Manip class.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe when plugin is called.

        #### Notes:
        - You can add internal methods to help perform your desired data manipulations.
        """
        pass

    def manipulate(self) -> None:
        """
        Abstract method for manipulating data before model training.

        ### Returns:
        - None; but you should print out a file path to a pickle that can
        be read back in by Jespipe with joblib. Stdout is captured by Jespipe.
        """
        print(".tmp/path/to/generated/pickle")
        pass


if __name__ == "__main__":
    # Read in stage and parameter dictionary sent by Jespipe
    stage, parameters = start()

    # Execute code block based on the passed stage from Jespipe
    if stage == "train":
        """
        This block will be executed during the training stage of Jespipe.

        ### Parameter dictionary content:
        - `dataset`: File to data set to manipulate.
        - `manip_tag`: Unique identifier for the manipulation.
        - `manip_params`: Parameters for the manipulation.
        - `save_path`: Where to save a copy of the manipulated.
        - `tmp_path`: Temporary directory used by Jespipe to store pickle files.
        """
        pass

    else:
        """You will typically never encounter this case, but it is here just in case."""
        raise ValueError("Received invalid stage {}. Please only pass valid stages from the pipeline.".format(stage))
