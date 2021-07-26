import uuid
from typing import Tuple

import jespipe.plugin.save as save
import joblib
import numpy as np
import pandas as pd
from jespipe.plugin.manip.manip import Manipulation
from jespipe.plugin.start import start


class VanillaManip(Manipulation):
    def __init__(self, parameters: dict) -> None:
        """
        Vanilla Manipulation class to facilitate vanilla data manipulation
        on many-to-one datasets with no header row. Target feature holds 
        index position -1 in the passed dataset.

        ### Parameters:
        - :param parameters: Parameter dictionary sent by Jespipe. Contains 
        the following key -> value pairings: 
          - dataset: "/path/to/dataset.csv"
          - manip_tag: "manip_tag_name"
          - manip_params: {"parameter": value}
          - save_path: "/path/to/save/directory"
          - tmp_path: "/path/to/data/.tmp"

        ### Methods:
        - public 
          - manipulate (abstract): Perform vanilla manipulation on passed dataset.
        - private 
          - _preproc_vanilla: Internal vanilla preprocessing method for passed dataset.
        """
        self.dataset = pd.read_csv(parameters["dataset"], header=None)
        self.manip_tag = parameters["manip_tag"]
        self.save_path = parameters["save_path"]
        self.tmp_path = parameters["tmp_path"]

    def manipulate(self) -> None:
        """
        Perform vanilla manipulation on passed dataset. Path to pickled
        manipulation is printed to stdout in order to be captured by
        subprocess.getoutput().
        """
        features, labels = self._preproc_vanilla()
        recomb = pd.concat([pd.DataFrame(features), pd.DataFrame(labels)], axis=1)

        # Save copy of current DataFrame for later analysis
        save.dataframe(self.save_path, self.manip_tag, recomb)

        # Save pickle of manipulated DataFrame
        pickle_path = self.tmp_path + "/" + str(uuid.uuid4()) + ".pkl"
        joblib.dump(recomb, pickle_path)

        # Print out pickle path to be captured by subprocess.getoutput()
        print(pickle_path)

    def _preproc_vanilla(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal vanilla preprocessing method for passed dataset.
        Splits DataFrame into features and labels.

        ### Returns:
        :return: Tuple with dataset features at index 0 and labels at index 1.
        """
        # Features for model training
        features = np.array(self.dataset)[:, :-1]

        # Labels
        labels = np.array(self.dataset)[:, -1]

        return features, labels


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "train":
        vanilla = VanillaManip(parameters)
        vanilla.manipulate()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
