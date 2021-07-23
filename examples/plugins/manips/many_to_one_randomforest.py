import uuid
from typing import Tuple

import jespipe.plugin.save as save
import joblib
import numpy as np
import pandas as pd
from jespipe.plugin.manip.manip import Manipulation
from jespipe.plugin.start import start
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


class RandomForestManip(Manipulation):
    def __init__(self, parameters: dict) -> None:
        """
        RandomForest Manipulation class to facilitate the RandomForst 
        feature selection technique on many-to-one datasets with no header row.
        Target feature holds index position -1 in the passed dataset.

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
          - _preproc_randomforest: Internal vanilla preprocessing method for passed dataset.
        """
        self.dataset = pd.read_csv(parameters["dataset"], header=None)
        self.manip_tag = parameters["manip_tag"]
        self.save_path = parameters["save_path"]
        self.tmp_path = parameters["tmp_path"]

    def manipulate(self) -> None:
        """
        Perform RandomForest feature selection technique on passed dataset. 
        Path to pickled manipulation is printed to stdout in order to be captured by
        subprocess.getoutput().
        """
        features, labels = self._preproc_randomforest()
        recomb = pd.concat([pd.DataFrame(features), pd.DataFrame(labels)], axis=1)

        # Save copy of current DataFrame for later analysis
        save.dataframe(self.save_path, self.manip_tag, recomb)

        # Save pickle of manipulated DataFrame
        pickle_path = self.tmp_path + "/" + str(uuid.uuid4()) + ".pkl"
        joblib.dump(recomb, pickle_path)

        # Print out pickle path to be captured by subprocess.getoutput()
        print(pickle_path)

    def _preproc_randomforest(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal RandomForest feature selection preprocessing method for passed dataset.
        Splits DataFrame into features and labels.

        ### Returns:
        :return: Tuple with dataset features at index 0 and labels at index 1. 
        """
        # Features for training
        features = np.array(self.dataset)[:, :-1]

        # Labels
        labels = np.array(self.dataset)[:, -1]

        sel = SelectFromModel(RandomForestRegressor(n_estimators=100))
        sel.fit(features, labels.astype('int'))

        features_to_select = sel.get_support(indices=True)
        # print('Features to select: ', features_to_select)

        new_features = features[:, features_to_select]

        return new_features, labels


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "train":
        randomforest = RandomForestManip(parameters)
        randomforest.manipulate()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
