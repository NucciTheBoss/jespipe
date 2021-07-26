import uuid
from typing import Tuple

import jespipe.plugin.save as save
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from jespipe.plugin.manip.manip import Manipulation
from jespipe.plugin.start import start
from sklearn.model_selection import train_test_split


class XGBManip(Manipulation):
    def __init__(self, parameters: dict) -> None:
        """
        XGBoost Manipulation class to facilitate the XGBoost feature 
        selection technique on many-to-one datasets with no header row.
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
          - manipulate (abstract): Perform XGBoost feature selection on passed dataset.
        - private
          - _preproc_xgb: Internal XGBoost feature selection preprocessing method for passed dataset.
        """
        self.dataset = pd.read_csv(parameters["dataset"], header=None)
        self.manip_tag = parameters["manip_tag"]
        self.manip_params = parameters["manip_params"]
        self.save_path = parameters["save_path"]
        self.tmp_path = parameters["tmp_path"]

    def manipulate(self) -> None:
        """
        Perform XGBoost feature selection technique on passed dataset. 
        Path to pickled manipulation is printed to stdout in order to be captured by
        subprocess.getoutput().
        """
        features, labels = self._preproc_xgb()
        recomb = pd.concat([pd.DataFrame(features), pd.DataFrame(labels)], axis=1)

        # Save copy of current DataFrame for later analysis
        save.dataframe(self.save_path, self.manip_tag, recomb)

        # Save pickle of manipulated DataFrame
        pickle_path = self.tmp_path + "/" + str(uuid.uuid4()) + ".pkl"
        joblib.dump(recomb, pickle_path)

        # Print out pickle path to be captured by subprocess.getoutput()
        print(pickle_path)

    def _preproc_xgb(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal XGBoost feature selection preprocessing method for passed dataset.
        Splits DataFrame into features and labels.

        ### Returns:
        :return: Tuple with dataset features at index 0 and labels at index 1. 
        """
        # Features for training
        features = np.array(self.dataset)[:, :-1]

        # Labels
        labels = np.array(self.dataset)[:, -1]

        feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels)

        # Training with best gamma
        regressor = xgb.XGBRegressor(
            n_estimators=100,
            gamma=1.5,
            max_depth=self.manip_params["n_features"]
        )

        regressor.fit(feature_train, labels_train)

        feature_importance = regressor.feature_importances_

        features_to_select = feature_importance.argsort()[-self.manip_params["n_features"]:][::-1]

        new_features = features[:, features_to_select]

        return new_features, labels


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "train":
        xgbmanip = XGBManip(parameters)
        xgbmanip.manipulate()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
