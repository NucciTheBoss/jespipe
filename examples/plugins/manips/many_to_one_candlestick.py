import uuid
from typing import Tuple

import jespipe.plugin.save as save
import joblib
import numpy as np
import pandas as pd
from jespipe.plugin.manip.manip import Manipulation
from jespipe.plugin.start import start


class CandlestickManip(Manipulation):
    def __init__(self, parameters: dict) -> None:
        """
        Candlestick Manipulation class to facilitate the Candlestick trend 
        extraction technique on many-to-one datasets with no header row.
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
          - manipulate (abstract): Perform Candlestick trend extraction on passed dataset.
        - private
          - _preproc_candlestick: Internal Candlestick trend extraction preprocessing method for passed dataset.
        """
        self.dataset = pd.read_csv(parameters["dataset"], header=None)
        self.manip_tag = parameters["manip_tag"]
        self.manip_params = parameters["manip_params"]
        self.save_path = parameters["save_path"]
        self.tmp_path = parameters["tmp_path"]

    def manipulate(self) -> None:
        """
        Perform Candlestick trend extraction technique on passed dataset. 
        Path to pickled manipulation is printed to stdout in order to be captured by
        subprocess.getoutput().
        """
        features, labels = self._preproc_candlestick()
        recomb = pd.concat([pd.DataFrame(features), pd.DataFrame(labels)], axis=1)

        # Save copy of current DataFrame for later analysis
        save.dataframe(self.save_path, self.manip_tag, recomb)

        # Save pickle of manipulated DataFrame
        pickle_path = self.tmp_path + "/" + str(uuid.uuid4()) + ".pkl"
        joblib.dump(recomb, pickle_path)

        # Print out pickle path to be captured by subprocess.getoutput()
        print(pickle_path)

    def _preproc_candlestick(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal Candlestick trend extraction preprocessing method for passed dataset.
        Splits DataFrame into features and labels.

        ### Returns:
        :return: Tuple with dataset features at index 0 and labels at index 1.
        """
        # Features for model training
        features = np.array(self.dataset)[:, :-1]

        # Labels
        labels = np.array(self.dataset)[:, -1]

        new_features = np.zeros((int(features.shape[0] / self.manip_params["time_interval"]), int(features.shape[1] * 4)))
        new_labels = np.zeros((int(labels.shape[0] / self.manip_params["time_interval"]),))
        for feature_ind in range(features.shape[1]):
            new_feature_ind = feature_ind * 4
            new_row_ind = 0
            for row_ind in range(0, features.shape[0] - self.manip_params["time_interval"], self.manip_params["time_interval"]):
                # Find the 'open' value
                open_value = features[row_ind, feature_ind]
                new_features[new_row_ind, new_feature_ind] = open_value

                # Find the 'close' value
                end_ind = int(row_ind + (self.manip_params["time_interval"]-1))
                close_value = features[end_ind, feature_ind]
                new_features[new_row_ind, new_feature_ind + 1] = close_value

                # Find the 'high' value
                high_value = np.max(features[row_ind:end_ind, feature_ind])
                new_features[new_row_ind, new_feature_ind + 2] = high_value

                # Find the 'low' value
                low_value = np.min(features[row_ind:end_ind, feature_ind])
                new_features[new_row_ind, new_feature_ind + 3] = low_value

                # Save the label -- to change probably
                new_labels[new_row_ind] = labels[row_ind]

                # Update row index for new_features
                new_row_ind += 1

        return new_features, new_labels


if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "train":
        candlestick = CandlestickManip(parameters)
        candlestick.manipulate()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
