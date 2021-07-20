import json
import os
from typing import Any

import joblib
import pandas as pd


def features(file_path: str, object: Any) -> None:
    """
    Save model test features as a pickle. Saved pickle
    file is named test_features.pkl.
    
    ### Parameters:
    :param file_path: System location to save test features pickle file to.
    :param object: Object containing test features.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/test_features.pkl")


def labels(file_path: str, object: Any) -> None:
    """
    Save model test labels as pickle. Saved pickle
    file is named test_labels.pkl
    
    ### Parameters:
    :param file_path: System location to save test labels pickle file to.
    :param object: Object containing test labels.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/test_labels.pkl")


def pickle_object(file_path: str, name: str, object: Any) -> None:
    """
    Save any generic object as pickle. Useful for generic analysis files.

    ### Parameters:
    :param file_path: System location to save pickle file.
    :param name: File name to use for the Object pickle file.
    :param object: Object containing data to pickle.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/{}.pkl".format(name))


def adver_example(file_path: str, min_change: float, object: Any) -> None:
    """
    Save a generated adversarial example as a pickle.
    
    ### Parameters:
    :param file_path: -- System location to save the adversarial example pickle file to.
    :param min_change: Minimum allowed change for the attack algorithm. Used
    as the name for the pickle file.
    :param object: Object containing generated adversarial example.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    joblib.dump(object, file_path + "/{}.pkl".format(min_change))


def compress_dataframe(file_path: str, name: str, dataset: pd.DataFrame) -> None:
    """
    Save a pandas DataFrame as a compressed .csv.gz file. Uses gzip compression
    algorithm to compress DataFrame.
    
    ### Parameters:
    :param file_path: System location to save the Pandas DataFrame to.
    :param name: File name to use for the Pandas DataFrame file.
    :param dataset: Pandas DataFrame to save as a compressed .csv.gz file.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.to_csv(file_path + "/{}.csv.gz".format(name), index=False, header=None, compression="gzip")


def dataframe(file_path: str, name: str, dataset: pd.DataFrame) -> None:
    """
    Save a Pandas DataFrame as an uncompressed .csv file. Does not use
    any compression algorithms.
    
    ### Parameters:
    :param file_path: System location to save the Pandas DataFrame to.
    :param name: File name to use for the Pandas DataFrame file.
    :param dataset: Pandas DataFrame to save as an uncompressed .csv file.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    dataset.to_csv(file_path + "/{}.csv".format(name), index=False, header=None)


def dictionary(file_path: str, name: str, dictdata: dict) -> None:
    """
    Save a Python dictionary to a .json file. Dictionary must
    be considered well-formed JSON data.
    
    ### Parameters:
    :param file_path: System location to save the dictionary data to.
    :param name: File name to use for the dictionary data file.
    :param dictdata: Dictionary to save as a .json file.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    fout = open(file_path + "/{}.json".format(name), "wt"); fout.write(json.dumps(dictdata)); fout.close()


def text(file_path: str, name: str, textdata: str) -> None:
    """
    Save unstructured string data as a .txt file.
    
    ### Parameters:
    :param file_path: System location to save the unstructured string data to.
    :param name: File name to use for the unstructured string data file.
    :param textdata: Unstructured string data to save as a .txt file.
    """
    if os.path.exists(file_path) is False:
        os.makedirs(file_path)

    fout = open(file_path + "/{}".format(name), "wt"); fout.write(textdata); fout.close()
