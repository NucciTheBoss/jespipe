import uuid
from typing import List

import joblib
import numpy as np
import pandas as pd


def manip_factory(dataset_path: str, manip_tag: str, manip_params: str, save_path: str, 
            tmp_path: str, root_path: str) -> str:
    """
    Create parameter dictionary that will be sent out to the user-specified manipulation plugin
    in the training stage. Save the parameter dictionary as a pickle file.

    ### Parameters:
    :param dataset_path: System file path to dataset.
    :param manip_tag: Tag to uniquely identify specific dataset manipulation.
    :param manip_params: User-specified parameters for the data manipulation.
    :param save_path: Where to save output data files.
    :param tmp_path: System location of temp directory to store temporary files.
    :param root_path: Root directory of Jespipe.

    ### Returns:
    :return: System file path reference to pickled parameter dictionary.
    """
    # Create root dictionary that will be converted to a pickle
    d = dict()

    # Create parameter dictionary
    d["dataset"] = dataset_path; d["manip_tag"] = manip_tag; d["manip_params"] = manip_params
    d["save_path"] = save_path; d["tmp_path"] = tmp_path

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)
    
    return pickle_path


def train_factory(name: str, model_name: str, dataframe: pd.DataFrame, model_params: dict, manip_params: dict, 
                    save_path: str, manip_name: str, manip_tag: str, root_path: str) -> str:
    """
    Create parameter dictionary that will be sent out to the user-specified training plugin
    in the training stage. Save the parameter dictionary as a pickle file.
    
    ### Parameters:
    :param name: Name of the dataset.
    :param model_name: Name to use for trained model.
    :param dataframe: Pandas DataFrame to train the model on.
    :param model_params: User-specified hyperparameters for the model being trained.
    :param manip_params: User-specified parameters for the data manipulation.
    :param save_path: Where to save output data files.
    :param manip_name: name of the manipulation used on the pandas DataFrame.
    :param manip_tag: Tag to uniquely identify specific dataset manipulation.
    :param root_path: Root directory of Jespipe.

    ### Returns:
    :return: System file path reference to pickled parameter dictionary.
    """
    # Create root dictionary that will be converted to a pickle
    d = dict()
    
    # Set dataset_name, model_name, dataframe, model parameters, and manipulation parameters
    d["dataset_name"] = name; d["model_name"] = model_name; d["dataframe"] = dataframe
    d["model_params"] = model_params; d["manip_params"] = manip_params

    # Generate save_path and log_path then add to root dictionary
    log_path = save_path + "/stat"
    d["save_path"] = save_path; d["log_path"] = log_path

    # Add tuple manip_info
    d["manip_info"] = (manip_name, manip_tag)

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)
    
    return pickle_path


def attack_factory(name: str, model_path: str, model_test_features: np.ndarray, attack_params: dict, 
                    save_path: str, root_path: str) -> str:
    """
    Create parameter dictionary that will be sent out to the user-specified attack plugin 
    in the attack stage. Save the parameter dictionary as a pickle file.
    
    ### Parameters:
    :param name: Name of the attack.
    :param model_path: System file path of model to attack.
    :param model_test_features: The data to manipulate for the attack.
    :param attack_params: Parameters to use for the attack.
    :param save_path: System location save the adversarial examples.
    :param root_path: Root directory of Jespipe.

    ### Returns:
    :return: System file path reference to pickled parameter dictionary.
    """
    d = dict()

    d["name"] = name
    d["model_path"] = model_path
    d["model_test_features"] = model_test_features
    d["attack_params"] = attack_params

    # Append name to save path
    d["save_path"] = save_path + "/" + name

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)

    return pickle_path


def attack_train_factory(adver_features: List[str], model_labels: np.ndarray, 
                            log_path: str, model_path: str, root_path: str) -> str:
    """
    Create parameter dictionary that will be sent out to the user-specified training plugin 
    in the attack stage. Save the parameter dictionary as a pickle file.
    
    ### Parameters:
    :param adver_features: List containing system file path references to adversarial data for attack.
    :param model_labels: The target feature(s) to evaluate the model on.
    :param log_path: System location to save data collected on model during attack.
    :param model_path: System file path of model.
    :param root_path: Root directory of Jespipe.

    ### Returns:
    :return: System file path reference to pickled parameter dictionary.
    """
    d = dict()

    d["adver_features"] = adver_features
    d["model_labels"] = model_labels
    d["log_path"] = log_path
    d["model_path"] = model_path

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)

    return pickle_path


def clean_factory() -> str:
    """
    Generate parameter dictionary that will be sent out to the cleaning plugins for the cleaning stage.
    Save as a pickle and return a file path reference to that pickle.
    
    ### Parameters:
    - TODO

    ### Returns:
    :return: System file path reference to pickled parameter dictionary.
    """
    # TODO: Update this function once you revisit the cleaning stage next week
    pass
