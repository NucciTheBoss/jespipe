import joblib
import uuid


def train_factory(name, model_name, dataframe, model_params, manip_params, save_path, manip_name, manip_tag, root_path):
    """Generate parameter dictionary that will be sent out to the training plugins for the training stage.
    Save as a pickle and return a file path reference to that pickle.
    
    Keyword arguments:
    name -- name of the dataset.
    model_name -- name of the model to be trained.
    dataframe -- pandas DataFrame to be worked on.
    model_params -- user specified hyperparameters for model being trained.
    manip_params -- user specified parameters for the data manipulation.
    save_path -- where to save manipulation specific model files.
    manip_name -- name of the manipulation used on the pandas DataFrame.
    manip_tag -- tag used to uniquely identify dataset manipulation.
    root_path -- root directory of jespipe."""
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


def attack_factory(name, model_path, model_test_features, attack_params, save_path, root_path):
    """Generate parameter dictionary that will be sent out to the attack plugins for the attack stage.
    Save as a pickle and return a file path reference to that pickle.
    
    Keyword arguments:
    name -- name of the attack.
    model_path -- path of model to attack.
    model_test_features -- the data to manipulate.
    attack_params -- parameters to use for the attack.
    save_path -- where to save the adversarial examples.
    root_path -- root directory of jespipe."""
    d = dict()

    d["name"] = name
    d["model_path"] = model_path
    d["model_test_features"] = model_test_features
    d["params"] = attack_params

    # Append name to save path
    d["save_path"] = save_path + "/" + name

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)

    return pickle_path


def attack_train_factory(adver_features, model_labels, log_path, model_path, root_path):
    """Generate parameter dictionary that will be sent out to the training plugins for the attack stage.
    Save as a pickle and return a file path reference to that pickle.
    
    Keyword arguments:
    adver_features -- list containing file path references to adversarial data.
    model_labels -- the labels to evaluate the model on.
    log_path -- where to save the data collected.
    model_path -- file path to model.
    root_path -- root directory of jespipe."""
    d = dict()

    d["adver_features"] = adver_features
    d["model_labels"] = model_labels
    d["log_path"] = log_path
    d["model_path"] = model_path

    # Establish path to file in .tmp directory and dump dictionary
    pickle_path = root_path + "/data/.tmp/" + str(uuid.uuid4()) + ".pkl"
    joblib.dump(d, pickle_path)

    return pickle_path



def clean_factory():
    """Generate parameter dictionary that will be sent out to the cleaning plugins for the cleaning stage.
    Save as a pickle and return a file path reference to that pickle.
    
    Keyword arguments:"""
    # TODO: Update this function once you revisit the cleaning stage next week
    pass
