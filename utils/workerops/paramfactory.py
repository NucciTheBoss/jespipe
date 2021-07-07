import joblib
import uuid


def train_factory(name, model_name, dataframe, model_params, manip_params, save_path, manip_name, manip_tag, root_path):
    """Generate parameter dictionary to be sent out to plugin modules.
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


def attack_factory():
    pass


def clean_factory():
    pass
