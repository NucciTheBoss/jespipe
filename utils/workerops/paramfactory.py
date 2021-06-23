def paramfactory(name, model_name, dataframe, model_params, root_path, manip_name, manip_tag):
    """Generate parameter dictionary to be sent out to plugin modules.
    
    Keyword arguments:
    name -- name of the dataset.
    model_name -- name of the model to be trained.
    dataframe -- pandas DataFrame to be worked on.
    model_params -- user specified hyperparameters for model being trained.
    root_path -- root directory of jespipe.
    manip_name -- name of the manipulation used on the pandas DataFrame.
    manip_tag -- tag used to uniquely identify dataset manipulation."""
    # Create root dictionary that will eventually be returned to main.py
    d = dict()
    
    # Set dataset_name and model_name dataframe
    d["dataset_name"] = name; d["model_name"] = model_name

    # Convert dataframe to list and then set value in root dictionary
    d["dataframe"] = dataframe.values.tolist()
    
    # Set model parameters
    d["model_params"] = model_params

    # Generate save_path and log_path then add to root dictionary
    save_path = root_path + "/data/" + name + "/models"
    log_path = save_path + "/stat"
    d["save_path"] = save_path; d["log_path"] = log_path

    # Add tuple manip_info
    d["manip_info"] = (manip_name, manip_tag)

    return d
