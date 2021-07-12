from typing import Any
from bs4 import BeautifulSoup
import copy


def _data_converter(data: str, deftype: str) -> Any:
    """
    Convert data to type specified by user in XML control file.
    Defaults to data type str if user specifies invalid type.

    ### Parameters
    :param data: Data to convert to user-defined type.
    :param deftype: User-defined type. Supported types are int|float|bool|str.

    ### Returns
    :return: Data converted to the user-defined type.
    """
    if deftype == "int":
        return int(data)

    elif deftype == "float":
        return float(data)

    elif deftype == "bool":
        return bool(data)

    elif deftype == "str":
        return str(data)

    else:
        return str(data)


def xml2dict(xml_file: str, config_dict: dict) -> dict:
    """
    Convert XML control file to an easily processable dictionary.
    
    ### Parameters
    :param xml_file: System file path reference to XML control file.
    :param config_dict: Global Jespipe configuration dictionary.

    ### Returns
    :return: XML control file processed into dictionary.

    ### Raises
    - AttributeError
      - Raised if an attribute for an element in the XML control file is not found.

    - KeyError
      - Raised if a key is not found in the configuration dictionary.
    """
    fin = open(xml_file, "rt"); xml_data = fin.read(); fin.close()
    soup = BeautifulSoup(xml_data, "xml")

    # Create root of dictionary
    d = dict()

    # Split macro file into training, attack, and cleanup
    train = soup.find("train"); attack = soup.find("attack"); clean = soup.find("clean")
    train = BeautifulSoup(str(train), "xml"); attack = BeautifulSoup(str(attack), "xml"); clean = BeautifulSoup(str(clean), "xml")
    train_data = train.find_all("dataset"); attack_data = attack.find_all("dataset")
    
    # Parse train tag; skip if not specified in XML file
    if train_data != []:
        d["train"] = dict()
        for dataset in train_data:
            # Get dataset path and name
            data_path = dataset["file"]
            data_name = data_path.split("/"); data_name = data_name[-1].split("."); data_name = data_name[0]
            d["train"][data_name] = dict()
            d["train"][data_name].update({"path": data_path})
            
            # Add model data manipulations and their parameters
            model_data = dataset.find_all("model")
            if model_data != []:
                for model_datum in model_data:
                    # Grab model name and add to dictionary
                    model_name = model_datum.find("name")
                    d["train"][data_name][model_name["value"]] = dict()

                    # Pull the algorithm/architecture for the model
                    model_arch = model_datum.find("algorithm")
                    try:
                        d["train"][data_name][model_name["value"]].update({"algorithm": model_arch["value"]})
                    
                    except AttributeError:
                        raise AttributeError("No algorithm specified in macro XML file.")

                    try:
                        arch_config = config_dict["algorithms"][model_arch["value"]]

                    except KeyError:
                        raise KeyError("Algorithm {} key not present in .config.json. Please add default architecture/algorithm parameters to .config.json".format(model_arch["value"]))

                    # Pull user specified model parameters
                    model_params = model_datum.find_all("parameters")

                    if model_params != []:
                        d["train"][data_name][model_name["value"]]["parameters"] = dict()
                        for param_set in model_params:
                            # Create a deepcopy of the default rnn_config dictionary
                            tmp_dict = copy.deepcopy(arch_config)

                            for param in arch_config:
                                # Pull parameters tag in XML file
                                feat = param_set.find(param)
                                if feat is not None:
                                    feat = _data_converter(feat["value"], feat["type"])
                                    tmp_dict.update({param: feat})

                            d["train"][data_name][model_name["value"]]["parameters"] = tmp_dict

                    else:
                        # Still need to add parameters to the job_control dictionary
                        # regradless of mention in XML macro file
                        d["train"][data_name][model_name["value"]]["parameters"] = arch_config

                    # Add the plugin that the user will be utilizing
                    try:
                        d["train"][data_name][model_name["value"]].update({"plugin": model_datum["plugin"]})

                    except KeyError:
                        try:
                            d["train"][data_name][model_name["value"]].update({"plugin": config_dict["plugins"][model_arch["value"]]})

                        except KeyError:
                            raise KeyError("Plugin for {} not available. Please specify plugin in .config.json.")
                    
                    # Pull available manipulations from configuration dictionary
                    manip_list = [k for k in config_dict["datamanips"]]

                    # Loop through available manips
                    for manip in manip_list:
                        manip_config = config_dict["datamanips"][manip]
                        manip_content = model_datum.find_all(manip)

                        if manip_content != []:
                            d["train"][data_name][model_name["value"]][manip] = dict()
                            for content in manip_content:
                                # Create deepcopy of default configuration dictionary
                                tmp_dict = copy.deepcopy(manip_config)

                                # Loop through all parameters to construct manipulation
                                for param in manip_config:
                                    feat = content.find(param)
                                    if feat is not None:
                                        feat = _data_converter(feat["value"], feat["type"])
                                        tmp_dict.update({param: feat})

                                d["train"][data_name][model_name["value"]][manip][manip["tag"]] = tmp_dict

                        else:
                            d["train"][data_name][model_name["value"]][manip] = None

    # Parse attack tag; skip if not specified in XML file
    if attack_data != []:
        d["attack"] = dict()
        for dataset in attack_data:
            # Get dataset path and name
            data_path = dataset["file"]
            data_name = data_path.split("/"); data_name = data_name[-1].split("."); data_name = data_name[0]
            d["attack"][data_name] = dict()
            d["attack"][data_name].update({"path": data_path})

            # Pull all the attack tags from the config dictionary
            key_list = [k for k in config_dict["attacks"]]

            # Loop through all available attacks to see if they are in the macro file
            for attack in key_list:
                current_attacks = dataset.find_all(attack)

                if current_attacks != []:
                    # Pull config dictionary
                    attack_config = config_dict["attacks"][attack]
                    d["attack"][data_name][attack] = dict()

                    for current_attack in current_attacks:
                        # Create deepcopy of current attack config
                        tmp_dict = copy.deepcopy(attack_config)

                        for param in attack_config:
                            feat = current_attack.find(param)
                            if feat is not None:
                                feat = _data_converter(feat["value"], feat["type"])
                                tmp_dict.update({param: feat})

                        d["attack"][data_name][attack][current_attack["tag"]] = dict()
                        try:
                            d["attack"][data_name][attack][current_attack["tag"]]["plugin"] = current_attack["plugin"]

                        except KeyError:
                            try:
                                d["attack"][data_name][attack][current_attack["tag"]]["plugin"] = config_dict["plugins"][current_attack.name]

                            except KeyError:
                                raise KeyError("Plugin for {} not available. Please specify plugin in .config.json.".format(current_attack.name))

                        d["attack"][data_name][attack][current_attack["tag"]]["model_plugin"] = current_attack["model_plugin"]
                        d["attack"][data_name][attack][current_attack["tag"]]["params"] = tmp_dict

                else:
                    d["attack"][data_name][attack] = None

    # Parse clean tag; skip if not specified in XML file
    if clean != []:
        d["clean"] = dict()
        clean_config = config_dict["clean"]

        # There are only three supported tags for clean stage: plot, clean_tmp, and compress
        plots = clean.find_all("plot")
        if plots != []:
            # Add plot top-level dict to root dictionary
            d["clean"]["plot"] = dict()
            for plot in plots:
                d["clean"]["plot"][plot["tag"]] = {"plugin": plot["plugin"]}

                # Find specified tags in macro file
                tags = plot.find_all("tag")
                if tags != []:
                    tag_list = list()
                    for tag in tags:
                        tag_list.append(tag["value"])

                    d["clean"]["plot"][plot["tag"]].update({"tags": tag_list})

        else:
            d["clean"]["plot"] = None

        clean_tmp = clean.find("clean_tmp")
        if clean_tmp is None:
            try:
                d["clean"]["clean_tmp"] = clean_config["clean_tmp"]

            except KeyError:
                d["clean"]["clean_tmp"] = 0

        else:
            d["clean"]["clean_tmp"] = 1

        compress = clean.find_all("compress")
        if compress != []:
            d["clean"]["compress"] = dict()
            try:
                compress_config = clean_config["compress"]

            except KeyError:
                raise KeyError("Clean key compress is not present in .config.json. Please add default compression parameters to .config.json")

            for compression in compress:
                format = compression.find("format")
                name = compression.find("name")
                path = compression.find("path")

                format = compress_config["format"] if format is None else format["value"]
                name = compress_config["name"] if name is None else name["value"]
                path = compress_config["path"] if path is None else path["value"]

                d["clean"]["compress"].update({name: {"format": format, "path": path}})

        else:
            d["clean"]["compress"] = None

    return d
