from bs4 import BeautifulSoup
import copy


def xml2dict(xml_file, config_file):
    """Return XML file as a Python dictionary.
    
    Keyword arguments:
    xml_file -- marco XML file to convert to a Python dictionary.
    config_file -- global configuration dictionary."""
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
                    d["train"][data_name][model_name.text] = dict()

                    # Pull the algorithm/architecture for the model
                    model_arch = model_datum.find("algorithm")
                    try:
                        d["train"][data_name][model_name.text].update({"algorithm": model_arch.text})
                    
                    except AttributeError:
                        raise AttributeError("No algorithm specified in macro XML file.")

                    try:
                        arch_config = config_file["algorithms"][model_arch.text]

                    except KeyError:
                        raise KeyError("Algorithm {} key not present in .config.json. Please add default architecture/algorithm parameters to .config.json".format(model_arch.text))

                    # Pull user specified model parameters
                    model_params = model_datum.find_all("parameters")

                    if model_params != []:
                        d["train"][data_name][model_name.text]["parameters"] = dict()
                        for param_set in model_params:
                            # Create a deepcopy of the default rnn_config dictionary
                            tmp_dict = copy.deepcopy(arch_config)

                            for param in arch_config:
                                # Pull parameters tag in XML file
                                feat = param_set.find(param)
                                if feat is not None:
                                    feat = feat.text
                                    tmp_dict.update({param: feat})

                            d["train"][data_name][model_name.text]["parameters"] = tmp_dict

                    else:
                        # Still need to add parameters to the job_control dictionary
                        # regradless of mention in XML macro file
                        d["train"][data_name][model_name.text]["parameters"] = arch_config

                    # Add the plugin that the user will be utilizing
                    try:
                        d["train"][data_name][model_name.text].update({"plugin": model_datum["plugin"]})

                    except KeyError:
                        try:
                            d["train"][data_name][model_name.text].update({"plugin": config_file["plugins"][model_arch.text]})

                        except KeyError:
                            raise KeyError("Plugin for {} not available. Please specify plugin in .config.json.")
                    
                    # Pull all user specified data manipulations
                    xgbmanip = model_datum.find_all("xgboost")
                    ranforestmanip = model_datum.find_all("randomforest")
                    pcamanip = model_datum.find_all("pca")
                    candlemanip = model_datum.find_all("candlestick")

                    if xgbmanip != []:
                        # Pull xgb configuration from config dictionary
                        try:
                            xgb_config = config_file["datamanips"]["xgboost"]

                        except KeyError:
                            raise KeyError("Datamanip xgboost key not present in .config.json. Please add default xgboost parameters to .config.json")

                        d["train"][data_name][model_name.text]["xgboost"] = dict()
                        for xgb in xgbmanip:
                            # Create deepcopy of default xgb_config dictionary
                            tmp_dict = copy.deepcopy(xgb_config)

                            for param in xgb_config:
                                # Pull parameter's tag in XML file
                                feat = xgb.find(param)
                                if feat is not None:
                                    feat = feat.text
                                    tmp_dict.update({param: feat})
                            
                            # Add final tmp_dict to root
                            d["train"][data_name][model_name.text]["xgboost"][xgb["tag"]] = tmp_dict

                    else:
                        # Set to none if xgboost is not present in macro file
                        d["train"][data_name][model_name.text]["xgboost"] = None

                    if ranforestmanip != []:
                        # Pull randomforest configuration from config dictionary
                        try:
                            rf_config = config_file["datamanips"]["randomforest"]

                        except KeyError:
                            raise KeyError("Datamanip randomforest key not present in .config.json. Please add default randomforest parameters to .config.json")

                        d["train"][data_name][model_name.text]["randomforest"] = dict()
                        for rf in ranforestmanip:
                            # Create deepcopy of default rf_config dictionary
                            tmp_dict = copy.deepcopy(rf_config)

                            # Mere mention of Random Forest will store placeholder value of 1.
                            tmp_dict.update({"placeholder": 1})
                            d["train"][data_name][model_name.text]["randomforest"][rf["tag"]] = tmp_dict

                    else:
                        # Set to none if randomforest is not present in macro file
                        d["train"][data_name][model_name.text]["randomforest"] = None

                    if pcamanip != []:
                        # Pull pca config from config dictionary
                        try:
                            pca_config = config_file["datamanips"]["pca"]

                        except KeyError:
                            raise KeyError("Datamanip pca key not present in .config.json. Please add default pca parameters to .config.json")

                        d["train"][data_name][model_name.text]["pca"] = dict()
                        for pca in pcamanip:
                            # Create deepcopy of default pca_config dictionary
                            tmp_dict = copy.deepcopy(pca_config)

                            for param in pca_config:
                                # Pull parameter's tag in XML file
                                feat = pca.find(param)
                                if feat is not None:
                                    feat = feat.text
                                    tmp_dict.update({param: feat})
                            
                            # Add final tmp_dict to root
                            d["train"][data_name][model_name.text]["pca"][pca["tag"]] = tmp_dict

                    else:
                        # Set to none if pca is not present in the macro XML file
                        d["train"][data_name][model_name.text]["pca"] = None

                    if candlemanip != []:
                        try:
                            candle_config = config_file["datamanips"]["candlestick"]

                        except KeyError:
                            raise KeyError("Datamanip candlestick key not present in .config.json. Please add default candlestick parameters to .config.json")
                        
                        d["train"][data_name][model_name.text]["candlestick"] = dict()
                        for cand in candlemanip:
                            # Create deepcopy of default candle_config dictionary
                            tmp_dict = copy.deepcopy(candle_config)

                            for param in candle_config:
                                # Pull parameter's tag in XML file
                                feat = cand.find(param)
                                if feat is not None:
                                    feat = feat.text
                                    tmp_dict.update({param: feat})
                                
                            d["train"][data_name][model_name.text]["candlestick"][cand["tag"]] = tmp_dict

                    else:
                        # Set to none if candlestick is not present in macro XML file
                        d["train"][data_name][model_name.text]["candlestick"] = None
                                

    # Parse attack tag; skip if not specified in XML file
    if attack_data != []:
        d["attack"] = dict()
        for dataset in attack_data:
            # Get dataset path and name
            data_path = dataset["file"]
            data_name = data_path.split("/"); data_name = data_name[-1].split("."); data_name = data_name[0]
            d["attack"][data_name] = dict()
            d["attack"][data_name].update({"path": data_path})

            # Pull all the attack tags
            

    # Parse clean tag; skip if not specified in XML file
    if clean != []:
        d["clean"] = dict()
        clean_config = config_file["clean"]

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
                        tag_list.append(tag.text)

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
                raise KeyError("Clean key compress is not present in .config.json. Please add default compress parameters to .config.json")

            for compression in compress:
                format = compression.find("format")
                name = compression.find("name")
                path = compression.find("path")

                format = compress_config["format"] if format is None else format.text
                name = compress_config["name"] if name is None else name.text
                path = compress_config["path"] if path is None else path.text

                d["clean"]["compress"].update({name: {"format": format, "path": path}})

        else:
            d["clean"]["compress"] = None

    return d
