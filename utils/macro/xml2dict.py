from bs4 import BeautifulSoup
import copy


def xml2dict(xml_file, xgb_config, rf_config, pca_config,
             candle_config, cw_inf_config, bim_config, fgsm_config, rnn_config):
    """Return XML file as a Python dictionary.
    
    Keyword arguments:
    xml_file -- marco XML file to convert to a Python dictionary.
    xgb_config -- configuration dictionary for XGBoost feature selection.
    rf_config -- configuration dictionary for Random Forest feature selection.
    pca_config -- configuration dictionary for PCA dimensionality reduction.
    candle_config -- configuration dictionary for Candlestick trend extraction.
    cw_inf_config -- configuration dictionary for CW_inf attack.
    bim_config -- configuration dictionary for BIM attack.
    fgsm_config -- configuration dictionary for FGSM attack.
    rnn_config -- configuration dictionary for RNN algorithm."""
    fin = open(xml_file, "rt"); xml_data = fin.read(); fin.close()
    soup = BeautifulSoup(xml_data, "xml")

    # Create root of dictionary
    d = dict()

    # Split macro file into training, attack, and cleanup
    train = soup.find("train"); attack = soup.find("attack"); clean = soup.find("clean")
    train = BeautifulSoup(str(train), "xml"); attack = BeautifulSoup(str(attack), "xml"); clean = BeautifulSoup(str(clean), "xml")
    train_data = train.find_all("dataset"); attack_data = attack.find_all("dataset"); clean_data = clean.find_all("dataset")
    
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
                    model_name = model_datum.find("name")
                    d["train"][data_name][model_name.text] = dict()

                    # Pull user specified model parameters
                    model_params = model_datum.find_all("parameters")

                    if model_params != []:
                        d["train"][data_name][model_name.text]["parameters"] = dict()
                        for param_set in model_params:
                            # Create a deepcopy of the default rnn_config dictionary
                            tmp_dict = copy.deepcopy(rnn_config)

                            for param in rnn_config:
                                # Pull parameters tag in XML file
                                feat = param_set.find(param)
                                if feat is not None:
                                    feat = feat.text
                                    tmp_dict.update({param: feat})

                            d["train"][data_name][model_name.text]["parameters"] = tmp_dict

                    else:
                        # Still need to add parameters to the job_control dictionary
                        # regradless of mention in XML macro file
                        d["train"][data_name][model_name.text]["parameters"] = rnn_config

                    # Pull all user specified data manipulations
                    xgbmanip = model_datum.find_all("xgboost")
                    ranforestmanip = model_datum.find_all("randomforest")
                    pcamanip = model_datum.find_all("pca")
                    candlemanip = model_datum.find_all("candlestick")

                    if xgbmanip != []:
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
            cw_inf_attacks = dataset.find_all("cw_inf")
            bim_attacks = dataset.find_all("bim")
            fgsm_attacks = dataset.find_all("fgsm")

            if cw_inf_attacks != []:
                # Add CW_inf branch to root dictionary
                d["attack"][data_name]["CW_inf"] = dict()
                for cw_inf in cw_inf_attacks:
                    # Create deepcopy of default cw_inf_config
                    tmp_dict = copy.deepcopy(cw_inf_config)

                    for param in cw_inf_config:
                        feat = cw_inf.find(param)
                        if feat is not None:
                            feat = feat.text
                            tmp_dict.update({param: feat})

                    d["attack"][data_name]["CW_inf"][cw_inf["tag"]] = tmp_dict                    

            else:
                # Set to none if CW_inf is not mentioned in XML macro file
                d["attack"][data_name]["CW_inf"] = None

            if bim_attacks != []:
                # Add BIM branch to root dictionary
                d["attack"][data_name]["BIM"] = dict()
                for bim in bim_attacks:
                    # Create deepcopy of default bim_config
                    tmp_dict = copy.deepcopy(bim_config)

                    for param in bim_config:
                        feat = bim.find(param)
                        if feat is not None:
                            feat = feat.text
                            tmp_dict.update({param: feat})

                    d["attack"][data_name]["BIM"][bim["tag"]] = tmp_dict

            else:
                # Set to none if BIM is not mentioned in macro XML file
                d["attack"][data_name]["BIM"] = None

            if fgsm_attacks != []:
                # Add FGSM branch to root dictionary
                d["attack"][data_name]["FGSM"] = dict()
                for fgsm in fgsm_attacks:
                    # Create deepcopy of default fgsm_config
                    tmp_dict = copy.deepcopy(fgsm_config)

                    for param in fgsm_config:
                        feat = fgsm.find(param)
                        if feat is not None:
                            feat = feat.text
                            tmp_dict.update({param: feat})

                    d["attack"][data_name]["FGSM"][fgsm["tag"]] = tmp_dict

            else:
                # Set to none if FGSM is not mentioned in the macro XML file
                d["attack"][data_name]["FGSM"] = None

    # TODO: Figure out from Sheila next week what she wants me to do with this section
    # # Parse clean tag; skip if not specified in XML file
    # if clean_data != []:
    #     d["clean"] = dict()
    #     for dataset in clean_data:
    #         # TODO: Still working on the specifics for the clean tag
    #         pass

    return d
