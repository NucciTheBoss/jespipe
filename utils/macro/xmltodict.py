from bs4 import BeautifulSoup


def xmltodict(xml_file, xgb_config, rf_config, pca_config,
              candle_config, cw_inf_config, bim_config, fgsm_config):
    """Return XML file as a Python dictionary.
    
    Keyword arguments:
    xml_file -- marco XML file to convert to a Python dictionary.
    xgb_config -- configuration dictionary for XGBoost feature selection.
    rf_config -- configuration dictionary for Random Forest feature selection.
    pca_config -- configuration dictionary for PCA dimensionality reduction.
    candle_config -- configuration dictionary for Candlestick trend extraction.
    cw_inf_config -- configuration dictionary for CW_inf attack.
    bim_config -- configuration dictionary for BIM attack.
    fgsm_config -- configuration dictionary for FGSM attack."""
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

                    # Pull all user specified data manipulations
                    xgbmanip = model_datum.find_all("xgboost")
                    ranforestmanip = model_datum.find_all("randomforest")
                    pcamanip = model_datum.find_all("pca")
                    candlemanip = model_datum.find_all("candlestick")

                    if xgbmanip != []:
                        for xgb in xgbmanip:
                            # Pull n_features text value and convert to int
                            n_feat = xgb.find("n_features"); n_feat = int(n_feat.text)
                            d["train"][data_name][model_name.text][xgb["tag"]] = dict()
                            d["train"][data_name][model_name.text][xgb["tag"]].update({"n_features": n_feat})

                    if ranforestmanip != []:
                        for rf in ranforestmanip:
                            # Mere mention of randomforest will tell the pipeline to use it
                            # Use 1 as a placeholder value
                            d["train"][data_name][model_name.text][rf["tag"]] = dict()
                            d["train"][data_name][model_name.text][rf["tag"]].update({"placeholder": 1})

                    if pcamanip != []:
                        for pca in pcamanip:
                            # Pull n_features text value and convert to int
                            n_feat = pca.find("n_features"); n_feat = int(n_feat.text)
                            d["train"][data_name][model_name.text][pca["tag"]] = dict()
                            d["train"][data_name][model_name.text][pca["tag"]].update({"n_features": n_feat})

                    if candlemanip != []:
                        for cand in candlemanip:
                            # Pull time_interval text value and convert to int
                            time_int = cand.find("time_interval"); int(time_int.text)
                            d["train"][data_name][model_name.text][cand["tag"]] = dict()
                            d["train"][data_name][model_name.text][cand["tag"]].update({"time_interval": time_int})

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
            cw_inf_attacks = attack_data.find_all("cw_inf")
            bim_attacks = attack_data.find_all("bim")
            fgsm_attacks = attack_data.find_all("fgsm")

            if cw_inf_attacks != []:
                pass

            if bim_attacks != []:
                pass

            if fgsm_attacks != []:
                pass

    # Parse clean tag; skip if not specified in XML file
    if clean_data != []:
        d["clean"] = dict()
        for dataset in clean_data:
            # TODO: Still working on the specifics for the clean tag
            pass
