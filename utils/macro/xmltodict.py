from bs4 import BeautifulSoup


def xmltodict(xml_file):
    """Return XML file as a Python dictionary.
    
    Keyword arguments:
    xml_file -- marco XML file to convert to a Python dictionary."""
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

                    # TODO: Add attribute to manipulation tag that allows it to be uniquely identified.
                    if xgbmanip != []:
                        # Pull n_features text value and convert to int
                        n_feat = xgbmanip.find("n_features"); n_feat = int(n_feat.text)
                        d["train"][data_name][model_name.text]["xgboost"] = {"n_features": n_feat}

                    if ranforestmanip != []:
                        # Mere mention of randomforest will tell the pipeline to use it
                        # Use 1 as a placeholder value
                        d["train"][data_name][model_name.text]["randomforest"] = 1

                    if pcamanip != []:
                        # Pull n_features text value and convert to int
                        n_feat = pcamanip.find("n_features"); n_feat = int(n_feat.text)

                    if candlemanip != []:
                        pass

    # Parse attack tag; skip if not specified in XML file
    if attack_data != []:
        d["attack"] = dict()
        for dataset in attack_data:
            pass

    # Parse clean tag; skip if not specified in XML file
    if clean_data != []:
        d["clean"] = dict()
        for dataset in clean_data:
            # TODO: Still working on the specifics for the clean tag
            pass
