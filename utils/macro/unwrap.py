import copy


def unwrap_train(train_dict):
    """Convert train job control dictionary to workable tuple.
    
    Keyword arguments:
    train_dict -- train job control dictionary to convert to workable tuple.
    
    Returned format:
    (dataset_name, dataset_path, model_name, algorithm, parameters, plugin, [(manip, {params: value})]"""
    # Create empty list that will eventually be returned to main.py
    root = list()

    # Split train_dict up by the stored datasets and loop over each dataset
    dataset_list = list(train_dict.items())
    for dataset in dataset_list:
        # Grab dataset name from top-level tuple and add to root tuple
        dataset_name = (dataset[0],)

        # Grab dictionary in index position 1 in tmp_level_1
        # and add path key -> value to root tuple
        # Use deepcopy to prevent any reference issues
        tmp_dict_level_1 = copy.deepcopy(dataset[1])
        dataset_path = (tmp_dict_level_1["path"],)

        # Delete path key -> value so only models are left behind
        try:
            del tmp_dict_level_1["path"]

        except KeyError:
            continue

        # Loop over each model(s) in the dictionary
        for model in tmp_dict_level_1:
            root_tuple = dataset_name + dataset_path
            root_tuple += (model,)

            # Create dictionary containing the datamanips for model
            # and loop over each of the manips
            tmp_dict_level_2 = copy.deepcopy(tmp_dict_level_1[model])

            # Pull out and delete algorithm, paremeters, and plugin key
            algorithm = tmp_dict_level_2["algorithm"]
            param_dict = tmp_dict_level_2["parameters"]
            plugin = tmp_dict_level_2["plugin"]
            root_tuple += (algorithm, param_dict, plugin)

            try:
                del tmp_dict_level_2["algorithm"]
                del tmp_dict_level_2["parameters"]
                del tmp_dict_level_2["plugin"]

            except KeyError:
                continue

            # Define list that all manips can be added to before
            # updating root list
            manip_list = list()
            for manip_tech in tmp_dict_level_2:
                if tmp_dict_level_2[manip_tech] is not None:
                    manip_tuple = (manip_tech,)
                    tmp_dict_level_3 = tmp_dict_level_2[manip_tech]

                    # Convert tmp_dict_level_3 to a list and
                    # append to manip_list
                    tmp_submanip_list = list(tmp_dict_level_3.items())
                    manip_tuple += (tmp_submanip_list,)
                    manip_list.append(manip_tuple)

            if manip_list != []:
                root_tuple += (manip_list,)

            else:
                root_tuple += (None,)

            root.append(root_tuple)

    return root


def unwrap_attack(attack_dict):
    """Convert attack job control dictionary to workable tuple.
    
    Keyword arguments:
    attack_dict -- attack job control dictionary to convert to workable tuple.
    
    Returned format:
    (dataset_name, dataset_path, [(attack, {attack_tag: params})])"""
    # Create empty list that will eventually be returned to main.py
    root = list()

    # Split attack_dict up into the stored datasets and loop over each dataset
    dataset_list = list(attack_dict.items())
    for dataset in dataset_list:
        # Grab dataset name from top-level tuple and add to root tuple
        dataset_name = (dataset[0],)

        # Grab dictionary in index position 1 in tmp_level_1
        # and add path key -> value to root tuple
        # Use deepcopy to prevent any reference issues
        tmp_dict_level_1 = copy.deepcopy(dataset[1])
        dataset_path = (tmp_dict_level_1["path"],)

        # Delete path key -> value so only attacks are left behind
        try:
            del tmp_dict_level_1["path"]

        except KeyError:
            continue

        # Convert remainder of tmp_dict_level_1 to a list
        attack_list = list(tmp_dict_level_1.items())
        tmp_list = copy.deepcopy(attack_list)
        for attack in tmp_list:
            if None in attack:
                attack_list.remove(attack)

        if attack_list != []:    
            root_tuple = dataset_name + dataset_path + (attack_list,)

        else:
            root_tuple = dataset_name + dataset_path + (None,)

        root.append(root_tuple)

    return root
