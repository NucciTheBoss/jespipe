from typing import List, Tuple

import numpy as np

from ..filesystem import getpaths as gp


def generate_train(macro_list: List) -> List[Tuple[str, str, str, str, dict, str, str, str, str, dict]]:
    """
    Generate scatterable training directive list using a list of low-level macro tuples.
    
    ### Paramters:
    :param macro_list: Low-level training tuple list to be convertered into a scatterable list
    for later task delegation to the worker nodes.

    ### Returns:
    :return: [(dataset_name, dataset_path, model_name, algorithm_name, algorithm_parameters, 
    model_plugin_path, data_manip_name, data_manip_tag, data_manip_plugin, data_manip_parameters)]
    - Positional value of each index in a tuple contained in the scatterable list:
      - 0: "dataset_name"
      - 1: "/path/to/dataset"
      - 2: "model_name"
      - 3: "algorithm_name"
      - 4: {"algorithm_param": value}
      - 5: "/path/to/model/plugin.py"
      - 6: "data_manipulation_name"
      - 7: "data_manipulation_tag"
      - 8: "/path/to/manip/plugin.py"
      - 9: {"manip_param": value}
    """
    # Initialize root list that will be returned to main.py
    root = list()

    # Loop over macro list to create scatterable directive list
    for directive in macro_list:
        # Pull dataset name, file path, model name, and parameters
        dataset_name = directive[0]
        dataset_path = directive[1]
        model_name = directive[2]
        algorithm = directive[3]
        parameters = directive[4]
        plugin = directive[5]

        # Grab manip list and iterate over each manipulation
        manip_type_list = directive[6]

        if manip_type_list is not None:
            for manip_type in manip_type_list:
                # Grab manipulation type: eg. xgboost, pca, randomforest, candlestick.
                manip_type_name = manip_type[0]

                # Grab list of manip names and parameters
                manip_list = manip_type[1]
                for manip in manip_list:
                    # Create directive tuple and add to root list
                    root.append((dataset_name, dataset_path, model_name, algorithm, parameters, plugin, manip_type_name, manip[0], manip[1]["plugin"], manip[1]["manip_params"]))

        else:
            root.append((dataset_name, dataset_path, model_name, algorithm, parameters, plugin, None, None, None, None))

    return root


def generate_attack(macro_list: List) -> List[Tuple[str, str, str, str, str, str, dict, str, str]]:
    """
    Generate scatterable attack directive list using a list of low-level macro tuples.
    
    ### Parameters:
    :param macro_list: Low-level attack tuple list to be convertered into a scatterable list
    for later task delegation to the worker nodes.

    ### Returns:
    :return: [(dataset_name, dataset_path, attack_type, attack_tag, attack_plugin, model_plugin, 
    attack_parameters, model_path, model_root_path)]
    - Positional value of each index in a tuple contained in the scatterable list:
      - 0: "dataset_name"
      - 1: "/path/to/dataset"
      - 2: "attack_type"
      - 3: "attack_tag"
      - 4: "/path/to/attack/plugin.py"
      - 5: "/path/to/model/plugin.py"
      - 6: {"attack_param": value}
      - 7: "/path/to/model"
      - 8: "/path/to/model/root/path"
    """
    # Initialize root list that will be returned to main.py
    root = list()

    # Loop over macro list to create scatterable directive list
    for directive in macro_list:
        # Pull dataset name and dataset path
        dataset_name = directive[0]
        dataset_path = directive[1]

        # Get attack name and tag
        for attack in directive[2]:
            attack_name = attack[0]

            if attack[1] is not None:
                attack_tags = [k for k in attack[1]]
                for attack_tag in attack_tags:
                    attack_plugin = attack[1][attack_tag]["plugin"]
                    model_plugin = attack[1][attack_tag]["model_plugin"]
                    params = attack[1][attack_tag]["params"]

                    # Loop through each model and then append to root list
                    if directive[3] != []:
                        for model in directive[3]:
                            # Grab model root path (use this to find test features and test labels)
                            model_root = model[0].split("/"); del model_root[-1]; model_root = "/".join(model_root)
                            root.append((dataset_name, dataset_path, attack_name, attack_tag, attack_plugin, model_plugin, params, model[0], model[1], model_root))

                    else:
                        root.append((dataset_name, dataset_path, attack_name, attack_tag, attack_plugin, model_plugin, params, None, None))

            else:
                root.append((dataset_name, dataset_path, attack_name, None, None, None, None, None, None))

    return root


def generate_clean(plot_dict: dict, save_path: str, data_path: str) -> List[Tuple[str, List[str], str, str]]:
    """
    Generate scatterable clean directive list using a list of low-level macro tuples.
    
    ### Parameters:
    :param plot_dict: Dictionary containing user-defined macros for the cleaning stage.
    :param save_path: System location to save user-generated plots.
    :param data_path: System location containing trained models and relevant data.

    ### Returns:
    :return: [(plugin_path, tag_list, plotting_tag, save_path)]
    - Positional value of each index in a tuple contained in the scatterable list:
      - 0: "/path/to/plotting/plugin.py"
      - 1: "/path/to/model/directories"
      - 2: "plot_name"
      - 3: "/location/to/save/plots" 
    """
    root = list()

    for key in plot_dict:
        plugin_path = plot_dict[key]["plugin"]
        tag_list = plot_dict[key]["tags"]

        root_paths = gp.getdirs(data_path)
        for i in range(0, len(tag_list)):
            for path in root_paths:
                if tag_list[i] in path and "adver_example" not in path:
                    tag_list[i] = path
                    break

        root.append((plugin_path, tag_list, key, save_path))

    return root


def slice(directive_list: List, mpi_size: int) -> List[List]:
    """
    Slice up a directive list into a number of chuncks.
    The total number of chunks is determined by how many worker 
    nodes are available in the MPI.COMM_WORLD.
    
    ### Parameters:
    :param directive_list: Directive list to slice into chunks.
    :param mpi_size: Size of the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).

    ### Returns:
    :return: List containing desired number of chunks.
    """
    # Initialize empty list that will be returned to main.py
    root = list()

    # Returned sliced list using numpy
    tmp_list = np.array_split(directive_list, mpi_size-1)
    for array in tmp_list:
        root.append(array.tolist())

    return root


def delegate(communicator, comm_size: int, sliced_directives: List) -> List[int]:
    """
    Send task list to every available worker node in the MPI.COMM_WORLD.
    Task list is sent using sliced directive list.
    
    ### Parameters:
    :param communicator: Communicator variable used to communicate with nodes in the 
    MPI.COMM_WORLD (typically comm = MPI.COMM_WORLD).
    :param comm_size: Size of the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).
    :param sliced_directives: Sliced directive list containing task to send to the worker nodes.

    ### Returns:
    :return: List containing the rank of each worker node in the MPI.COMM_WORLD.
    """
    # Define node rank in order to send messages out
    node_rank = [i+1 for i in range(comm_size-1)]

    # Send slices out to the workers
    node_iter = 0
    for slice in sliced_directives:
        communicator.send(slice, dest=node_rank[node_iter], tag=node_rank[node_iter])
        node_iter += 1

    # Send node_rank back to main.py so it can track task completion
    return node_rank
