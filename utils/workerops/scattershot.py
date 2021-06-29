import numpy as np
from ..managerops import getpaths as gp


def generate_train(macro_list):
    """Generate scatterable training directive list.
    
    Keyword arguments:
    macro_list -- training tuple list to be convertered into a scatterable list."""
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
                    root.append((dataset_name, dataset_path, model_name, algorithm, parameters, plugin, manip_type_name, manip[0], manip[1]))

        else:
            root.append((dataset_name, dataset_path, model_name, algorithm, parameters, plugin, None, None, None))

    return root


def generate_attack(macro_list):
    """Generate scatterable attack directive list.
    
    Keyword arguments:
    macro_list -- attack tuple list to be convertered into a scatterable list."""
    # Initialize root list that will be returned to main.py
    root = list()

    # Loop over macro list to create scatterable directive list
    for directive in macro_list:
        # Pull dataset name and dataset path
        dataset_name = directive[0]
        dataset_path = directive[1]

        # Grab attack list and iterate over each attack
        attack_type_list = directive[2]

        if attack_type_list is not None:
            for attack_type in attack_type_list:
                # Grab attack type: eg. CW_inf, BIM, FGSM
                attack_type_name = attack_type[0]

                # Grab list of attack names and parameters
                attack_list = attack_type[1]
                for attack in attack_list:
                    # Create directive tuple and add to root list
                    root.append((dataset_name, dataset_path, attack_type_name, attack[0], attack[1]))

        else:
             root.append((dataset_name, dataset_path, None, None, None))

    return root


def generate_clean(plot_dict, save_path, data_path):
    """Generate scatterable clean directive list.
    
    Keyword arguments:
    plot_dict -- dictionary containing user-defined macros for clean stage.
    save_path -- location to save user-generated plots.
    data_path -- location of trained models with stored data."""
    root = list()

    for key in plot_dict:
        plugin_path = plot_dict[key]["plugin"]
        tag_list = plot_dict[key]["tags"]

        root_paths = gp.getdirs(data_path)
        for i in range(0, len(tag_list)):
            for path in root_paths:
                if tag_list[i] in path:
                    tag_list[i] = path
                    break

        root.append((plugin_path, tag_list, key, save_path))

    return root


def slice(directive_list, mpi_size):
    """Slice up a directive list up based on how many available workers there are.
    
    Keyword arguments:
    directive_list -- the directive list to slice into chunks.
    mpi_size -- the size of the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size())."""
    # Initialize empty list that will be returned to main.py
    root = list()

    # Returned sliced list using numpy
    tmp_list = np.array_split(directive_list, mpi_size-1)
    for array in tmp_list:
        root.append(array.tolist())

    return root


def delegate(communicator, comm_size, sliced_directives):
    """Delegate out a sliced directive list to how many workers are available.
    
    Keyword arguments:
    communicator -- comm variable used to communicate with nodes (typically comm = MPI.COMM_WORLD).
    comm_size -- how many nodes exist in the MPI.COMM_WORLD (typically MPI.COMM_WORLD.Get_size()).
    sliced_directives -- sliced directive list to send to workers."""
    # Define node rank in order to send messages out
    node_rank = [i+1 for i in range(comm_size-1)]

    # Send slices out to the workers
    node_iter = 0
    for slice in sliced_directives:
        communicator.send(slice, dest=node_rank[node_iter], tag=node_rank[node_iter])
        node_iter += 1

    # Send node_rank back to main.py so it can track task completion
    return node_rank
