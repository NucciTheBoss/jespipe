import numpy as np


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
        parameters = directive[3]

        # Grab manip list and iterate over each manipulation
        manip_type_list = directive[4]
        for manip_type in manip_type_list:
            # Grab manipulation type: eg. xgboost, pca, randomforest, candlestick.
            manip_type_name = manip_type[0]

            # Grab list of manip names and parameters
            manip_list = manip_type[1]
            for manip in manip_list:
                # Create directive tuple and add to root list
                root.append((dataset_name, dataset_path, model_name, parameters, manip_type_name, manip[0], manip[1]))

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
        for attack_type in attack_type_list:
            # Grab attack type: eg. CW_inf, BIM, FGSM
            attack_type_name = attack_type[0]

            # Grab list of attack names and parameters
            attack_list = attack_type[1]
            for attack in attack_list:
                # Create directive tuple and add to root list
                root.append((dataset_name, dataset_path, attack_type_name, attack[0], attack[1]))


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


def delegate():
    """Delegate out a sliced directive list to how many workers are available.
    
    Keyword arguments:"""
    pass
