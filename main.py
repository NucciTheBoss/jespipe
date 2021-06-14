from mpi4py import MPI
import json
import sys
import re
import os


# Global values to be shared across all nodes
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
CONFIG_FILE = ".config.json"


if rank == 0:
    # Staging: neccesary preprocessing before beginning the execution of the pipeline
    # Imports only necessary for manager node
    from utils.workeradmin import greenlight as gl
    from utils.macro import xml2dict as x2d
    from utils.macro.unwrap import unwrap_train, unwrap_attack
    from utils.workerops import scattershot as scat

    # Read in config to get default configurations file
    fin = open(CONFIG_FILE, "rt"); config = fin.read(); fin.close()
    config = json.loads(config)

    # Parse in data from the config file
    RNN_PARAM = config["algorithms"]["RNN"]
    XGBOOST = config["datamanips"]["xgboost"]
    RANDOM_FOREST = config["datamanips"]["randomforest"]
    PCA = config["datamanips"]["pca"]
    CANDLESTICK = config["datamanips"]["candlestick"]
    CW_INF = config["attacks"]["CW_inf"]
    BIM = config["attacks"]["BIM"]
    FGSM = config["attacks"]["FGSM"]

    # Read in marco XML file
    if len(sys.argv) != 2:
        gl.killmsg(comm, size, True)
        raise ValueError("No macro XML file specified before launching pipeline.")

    macro_file = sys.argv[1]

    # Perform checks to verify that XML file is in good format
    if re.search("\Wxml", macro_file):
        pass

    else:
        gl.killmsg(comm, size, True)
        raise(ValueError("Specified macro file not in XML format."))
    
    # Convert macro XML file to dictionary to begin staging for the pipeline
    job_control = x2d.xml2dict(macro_file, XGBOOST, RANDOM_FOREST, PCA, CANDLESTICK,
                               CW_INF, BIM, FGSM, RNN_PARAM)

    # Split job control dictionary into its three parts: train, attack, cleanup
    train_control = job_control["train"] if "train" in job_control else None
    attack_control = job_control["attack"] if "attack" in job_control else None
    clean_control = job_control["clean"] if "clean" in job_control else None

    # Begin execution the stages for the pipeline
    # Train: launch training stage of the pipeline
    if train_control is not None:
        train_macro_list = unwrap_train(train_control)

        # Loop through train_macro_list, creating directories for
        # storing models for each dataset, as well as verifying
        # that each specified dataset does exists
        for macro in train_macro_list:
            # Check that dataset exists. If not, raise file not found error
            if os.path.isfile(macro[1]) is False:
                gl.killmsg(comm, size, True)
                raise(FileNotFoundError("Specified dataset is not found. Please verify that you are using the correct file path."))

            # If dataset file is verified to exist, create necessary directories
            # Create data/$dataset_name/models <- where trained models are stored
            os.makedirs("data/" + macro[0] + "/models", exist_ok=True)

        # Create directives for the worker nodes
        train_directive_list = scat.generate_train(train_macro_list)
        sliced_directive_list = scat.slice(train_directive_list, size)

        # Send greenlight to workers and then follow up with tasks
        gl.killmsg(comm, size, False)
        node_rank = scat.delegate(comm, size, sliced_directive_list)

    # Attack: launch attack stage of the pipeline
    if attack_control is not None:
        attack_macro_list = unwrap_attack(attack_control)

        # Loop through attack_macro_list, creating directories for
        # storing adversarial examples for each dataset, as well as verifying
        # that each specified dataset does exists. Also, checks to see if the
        # models exist as well
        for macro in attack_macro_list:
            # Check that dataset exists. If not, raise file not found error
            if os.path.isfile(macro[1]) is False:
                gl.killmsg(comm, size, True)
                raise(FileNotFoundError("Specified dataset is not found. Please verify that you are using the correct file path."))

            # Check if models exist. If not, raise file not found error
            if os.path.exists("data/" + macro[0] + "/models") is False:
                gl.killmsg(comm, size, True)
                raise(FileNotFoundError("Model(s) not found. Please verify that models are stored in data/{}/models.".format(macro[0])))

            # Once all checks are good, create directory for storing adversarial examples
            os.makedirs("data/" + macro[0] + "/adver_examples")

    # Clean: launch cleaning stage of the pipeline
    if clean_control is not None:
        # TODO: Create method for generating clean_macro_list
        pass

    gl.killmsg(comm, size, False)
    print("All done!")

elif rank == 1:
    greenlight = comm.recv(source=0, tag=1)
    if greenlight != 1:
        exit(127)

    print("Worker #1 is ready to go!")
    task_list = comm.recv(source=0, tag=1)
    print("My task is", task_list)

elif rank == 2:
    greenlight = comm.recv(source=0, tag=2)
    if greenlight != 1:
        exit(127)

    print("Worker #2 is ready to go!")
    task_list = comm.recv(source=0, tag=2)
    print("My task is", task_list)

elif rank == 3:
    greenlight = comm.recv(source=0, tag=3)
    if greenlight != 1:
        exit(127)

    print("Worker #3 is ready to go!")
    task_list = comm.recv(source=0, tag=3)
    print("My task is", task_list)

elif rank == 4:
    greenlight = comm.recv(source=0, tag=4)
    if greenlight != 1:
        exit(127)

    print("Worker #4 is ready to go!")
    task_list = comm.recv(source=0, tag=4)
    print("My task is", task_list)

elif rank == 5:
    greenlight = comm.recv(source=0, tag=5)
    if greenlight != 1:
        exit(127)

    print("Worker #5 is ready to go!")
    task_list = comm.recv(source=0, tag=5)
    print("My task is", task_list)

elif rank == 6:
    greenlight = comm.recv(source=0, tag=6)
    if greenlight != 1:
        exit(127)

    print("Worker #6 is ready to go!")
    task_list = comm.recv(source=0, tag=6)
    print("My task is", task_list)

elif rank == 7:
    greenlight = comm.recv(source=0, tag=7)
    if greenlight != 1:
        exit(127)

    print("Worker #7 is ready to go!")
    task_list = comm.recv(source=0, tag=7)
    print("My task is", task_list)
