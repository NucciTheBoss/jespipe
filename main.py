from mpi4py import MPI
import json
import sys
import re


# Global values to be shared across all nodes
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
CONFIG_FILE = ".config.json"


"""Bare bones skeletal structure of what the MPI application will
   look like. Will flesh out more as more pieces of the pipeline
   are completed. **Will use 8 nodes on Ursula but I don't want
   to kill my computer during my initial development**"""
if rank == 0:
    # Staging: neccesary preprocessing before beginning the execution of the pipeline
    # Imports only necessary for manager node
    from utils.workerops import greenlight as gl
    from utils.macro import xml2dict as x2d

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
                               CW_INF, BIM, FGSM)

    # Split job control dictionary into its three parts: train, attack, cleanup
    train_control = job_control["train"] if "train" in job_control else None
    attack_control = job_control["attack"] if "attack" in job_control else None
    clean_control = job_control["clean"] if "clean" in job_control else None

    # Begin execution the stages for the pipeline
    # Train: launch training stage of the pipeline
    if train_control is not None:
        pass

    # Attack: launch attack stage of the pipeline
    if attack_control is not None:
        pass

    # Clean: launch cleaning stage of the pipeline
    if clean_control is not None:
        pass 

    gl.killmsg(comm, size, False)
    print("All done!")

elif rank == 1:
    greenlight = comm.recv(source=0, tag=1)
    if greenlight != 1:
        exit(127)

    else:
        print("Worker #1 is ready to go!")

elif rank == 2:
    greenlight = comm.recv(source=0, tag=2)
    if greenlight != 1:
        exit(127)

    else:
        print("Worker #2 is ready to go!")

elif rank == 3:
    greenlight = comm.recv(source=0, tag=3)
    if greenlight != 1:
        exit(127)

    else:
        print("Worker #3 is ready to go!")

elif rank == 4:
    pass

elif rank == 5:
    pass

elif rank == 6:
    pass

elif rank == 7:
    pass
