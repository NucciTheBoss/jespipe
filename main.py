from utils.workeradmin.skip import skip_attack
from mpi4py import MPI
import json
import sys
import re
import os
import time
import logging


# Global values to be shared across all nodes
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
CONFIG_FILE = ".config.json"
TIME_FMT = "%d-%m-%Y:%I:%M:%S-%p"
TIME = time.localtime(); TIME = time.strftime(TIME_FMT, TIME)


if rank == 0:
    # Staging: neccesary preprocessing before beginning the execution of the pipeline
    # Imports only necessary for manager node
    from utils.workeradmin import greenlight as gl
    from utils.workeradmin import skip
    from utils.macro import xml2dict as x2d
    from utils.macro.unwrap import unwrap_train, unwrap_attack
    from utils.workerops import scattershot as scat

    # Read in config to get default configurations file
    fin = open(CONFIG_FILE, "rt"); config = fin.read(); fin.close()
    config = json.loads(config)

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
    try:
        job_control = x2d.xml2dict(macro_file, config)

    except:
        gl.killmsg(comm, size, True)
        raise RuntimeError("A fatal was encountered while parsing the XML file.")

    # Split job control dictionary into its three parts: train, attack, cleanup
    train_control = job_control["train"] if "train" in job_control else None
    attack_control = job_control["attack"] if "attack" in job_control else None
    clean_control = job_control["clean"] if "clean" in job_control else None

    # Begin execution the stages for the pipeline. Inform workers they are ready to start!
    gl.killmsg(comm, size, False)

    # Train: launch training stage of the pipeline
    if train_control is not None:
        # Broadcast out to workers that we are now operating on the training stage
        skip.skip_train(comm, size, False)

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

        # Create directory for nodes to log their status if not exist
        os.makedirs("data/.logs", exist_ok=True)

        # Send greenlight to workers and then follow up with tasks
        node_rank = scat.delegate(comm, size, sliced_directive_list)

        # Block until manager hears back from all workers
        node_status = list()
        for node in node_rank:
            node_status.append(comm.recv(source=node, tag=node))

        print("Training is done!")
        print(node_status)

    else:
        # Broadcast out to workers that manager is skipping the training stage
        skip.skip_train(comm, size, True)

    # Attack: launch attack stage of the pipeline
    if attack_control is not None:
        # Broadcast out to workers that we are now operating on the attack stage
        skip.skip_attack(comm, size, False)

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
                raise(FileNotFoundError("Model(s) not found. Please verify that models are stored in data/{}/models.".format(macro[0])))

            # Once all checks are good, create directory for storing adversarial examples
            os.makedirs("data/" + macro[0] + "/adver_examples", exist_ok=True)

            # Create directory for nodes to log their status if not exist
            os.makedirs("data/.logs", exist_ok=True)

    else:
        skip.skip_attack(comm, size, True)

    # Clean: launch cleaning stage of the pipeline
    if clean_control is not None:
        # TODO: Create method for generating clean_macro_list
        pass

    else:
        # TODO: Implement broadcast for cleaning stage
        pass

    print("All done!")

elif rank == 1:
    greenlight = comm.recv(source=0, tag=1)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-1", exist_ok=True)
    logger = logging.getLogger("worker-1-logger")
    f_handler = logging.FileHandler("data/.logs/worker-1/{}.log".format(TIME))
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=1)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=1)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=1)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=1)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 2:
    greenlight = comm.recv(source=0, tag=2)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-2", exist_ok=True)
    logger = logging.getLogger("worker-2-logger")
    f_handler = logging.FileHandler("data/.logs/worker-2/{}.log".format(TIME))
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=2)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=2)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=2)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=2)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 3:
    greenlight = comm.recv(source=0, tag=3)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-3", exist_ok=True)
    logger = logging.getLogger("worker-3-logger")
    f_handler = logging.FileHandler("data/.logs/worker-3/{}.log".format(TIME), mode="w")
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=3)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=3)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=3)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=3)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 4:
    greenlight = comm.recv(source=0, tag=4)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-4", exist_ok=True)
    logger = logging.getLogger("worker-4-logger")
    f_handler = logging.FileHandler("data/.logs/worker-4/{}.log".format(TIME), mode="w")
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=4)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=4)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=4)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

        # Send message to manager acknowledging the skipping of the training stage
        comm.send(1, dest=0, tag=2)

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=4)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 5:
    greenlight = comm.recv(source=0, tag=5)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-5", exist_ok=True)
    logger = logging.getLogger("worker-5-logger")
    f_handler = logging.FileHandler("data/.logs/worker-5/{}.log".format(TIME), mode="w")
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=5)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=5)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=5)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=5)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 6:
    greenlight = comm.recv(source=0, tag=6)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-6", exist_ok=True)
    logger = logging.getLogger("worker-6-logger")
    f_handler = logging.FileHandler("data/.logs/worker-6/{}.log".format(TIME), mode="w")
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))
    
    # TRAINING STAGE
    skip_stage_training = comm.recv(source=0, tag=6)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=6)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=6)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=6)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes

elif rank == 7:
    greenlight = comm.recv(source=0, tag=7)
    if greenlight != 1:
        exit(127)

    # After getting greenlight, create logger for node
    os.makedirs("data/.logs/worker-7", exist_ok=True)
    logger = logging.getLogger("worker-7-logger")
    f_handler = logging.FileHandler("data/.logs/worker-7/{}.log".format(TIME), mode="w")
    logger.addHandler(f_handler)
    logger.warning("INFO: Received greenlight message {} from manager node. Begin execution.".format(greenlight))

    # TRAINING warning
    skip_stage_training = comm.recv(source=0, tag=7)

    if skip_stage_training != 1:
        logger.warning("INFO: Beginning execution of model training stage.")

        # Receive task from manager
        task_list = comm.recv(source=0, tag=7)
        logger.warning("Received task list {} from manager.".format(task_list))
        comm.send(1, dest=0, tag=7)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=7)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    # TODO: Write cleaning stage implementation for worker nodes
