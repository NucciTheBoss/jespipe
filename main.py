from mpi4py import MPI
import json
import sys
import warnings
import re
import os
from tqdm import tqdm
import shutil
import time
import logging
import subprocess
from utils.workerops.preprocessing import preprocessing
from utils.workerops.recombine import recombine
from utils.workerops.paramfactory import paramfactory


# Deactivate warnings from Python unless requested at command line
if not sys.warnoptions:
    warnings.simplefilter("ignore")


# Global values to be shared across all nodes
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
ROOT_PATH = os.getcwd()
CONFIG_FILE = ".config.json"
TIME_FMT = "%d-%m-%Y:%I:%M:%S-%p"
TIME = time.localtime(); TIME = time.strftime(TIME_FMT, TIME)
PYTHON_PATH = subprocess.getoutput("which python")


if rank == 0:
    # Staging: neccesary preprocessing before beginning the execution of the pipeline
    # Imports only necessary for manager node
    from colorama import Fore, Style, init
    from utils.workeradmin import greenlight as gl
    from utils.workeradmin import skip
    from utils.macro import xml2dict as x2d
    from utils.macro.unwrap import unwrap_train, unwrap_attack
    from utils.workerops import scattershot as sst
    from utils.managerops.compress import Compression

    
    # Initialize colorama and define lambda functions
    init(autoreset=True)
    print_good = lambda x: print(Fore.GREEN + x)
    print_info = lambda x: print(Fore.BLUE + x)
    print_dim_info = lambda x: print(Fore.BLUE + Style.DIM + x)
    print_bad = lambda x: print(Fore.RED + x)
    print_status = lambda x: print(Fore.YELLOW + x)


    print_status("Launching preprocessing stage.")
    # Check if we are working in the same directory as main.py.
    # If not, throw error as that will impact the pipelines ability to run.
    print_info("Checking if running out of same directory as {}".format(sys.argv[0]))
    cwd_contents = [f for f in os.listdir(".") if os.path.isfile(f)]
    if sys.argv[0] not in cwd_contents:
        gl.killmsg(comm, size, True)
        raise OSError(Fore.RED + "Not in same directory as {}. Please change current working directory to where {} is located.".format(sys.argv[0], sys.argv[0]))

    # Read in config to get default configurations file
    print_info("Loading configuration file {}.".format(CONFIG_FILE))
    try:
        fin = open(CONFIG_FILE, "rt"); config = fin.read(); fin.close()
        config = json.loads(config)
    
    except:
        gl.killmsg(comm, size, True)
        raise OSError(Fore.RED + "Cannot find or read {}. Please veify that {} exists and is readable.".format(CONFIG_FILE, CONFIG_FILE))

    # Read in marco XML file
    print_info("Checking if macro XML file passed at command line.")
    if len(sys.argv) != 2:
        gl.killmsg(comm, size, True)
        raise ValueError(Fore.RED + "No macro XML file specified before launching pipeline.")

    print_info("Loading macro XML file {}.".format(sys.argv[1]))
    macro_file = sys.argv[1]

    # Perform checks to verify that XML file is in good format
    if re.search("\Wxml", macro_file):
        pass

    else:
        gl.killmsg(comm, size, True)
        raise ValueError(Fore.RED + "Specified macro file {} not in XML format.".format(macro_file))
    
    # Convert macro XML file to dictionary to begin staging for the pipeline
    print_info("Parsing macro file {} in control dictionaries.".format(macro_file))
    try:
        job_control = x2d.xml2dict(macro_file, config)

    except:
        gl.killmsg(comm, size, True)
        raise RuntimeError(Fore.RED + "A fatal was encountered while parsing the XML file.")

    # Split job control dictionary into its three parts: train, attack, cleanup
    train_control = job_control["train"] if "train" in job_control else None
    attack_control = job_control["attack"] if "attack" in job_control else None
    clean_control = job_control["clean"] if "clean" in job_control else None

    print_info("Creating data directories.")
    # Create directory for nodes to log their status if not exist
    os.makedirs("data/.logs", exist_ok=True)

    # Create directory for processes to write temporary files to
    os.makedirs("data/.tmp", exist_ok=True)

    # Begin execution the stages for the pipeline. Inform workers they are ready to start!
    gl.killmsg(comm, size, False)
    print_good("Preprocessing stage complete!")

    # Train: launch training stage of the pipeline
    if train_control is not None:
        print_status("Launching training stage.")
        # Broadcast out to workers that we are now operating on the training stage
        skip.skip_train(comm, size, False)

        print_info("Unwrapping train control dictionary.")
        train_macro_list = unwrap_train(train_control)

        print_info("Converting relative file paths to absolute paths.")
        # Loop through train_macro_list and convert relative paths to absolute paths
        for i in range(0, len(train_macro_list)):
            # Convert to list in order to change elements
            dataset = list(train_macro_list[i])

            # Check if path to dataset is absolute
            if os.path.isabs(dataset[1]) is False:
                dataset[1] = os.path.abspath(dataset[1])

            # Check if path to plugin is absolute
            if os.path.isabs(dataset[5]) is False:
                dataset[5] = os.path.abspath(dataset[5])

            # Convert back to tuple
            train_macro_list[i] = tuple(dataset)

        print_info("Checking that file paths to dataset(s) and plugin(s) are valid.")
        # Loop through train_macro_list, creating directories for
        # storing models for each dataset, as well as verifying
        # that each specified dataset does exists
        for macro in train_macro_list:
            # Check that the dataset and plugin exist. If not, raise file not found error
            if os.path.isfile(macro[1]) is False or os.path.isfile(macro[5]) is False:
                gl.killmsg(comm, size, True)

                if os.path.isfile(macro[1]) is False:
                    raise FileNotFoundError(Fore.RED + "The dataset {} is not found. Please verify that you are using the correct file path.".format(macro[1]))

                else:
                    raise FileNotFoundError(Fore.RED + "The plugin {} is not found. Please verify that you are using the correct file path.".format(macro[5]))

            # If the dataset and plugin are verified to exist, create necessary directories
            # Create data/$dataset_name/models <- where trained models are stored
            os.makedirs("data/" + macro[0] + "/models", exist_ok=True)

        # Create directives for the worker nodes
        print_info("Generating directive list for worker nodes.")
        train_directive_list = sst.generate_train(train_macro_list)
        sliced_directive_list = sst.slice(train_directive_list, size)

        # Broadcast that everything is good to go for the training stage
        gl.killmsg(comm, size, False)

        print_info("Sending tasks to workers.")
        # Send greenlight to workers and then follow up with tasks
        node_rank = sst.delegate(comm, size, sliced_directive_list)

        print_info("Blocking until all workers complete training tasks.")
        print_dim_info("Warning: This procedure may take a few minutes to a couple hours to complete depending " +
            "on the complexity of your data, architecture of your model(s), number of models to train, etc.")
        # Block until manager hears back from all workers
        node_status = list()
        for node in tqdm(node_rank, desc="Worker node task completion progress"):
            node_status.append(comm.recv(source=node, tag=node))

        print_good("Training stage complete!")

    else:
        print_status("Skipping training stage.")
        # Broadcast out to workers that manager is skipping the training stage
        skip.skip_train(comm, size, True)

    # Attack: launch attack stage of the pipeline
    if attack_control is not None:
        print_status("Launching attack stage.")
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
                raise FileNotFoundError(Fore.RED + "Specified dataset is not found. Please verify that you are using the correct file path.")

            # Check if models exist. If not, raise file not found error
            if os.path.exists("data/" + macro[0] + "/models") is False:
                raise FileNotFoundError(Fore.RED + "Model(s) not found. Please verify that models are stored in data/{}/models.".format(macro[0]))

            # Once all checks are good, create directory for storing adversarial examples
            os.makedirs("data/" + macro[0] + "/adver_examples", exist_ok=True)

        print_good("Attack stage complete!")

    else:
        print_status("Skipping attack stage.")
        # Broadcast out to workers that manager is skipping the attack stage
        skip.skip_attack(comm, size, True)

    # Clean: launch cleaning stage of the pipeline
    if clean_control is not None:
        print_status("Launching cleaning stage.")
        # Broadcast out to workers that we are now operating on the cleaning stage
        skip.skip_clean(comm, size, False)

        if clean_control["plot"] is not None:
            print_info("Checking that file paths to plugin(s) are valid.")
            # Loop through plot keys and convert relative paths to absolute paths
            for key in clean_control["plot"]:
                if os.path.isabs(clean_control["plot"][key]["plugin"]) is False:
                    clean_control["plot"][key]["plugin"] = os.path.abspath(clean_control["plot"][key]["plugin"])

                # Check if path to the plugin is valid
                if os.path.isfile(clean_control["plot"][key]["plugin"]) is False:
                    raise FileNotFoundError(Fore.RED + "The plugin {} is not found. Please verify that you are using the correct file path.".format(
                        clean_control["plot"][key]["plugin"]))

            print_info("Generating directive list for worker nodes.")
            # Generate and slice directive list that will be sent out to the workers
            clean_directive_list = sst.generate_clean(clean_control["plot"], ROOT_PATH + "/data/plots", ROOT_PATH + "/data")
            sliced_directive_list = sst.slice(clean_directive_list, size)
            
            # Send greenlight to workers
            gl.killmsg(comm, size, False)

            print_info("Sending tasks to workers.")
            # Delegate tasks out to the available workers in the COMM_WORLD
            node_rank = sst.delegate(comm, size, sliced_directive_list)

            print_info("Blocking until all workers complete plotting tasks.")
            print_dim_info("Warning: This procedure may take some time to complete depending on how many plots are being generated, " +
                "complexity of the data being anaylzed, format of the plot, etc.")
            # Block until hearing back from all the worker nodes
            node_status = list()
            for node in tqdm(node_rank, desc="Worker node task completion progress"):
                node_status.append(comm.recv(source=node, tag=node))

        else:
            gl.killmsg(comm, size, True)

        if clean_control["clean_tmp"] == 1:
            print_info("Deleting data/.tmp directory.")
            shutil.rmtree("data/.tmp", ignore_errors=True)

        if clean_control["compress"] is not None:
            for key in clean_control["compress"]:
                print_info("Renaming data directory to {} and compressing into format {}.".format(key, clean_control["compress"][key]["format"]))
                # Create compressor that will be used to shrink the data directory
                shutil.move("data", key)
                compressor = Compression(key, key)

                # Create archive based on user-specified compression algorithm
                if clean_control["compress"][key]["format"] == "gzip":
                    compressor.togzip()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.tar.gz".format(key), "{}/{}.tar.gz".format(clean_control["compress"][key]["path"], key))
                
                elif clean_control["compress"][key]["format"] == "bz2":
                    compressor.tobzip()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.tar.bz2".format(key), "{}/{}.tar.bz2".format(clean_control["compress"][key]["path"], key))

                elif clean_control["compress"][key]["format"] == "zip":
                    compressor.tozip()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.zip".format(key), "{}/{}.zip".format(clean_control["compress"][key]["path"], key))

                elif clean_control["compress"][key]["format"] == "xz":
                    compressor.toxz()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.tar.xz".format(key), "{}/{}.tar.xz".format(clean_control["compress"][key]["path"], key))

                elif clean_control["compress"][key]["format"] == "tar":
                    compressor.totar()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.tar".format(key), "{}/{}.tar".format(clean_control["compress"][key]["path"], key))

                else:
                    # Catch all for if user passes invalid compression algorithm
                    compressor.togzip()
                    shutil.rmtree(key, ignore_errors=True)

                    if os.path.exists(clean_control["compress"][key]["path"]):
                        shutil.move("{}.tar.gz".format(key), "{}/{}.tar.gz".format(clean_control["compress"][key]["path"], key))

        print_good("Cleaning stage complete!")

    else:
        print_status("Skipping cleaning stage.")
        # Broadcast out to workers that manager is skipping the cleaning stage
        skip.skip_clean(comm, size, True)

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=1)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=1)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))
        
        # Check if task list sent is empty. If so, return message to the manager
        if task_list != []:
            # Loop through each of the tasks and perform necessary data manipulations
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-1/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=1)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=1)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=1)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=1)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=1)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=1)

    else:
        logger.warning("WARNING: Skipping clean stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=2)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=2)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-2/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=2)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=2)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=2)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=2)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=2)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=2)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=3)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=3)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-3/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=3)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=3)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=3)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=3)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=3)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=3)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=4)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=4)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-4/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Set sys.stdout back to its original output method
                fout.close()

            comm.send(1, dest=0, tag=4)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=4)

    else:
        logger.warning("WARNING: Skipping training stage of pipeline.")

    # ATTACK STAGE
    skip_stage_attack = comm.recv(source=0, tag=4)

    if skip_stage_attack != 1:
        pass

    else:
        logger.warning("WARNING: Skipping attack stage of pipeline.")

    # CLEANING STAGE
    skip_clean_stage = comm.recv(source=0, tag=4)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=4)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=4)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=4)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=4)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=5)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=5)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-5/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=5)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=5)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=5)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=5)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=5)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=5)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=6)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=6)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-6/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=6)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=6)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=6)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=6)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=6)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=6)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")

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
        logger.warning("INFO: Waiting for greenlight to start training stage.")
        
        training_greenlight = comm.recv(source=0, tag=7)
        if training_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for training stage. Aborting execution.".format(training_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model training stage.".format(training_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=7)
        logger.warning("Received task list {} from manager.".format(task_list))

        if task_list != []:
            for task in task_list:
                logger.warning("INFO: Beginning training of model {} using directive list {}.".format(task[2], task))

                # Perform data manipulation using user specified data manipulation
                feat, label = preprocessing(task[1], task[6], task[8])

                # Reassign task[6], task[7], and task[8] if they set to None
                task[6] = "default" if task[6] is None else task[6]
                task[7] = "default" if task[7] is None else task[7]
                task[8] = "default" if task[8] is None else task[8]

                recomb = recombine(feat, label, save=True, save_path="data/" + task[0] + "/maniped_data", manip_tag=task[7])

                # Created special directory for each individual manipulation
                save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                if os.path.exists(save_path):
                    shutil.rmtree(save_path, ignore_errors=True)

                else:
                    os.makedirs(save_path, exist_ok=True)

                # Create dictionary that will be passed to plugin
                param_dict = paramfactory(task[0], task[2], recomb, task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                # Spawn plugin execution and block until the training section of the plugin has completed
                logger.warning("INFO: Training model...")
                file_output = "data/.logs/worker-7/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                # Swap stdout to log file in order to prevent worker from writing out to the shell
                fout = open(file_output, "wt")

                try:
                    subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                # Close the appension for the log file
                fout.close()

            comm.send(1, dest=0, tag=7)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
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
    skip_clean_stage = comm.recv(source=0, tag=7)

    if skip_clean_stage != 1:
        logger.warning("INFO: Waiting for greenlight to start cleaning stage.")

        cleaning_greenlight = comm.recv(source=0, tag=7)
        if cleaning_greenlight != 1:
            # 0 message means worker is not needed any more
            logger.warning("WARNING: Received greenlight message {} for cleaning stage. Aborting execution.".format(cleaning_greenlight))
            exit(0)

        logger.warning("INFO: Received greenlight {}. Beginning execution of cleaning stage.".format(cleaning_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=7)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            comm.send(1, dest=0, tag=7)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=7)

    else:
        logger.warning("WARNING: Skipping cleaning stage of pipeline.")
