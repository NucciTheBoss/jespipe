import copy
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from decimal import Decimal

import joblib
from mpi4py import MPI
from tqdm import tqdm

import utils.filesystem.getpaths as gp
from utils.workerops.paramfactory import (attack_factory, attack_train_factory, clean_factory,
                                          manip_factory, train_factory)


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
    # PREPROCESSING: neccesary preprocessing before beginning the execution of the pipeline
    # Imports only necessary for manager node
    from colorama import Fore, Style, init

    from utils.macro import xml2dict as x2d
    from utils.macro.unwrap import unwrap_attack, unwrap_train
    from utils.managerops.compress import Compression
    from utils.workeradmin import greenlight as gl
    from utils.workeradmin import skip
    from utils.workerops import scattershot as sst


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

    # TRAIN: launch training stage of the pipeline
    if train_control is not None:
        print_status("Launching training stage.")

        # Broadcast out to workers that we are now operating on the training stage
        skip.skip_train(comm, size, False)

        print_info("Unwrapping train control dictionary.")
        train_macro_list = unwrap_train(train_control)

        print_info("Converting relative file paths to absolute paths.")
        print_info("Checking that file paths to dataset(s) and plugin(s) are valid.")
        # Loop through train_macro_list:
        # - Convert relative paths to absolute paths
        # - Verify path to dataset and plugin exists
        # - Create directory for each dataset
        for i in range(0, len(train_macro_list)):
            # Convert to list in order to change elements
            macro = list(train_macro_list[i])

            # Check if path to dataset is absolute
            if os.path.isabs(macro[1]) is False:
                macro[1] = os.path.abspath(macro[1])

            # Check if dataset exists
            if os.path.isfile(macro[1]) is False:
                gl.killmsg(comm, size, True)
                raise FileNotFoundError(Fore.RED + "The dataset {} is not found. Please verify that you are using the correct file path.".format(macro[1])) 

            # Check if path to model plugin is absolute
            if os.path.isabs(macro[5]) is False:
                macro[5] = os.path.abspath(macro[5])

            # Check if model plugin exists
            if os.path.isfile(macro[5]) is False:
                gl.killmsg(comm, size, True)
                raise FileNotFoundError(Fore.RED + "The plugin {} is not found. Please verify that you are using the correct file path.".format(macro[5]))

            # Loop through manipulations to check if data manipulation plugins exist
            for manip in macro[6]:
                for manip_tag in manip[1]:
                    if os.path.isabs(manip_tag[1]["plugin"]) is False:
                        manip_tag[1]["plugin"] = os.path.abspath(manip_tag[1]["plugin"])

                    if os.path.isfile(manip_tag[1]["plugin"]) is False:
                        gl.killmsg(comm, size, True)
                        raise FileNotFoundError(Fore.RED + "The plugin {} is not found. Please verify that you are using the correct file path.".format(manip_tag[1]["plugin"]))

            # Convert back to tuple
            train_macro_list[i] = tuple(macro)

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

    # ATTACK: launch attack stage of the pipeline
    if attack_control is not None:
        print_status("Launching attack stage.")

        # Broadcast out to workers that we are now operating on the attack stage
        skip.skip_attack(comm, size, False)

        attack_macro_list = unwrap_attack(attack_control)

        # Loop through attack_macro_list:
        # - Convert relative paths to absolute paths
        # - Verify that trained models exist (use autodetection)
        # - Verify that mentioned dataset and plugins exist
        print_info("Converting relative file paths to absolute paths.")
        print_info("Checking that file paths to dataset(s) and plugin(s) are valid.")
        for i in range(0, len(attack_macro_list)):
            # Convert to list in order to change elements
            macro = list(attack_macro_list[i])

            # Convert dataset path to absolute path
            if os.path.isabs(macro[1]) is False:
                macro[1] = os.path.abspath(macro[1])

            # Check that dataset exists
            if os.path.isfile(macro[1]) is False:
                gl.killmsg(comm, size, True)
                raise FileNotFoundError(Fore.RED + "Specified dataset {} is not found. Please verify that you are using the correct file path.".format(macro[1]))

            # Convert plugin and model plugin to absolute paths and check that they exist
            for attack in macro[2]:
                if attack[1] is not None:
                    attack_params = [k for k in attack[1]]
                    for param in attack_params:
                        if os.path.isabs(attack[1][param]["plugin"]) is False:
                            attack[1][param]["plugin"] = os.path.abspath(attack[1][param]["plugin"])

                        if os.path.isfile(attack[1][param]["plugin"]) is False:
                            gl.killmsg(comm, size, True)
                            raise FileNotFoundError(Fore.RED + "The plugin {} is not found. Please verify that you are using the correct file path.".format(attack[1][param]["plugin"]))

                        if os.path.isabs(attack[1][param]["model_plugin"]) is False:
                            attack[1][param]["model_plugin"] = os.path.abspath(attack[1][param]["model_plugin"])

                        if os.path.isfile(attack[1][param]["model_plugin"]) is False:
                            gl.killmsg(comm, size, True)
                            raise FileNotFoundError(Fore.RED + "The model plugin {} is not found. Please verify that you are using the correct file path.".format(attack[1][param]["model_plugin"]))

            # Check if models exist
            if os.path.exists("data/" + macro[0] + "/models") is False:
                gl.killmsg(comm, size, True)
                raise FileNotFoundError(Fore.RED + "Model(s) not found. Please verify that models are stored in data/{}/models.".format(macro[0]))

            # If models do exist, autodetect the .h5 files and add to macro list
            print_info("Auto-detecting models for dataset {}.".format(macro[0]))
            model_list = gp.getmodels(ROOT_PATH + "/data/" + macro[0] + "/models", format=".h5")
            macro.append(model_list)

            attack_macro_list[i] = tuple(macro)

            # Once all checks are good, create directory for storing adversarial examples
            os.makedirs("data/" + macro[0] + "/adver_examples", exist_ok=True)

        # Create directives for the worker nodes
        print_info("Generating adversarial generation directive list for worker nodes.")
        attack_directive_list = sst.generate_attack(attack_macro_list)
        
        # Loop through directive list and generate more directives based on the change step
        adver_example_directive_list = list()
        for directive in attack_directive_list:
            max_change = Decimal(str(directive[6]["max_change"]))
            min_change = Decimal(str(directive[6]["min_change"]))
            change_step = Decimal(str(directive[6]["change_step"]))

            # Construct perturbation steps list using max_change, min_change, and change_step
            tmp_list = list()
            while min_change <= max_change:
                tmp_list.append(min_change); min_change += change_step

            # Convert decimal values back to float values                
            change_values = [float(i) for i in tmp_list]

            # Expand directive list using change values
            for change in change_values:
                tmp_direct = copy.deepcopy(directive); tmp_direct = list(tmp_direct)
                del tmp_direct[6]["max_change"]; del tmp_direct[6]["min_change"]; del tmp_direct[6]["change_step"]
                tmp_direct[6].update({"change": change})
                adver_example_directive_list.append(tuple(tmp_direct))

        sliced_directive_list = sst.slice(adver_example_directive_list, size)
        
        # Broadcast that everything is good to go to the worker nodes
        gl.killmsg(comm, size, False)

        print_info("Sending adversarial example generation tasks to workers.")
        # Follow greenlight up with task list
        node_rank = sst.delegate(comm, size, sliced_directive_list)

        print_info("Blocking until all workers complete adversarial example generation tasks.")
        print_dim_info("Warning: This procedure may take a few minutes to a couple hours to complete depending " +
            "on the complexity of your attack, batch size of your attack, number of attacks, etc.")
        
        # Block until manager hears back from all workers
        node_status = list()
        for node in tqdm(node_rank, desc="Adversarial example generation task completion progress"):
            node_status.append(comm.recv(source=node, tag=node))

        print_info("Generating model evaluation directive list for worker nodes.")
        sliced_directive_list = sst.slice(attack_directive_list, size)
        print_info("Sending model evaluation directive list to worker nodes.")
        node_rank = sst.delegate(comm, size, sliced_directive_list)

        print_info("Blocking until all workers complete model evaluation tasks.")
        print_dim_info("Warning: This procedure may take a few minutes to a couple hours to complete depending " +
            "on the number of models, size of adversarial examples, number of adversarial examples, etc.")

        # Block until manager hears back from all workers
        node_status = list()
        for node in tqdm(node_rank, desc="Model evaluation task completion progress"):
            node_status.append(comm.recv(source=node, tag=node))

        print_good("Attack stage complete!")

    else:
        print_status("Skipping attack stage.")
        # Broadcast out to workers that manager is skipping the attack stage
        skip.skip_attack(comm, size, True)

    # CLEAN: launch cleaning stage of the pipeline
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

    print_good("Jespipe has completed!")

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[9], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-1/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=1)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=1)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-1/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=1)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=1)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=1)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-1/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=1)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=1)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-2/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=2)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=2)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-2/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=2)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=2)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=2)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-2/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=2)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=2)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-3/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=3)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=3)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-3/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=3)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=3)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=3)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-3/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=3)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=3)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-4/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review the above output for error diagnostics.".format(task[2]))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=4)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=4)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-4/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=4)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=4)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=4)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-4/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=4)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=4)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-5/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=5)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=5)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-5/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=5)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=5)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=5)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-5/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=5)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=5)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-6/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=6)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=6)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-6/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=6)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=6)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=6)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-6/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=6)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=6)

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

                # Check if task[6], task[7], task[8], or task[9] is None
                # If so, skip execution and tell user they need to mention something.
                if task[6] is None or task[7] is None or task[8] is None or task[9] is None:
                    logger.warning("ERROR: Skipping model {} because no manipulation was specified " +
                                    "Please use the tag <vanilla tag='default1' /> or something similiar " +
                                    "in your control file.")
                    pass

                else:
                    manip_save_path = ROOT_PATH + "/data/" + task[0] + "/maniped_data"
                    if os.path.exists(manip_save_path) is False:
                        os.makedirs(manip_save_path, exist_ok=True)

                    logger.warning("INFO: Using {} on dataset {} with parameters {}.".format(task[6], task[0], task[9]))

                    # Perform data manipulation using manipulation plugin
                    param_dict = manip_factory(task[1], task[7], task[9], manip_save_path, ROOT_PATH + "/data/.tmp", ROOT_PATH)
                    maniped_pickle = subprocess.getoutput("{} {} {} {}".format(PYTHON_PATH, task[8], "train", param_dict))

                    # Created special directory for each individual manipulation
                    save_path = ROOT_PATH + "/data/" + task[0] + "/models/" + task[7]
                    if os.path.exists(save_path):
                        shutil.rmtree(save_path, ignore_errors=True)

                    os.makedirs(save_path, exist_ok=True)

                    # Create dictionary that will be passed to the training plugin
                    param_dict = train_factory(task[0], task[2], joblib.load(maniped_pickle), task[4], task[8], save_path, task[6], task[7], ROOT_PATH)

                    # Spawn plugin execution and block until the training section of the plugin has completed
                    logger.warning("INFO: Training model...")
                    file_output = "data/.logs/worker-7/{}-{}-{}-{}.log".format(TIME, task[2], task[6], task[7])
                    logger.warning("INFO: Saving output of {} for model {} to logfile {}.".format(task[5], task[2], file_output))

                    # Open a file that the training plugin can use for stdout and stderr
                    fout = open(file_output, "wt")

                    try:
                        subprocess.run([PYTHON_PATH, task[5], "train", param_dict], stdout=fout, stderr=fout)

                    except subprocess.SubprocessError:
                        logger.warning("ERROR: Build for model {} failed. Please review logfile {} for error diagnostics.".format(task[2], file_output))

                    # Close the file the training plugin is using to log stdout and stderr
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
        logger.warning("INFO: Waiting for greenlight to start attack stage.")
        
        attack_greenlight = comm.recv(source=0, tag=7)
        if attack_greenlight != 1:
            logger.warning("ERROR: Received greenlight message {} for attack stage. Aborting execution.".format(attack_greenlight))
            exit(127)

        logger.warning("INFO: Received greenlight {}. Beginning execution of model attack stage.".format(attack_greenlight))

        # Receive task from manager
        task_list = comm.recv(source=0, tag=7)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Generate adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning adversarial attack on model {} with attack {}".format(task[7], task[2]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Get change value
                change = task[6]["change"]

                logger.warning("INFO: Generating adversial example with minimum change set to {}.".format(change))

                # Open file that the attack plugin can use as a log file
                file_output = "data/.logs/worker-7/{}-attack-{}-{}-{}-{}.log".format(TIME, task[2], task[3], model_name, change)
                logger.warning("INFO: Saving output of {} for attack {} to logfile {}.".format(task[3], task[2], file_output))
                fout = open(file_output, "wt")

                test_features = gp.gettestfeat(task[8], feature_file="test_features.pkl")
                attack_param = attack_factory(task[3], task[7], joblib.load(test_features), task[6], 
                                                ROOT_PATH + "/data/" + task[0] + "/adver_examples", ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[4], "attack", attack_param], stdout=fout, stderr=fout)
                
                except subprocess.SubprocessError:
                    logger.warning("ERROR: Attack on model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))

                # Close the attack plugin log file
                fout.close()

            comm.send(1, dest=0, tag=7)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=7)

        # Receive second task list from manager
        task_list = comm.recv(source=0, tag=7)
        logger.warning("INFO: Received task list {} from manager.".format(task_list))

        if task_list != []:
            # Evaluate model using adversarial examples
            for task in task_list:
                logger.warning("INFO: Beginning evaluation of model {} using adversarial examples.".format(task[7]))

                # Get model name
                model_name = task[7].split("/"); model_name = model_name[-1].split("."); model_name = model_name[0]

                # Open file that the training plugin can use as a log file during evaluation
                file_output = "data/.logs/worker-7/{}-eval-{}-{}-{}.log".format(TIME, task[2], task[3], model_name)
                logger.warning("INFO: Saving output of {} evaluation logfile {}.".format(task[7], file_output))
                fout = open(file_output, "wt")

                adver_examples = gp.getfiles(ROOT_PATH + "/data/" + task[0] + "/adver_examples/" + task[3])
                test_labels = gp.gettestlabel(task[8], label_file="test_labels.pkl")
                train_attack_param = attack_train_factory(adver_examples, joblib.load(test_labels), 
                                                            task[8] + "/stat", task[7], ROOT_PATH)
                try:
                    subprocess.run([PYTHON_PATH, task[5], "attack", train_attack_param], stdout=fout, stderr=fout)

                except subprocess.SubprocessError:
                    logger.warning("ERROR: Evaluation for model {} failed. Please review logfile {} for error diagnostics.".format(model_name, file_output))


                # Close the training plugin log file
                fout.close()

            comm.send(1, dest=0, tag=7)

        else:
            logger.warning("WARNING: Received empty task list. Returning status 1 to manager.")
            comm.send(1, dest=0, tag=7)

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
