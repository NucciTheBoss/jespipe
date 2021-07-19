import os
from typing import List, Union


def getdirs(*paths) -> List:
    """
    Get subdirectory structure of passed paths
    
    ### Parameters:
    - args
      - paths: Paths to get the subdirectory structure of.

    ### Returns:
    :return: List of directory paths.
    """
    root_list = list()
    for path in paths:
        for root, directories, files in os.walk(path):
            for directory in directories:
                root_list.append(os.path.join(root, directory))

    return root_list


def getfiles(*paths) -> List:
    """
    Get system file tree structure of passed paths.
    
    ### Parameters:
    - args
      - paths: Paths to get file tree structure of.
    
    ### Returns:
    :return: List of system file paths.
    """
    root_list = list()
    for path in paths:
        for root, directories, files in os.walk(path):
            for filename in files:
                root_list.append(os.path.join(root, filename))

    return root_list


def getmodels(model_path: str, **kwargs) -> List:
    """
    Get the system file paths to models using passed model path.
    
    ### Parameters:
    model_path -- root path to models.
    **kwargs; format - format models are saved in (i.e. .pkl, .h5).

    ### Returns:
    """
    root_list = list()
    for root, directories, files in os.walk(model_path):
        for filename in files:
            if filename.lower().endswith(kwargs.get("format")):
                root_list.append(os.path.join(root, filename))

    return root_list


def getfile(root_path: str, file_name: str) -> Union[str, None]:
    """
    Get the system file path to the passed file name.

    ### Parameters:
    :param root_path: Root directory to recursively look through.
    :param file_name: File to look for.

    ### Returns:
    :return: System file path to the passed file name.
    """
    for root, directories, files in os.walk(root_path):
        for filename in files:
            if filename.lower().endswith(file_name):
                return os.path.join(root, filename)


def gettestfeat(model_root_path: str, **kwargs) -> Union[str, None]:
    """
    Get the system file path to test features pickle file.

    ### Parameters:
    :param model_root_path: Root model directory to recursively look through.
    - kwargs
      - feature_file: Test features pickle file to look for.

    ### Returns:
    :return: System file path to test features pickle file.
    """
    return getfile(model_root_path, kwargs.get("feature_file"))


def gettestlabel(model_root_path: str, **kwargs) -> Union[str, None]:
    """
    Get the system file path to test labels pickle file.

    ### Parameters:
    :param model_root_path: Root model directory to recursively look through.
    - kwargs
      - label_file: Test labels pickle file to look for.

    ### Returns:
    :return: System file path to test labels pickle file.
    """
    return getfile(model_root_path, kwargs.get("label_file"))
