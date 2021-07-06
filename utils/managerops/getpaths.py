import os


def getdirs(*paths):
    """Get subdirectory structure of given paths.
    Returns list of subdirectory paths.
    
    Keyword arguments:
    *paths -- paths to get the subdirectory structure of."""
    root_list = list()
    for path in paths:
        for root, directories, files in os.walk(path):
            for directory in directories:
                root_list.append(os.path.join(root, directory))

    return root_list


def getfiles(*paths):
    """Get file tree structure of given paths.
    Returns list of file paths.
    
    Keyword arguments:
    *paths -- paths to get file tree structure of."""
    root_list = list()
    for path in paths:
        for root, directories, files in os.walk(path):
            for filename in files:
                root_list.append(os.path.join(root, filename))

    return root_list


def getmodels(model_path, **kwargs):
    """Get the paths to models using given model path.
    Returns list of file paths to models.
    
    Keyword arguments:
    model_path -- root path to models.
    **kwargs; format - format models are saved in (i.e. .pkl, .h5)."""
    root_list = list()
    for root, directories, files in os.walk(model_path):
        for filename in files:
            if filename.lower().endswith(kwargs.get("format")):
                root_list.append(os.path.join(root, filename))

    return root_list
