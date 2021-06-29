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
                filepath = os.path.join(root, filename)
                root_list.append(filepath)

    return root_list