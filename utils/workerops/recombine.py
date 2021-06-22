import pandas as pd
import os


def recombine(features, labels, save=False, **kwargs):
    """Recombine a manipulated dataset back into a single pandas DataFrame.
    
    Keyword arguments:
    features -- features of the manipulated dataset.
    labels -- target labels of the manipulated dataset.
    save -- save a copy of the recombined dataset in CSV format (default False).
    **kwargs -> key: save_path -- path to save recombined dataset (not needed if save == False).
    **kwargs -> key: manip_tag -- name of the data manipulation used on the dataset."""
    # Convert features and labels from numpy.ndarray to pandas.DataFrame
    features = pd.DataFrame(features); labels = pd.DataFrame(labels)

    # Recombine dataset using pd.concat
    recomb = pd.concat([features, labels], axis=1, join="inner")

    if save is False:
        return recomb

    else:
        save_path = kwargs.get("save_path"); manip_tag = kwargs.get("manip_tag")

        if os.path.exists(save_path) is False:
            os.makedirs(save_path, exist_ok=True)

        recomb.to_csv(save_path + "/" + manip_tag + ".csv", index=False, header=None)
        return recomb
