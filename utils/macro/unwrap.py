def unwrap_train(train_dict):
    """Convert train job control dictionary to workable tuple.
    
    Keyword arguments:
    train_dict -- train job control dictionary to convert to workable tuple."""
    # Create empty list that will eventually be returned to main.py
    l = list()

    # Split train_dict up by the stored datasets and loop over each dataset
    dataset_list = list(train_dict.items())
    for dataset in dataset_list:
        # TODO: Write technique for unwrapping dictionaries by dataset.
        pass


def unwrap_attack(attack_dict):
    """Convert attack job control dictionary to workable tuple.
    
    Keyword arguments:
    attack_dict -- attack job control dictionary to convert to workable tuple."""
    pass


# TODO: Can write this function once I discover what Sheila wants me to do with clean stage
# def unwrap_clean(clean_dict):
#     """Convert clean job control dictionary to workable tuple.
    
#     Keyword arguments:
#     clean_dict -- clean job dictionary to convert to a workable tuple."""
#     pass
