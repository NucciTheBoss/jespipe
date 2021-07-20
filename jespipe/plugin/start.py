import argparse
import joblib


def start():
    """Pull two parameters from stdin; stage and parameters."""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str)
    parser.add_argument("parameters", type=str)
    args = parser.parse_args()

    # Load pickled parameter dictionary
    params = joblib.load(args.parameters)

    # Return tuple in the following format: (stage, parameters)
    return args.stage, params
