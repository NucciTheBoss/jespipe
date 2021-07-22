# IMPORTANT: psuedo code; do not try to execute on your system
# as it will not work for you.

# "start" returns two parameters passed from the command line (specified via subproccess)
# "Build" is an abstract class that has one abstract method; build()
# "Execute" is an abstract class that has three abstract methods; model_fit() & model_predict() & model_evaluate()
from jespipe.plugin import start, Build, Execute

# The Execute abstract class will have some predefined methods for the user to utilize
# such as _to_csv() and if time _to_json()

# "start" by default accepts three parameters: stage, dataset_name, and model_parameters

# Main block of code that runs
if __name__ == "__main__":
    # Start accepts three positional arguments from standard input; stage, dataset name, params
    params = jespipe.start()
    if params[0] == "train":
        build = Build()  # Build doesn't accept any params at initialization
        build.build(params[1], params[2], user_defined_params="NaN")  # Build needs to accept two arguments from start

    elif params[0] == "attack":
        exec = Execute()  # Like build, doesn't accept any params at initialization
        model = exec.model_fit(params[2])
        model_stat = exec.model_predict(params[2])
        exec.model_evaluate(params[2], model_stat)
