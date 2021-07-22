#!/usr/bin/env python3

from jespipe.plugin.start import start
from jespipe.plugin.clean.plotter import Plot


class Plotter(Plot):
    """Class to automate the analysis of data during Jespipe's runtime."""
    def __init__(self, parameters: dict) -> None:
        """
        Create instance of the Plotter class.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe when plugin is called.

        #### Notes:
        - You can add internal methods to assist with data plotting and analysis.
        """
        pass

    def plot(self) -> None:
        """
        Abstract method for plotting data.

        ### Returns:
        None; in this block of code you should be generating plots and saving them.
        """
        pass


if __name__ == "__main__":
    # Read in stage and parameter dictionary sent by Jespipe
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "clean":
        """
        This block will be executed during the attack stage of Jespipe.

        ### Parameter dictionary content:
        - `models`: List of model root directories for models that will be included in the plot.
        - `plot_name`: Name to use for the plot.
        - `save_path`: Where to save the generated plots.
        """
        pass

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
