import math
import random
from typing import List, Tuple

import joblib
import matplotlib.colors as color
import matplotlib.pyplot as plt
import numpy as np
from jespipe.plugin.start import start
from jespipe.plugin.clean.plotter import Plot


class MatplotColors:
    """Class for selecting a random HTML color."""
    def __init__(self) -> None:
        """Create instance of MatplotColors class."""
        self.html_colors = self._random_color_tuple()

    def getrandomcolor(self) -> str:
        """
        Pick a random hexadecimal color code supported by Matplotlib.
        
        ### Returns:
        :return: Random hexdecimal color code supported by Matplotlib.
        """
        return self.html_colors[random.randint(0, len(self.html_colors))]

    def _random_color_tuple(self) -> Tuple[str]:
        """
        Internal function to query supported colors by Matplotlib.

        ### Returns:
        :return: Tuple containing the supported hexadecimal color codes.
        """
        color_dict = color.get_named_colors_mapping()
        color_tuple = tuple()
        for key in color_dict:
            if isinstance(color_dict[key], str):
                color_tuple += (color_dict[key],)

        return color_tuple


class RmseSiMae(Plot):
    def __init__(self, parameters: dict) -> None:
        """
        Create a RmseSiMae plotter instance.

        ### Parameters:
        :param parameters: Parameter dictionary sent by Jespipe

        ### Methods:
        - public
          - plot (abstract): Create plot using data passed by the user.
        """
        self.model_list = parameters["model_list"]
        self.plot_name = parameters["plot_name"]
        self.save_path = parameters["save_path"]

    def plot(self) -> None:
        """
        Create plot using data passed by the user. Makes two separate graphs.
        One graph is for RMSE v.s. MAE over the change budget, the other for Scatter Index
        over the change budget.
        """
        cw_l2_data_list = list(); cw_linf_data_list = list()

        for model in self.model_list:
            cw_l2_data_list.append(joblib.load(model + "/stat/mse-rmse-si-mae-cw_l2_1.pkl"))

        cw_l2_attack = list(zip(self.model_list, cw_l2_data_list))

        for model in self.model_list:
            cw_linf_data_list.append(joblib.load(model + "/stat/mse-rmse-si-mae-cw_inf_1.pkl"))

        cw_linf_attack = list(zip(self.model_list, cw_linf_data_list))

        # RMSE v.s. MAE over change budget
        # There will be one graph for each manipulation
        # CW_L2 ATTACK
        for datum in cw_l2_attack:
            ran_color_list = self._random_color_picker(2)
            fig, axis_1 = plt.subplots()

            # Generate x_axis
            x_axis = list()
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

            x_axis.sort()

            # Sort data in datum[1]
            data_dict = self._sort_dict(x_axis, datum[1])

            # PLOT RMSE ON AXIS 1
            # Generate y_axis ticks for RMSE
            rmse_values = list()
            for key in data_dict:
                rmse_values.append(data_dict[key]["rmse"])

            # Generate 10 ticks for the y_axis
            y_axis_1_ticks = np.linspace(0.0, float(math.ceil(max(rmse_values))), num=10)

            # Plot RMSE
            axis_1.plot(x_axis, rmse_values, color=ran_color_list[0], linestyle="solid")
            axis_1.set_xlabel("Perturbation Budget")
            axis_1.set_ylabel("Root Mean Squared Error (RMSE)", color=ran_color_list[0])
            axis_1.set_yticks(y_axis_1_ticks)
            
            for tick_label, tick_line in zip(axis_1.get_yticklabels(), axis_1.get_yticklines()):
                tick_label.set_color(ran_color_list[0])
                tick_line.set_color(ran_color_list[0])

            # PLOT MAE ON AXIS 2
            axis_2 = axis_1.twinx()

            # Generate y-axis ticks for MAE
            mae_values = list()
            for key in data_dict:
                mae_values.append(data_dict[key]["mae"])

            # Generate 10 ticks for the y_axis
            y_axis_2_ticks = np.linspace(0.0, float(math.ceil(max(mae_values))), num=10)

            # Plot MAE
            axis_2.plot(x_axis, mae_values, color=ran_color_list[1], linestyle="solid")
            axis_2.set_ylabel("Mean Absolute Error (MAE)", color=ran_color_list[1])
            axis_2.set_yticks(y_axis_2_ticks)
            
            for tick_label, tick_line in zip(axis_2.get_yticklabels(), axis_2.get_yticklines()):
                tick_label.set_color(ran_color_list[1])
                tick_line.set_color(ran_color_list[1])

            model_tag = datum[0].split("/"); model_tag = model_tag[-1]
            plt.title("RMSE and MAE as Perturbation Budget increases for CW_L2 attack on model {}".format(model_tag))
            plt.savefig(self.save_path + "/{}-cw_l2-rmse-mae-{}.png".format(self.plot_name, model_tag), bbox_inches="tight")
            plt.close()

        # CW_Linf ATTACK
        for datum in cw_linf_attack:
            ran_color_list = self._random_color_picker(2)
            fig, axis_1 = plt.subplots()

            # Generate x_axis
            x_axis = list()
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

            x_axis.sort()

            # Sort data in datum[1]
            data_dict = self._sort_dict(x_axis, datum[1])

            # PLOT RMSE ON AXIS 1
            # Generate y_axis ticks for RMSE
            rmse_values = list()
            for key in data_dict:
                rmse_values.append(data_dict[key]["rmse"])

            # Generate 10 ticks for the y_axis
            y_axis_1_ticks = np.linspace(0.0, float(math.ceil(max(rmse_values))), num=10)

            # Plot RMSE
            axis_1.plot(x_axis, rmse_values, color=ran_color_list[0], linestyle="solid")
            axis_1.set_xlabel("Perturbation Budget")
            axis_1.set_ylabel("Root Mean Squared Error (RMSE)", color=ran_color_list[0])
            axis_1.set_yticks(y_axis_1_ticks)

            for tick_label, tick_line in zip(axis_1.get_yticklabels(), axis_1.get_yticklines()):
                tick_label.set_color(ran_color_list[0])
                tick_line.set_color(ran_color_list[0])

            # PLOT MAE ON AXIS 2
            axis_2 = axis_1.twinx()

            # Generate y-axis ticks for MAE
            mae_values = list()
            for key in data_dict:
                mae_values.append(data_dict[key]["mae"])

            # Generate 10 ticks for the y_axis
            y_axis_2_ticks = np.linspace(0.0, float(math.ceil(max(mae_values))), num=10)

            # Plot MAE
            axis_2.plot(x_axis, mae_values, color=ran_color_list[1], linestyle="solid")
            axis_2.set_ylabel("Mean Absolute Error (MAE)", color=ran_color_list[1])
            axis_2.set_yticks(y_axis_2_ticks)
            
            for tick_label, tick_line in zip(axis_2.get_yticklabels(), axis_2.get_yticklines()):
                tick_label.set_color(ran_color_list[1])
                tick_line.set_color(ran_color_list[1])
            
            model_tag = datum[0].split("/"); model_tag = model_tag[-1]
            plt.title("RMSE and MAE as Perturbation Budget increases for CW_Linf attack on model {}".format(model_tag))
            plt.savefig(self.save_path + "/{}-cw_linf-rmse-mae-{}.png".format(self.plot_name, model_tag), bbox_inches="tight")
            plt.close()
        
        # Scattter Index over the change budget
        # All the manipulations will be put on the same graph.
        # CW_L2 ATTACK
        plt.figure()
        plt.xlabel("Perturbation Budget"); plt.ylabel("Scatter Index (in %)")
        plt.title("Scatter Index as Perturbation Budget increases for CW_L2 attack")
        ran_color_list = self._random_color_picker(len(cw_l2_attack)); i = 0

        # Find maximum scatter index value
        scatter_values = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                scatter_values.append(datum[1][key]["scatter_index"])

        # Generate y_axis ticks; generate 10 ticks
        y_axis_ticks = np.linspace(0.0, float(math.ceil(max(scatter_values))), num=10)
        plt.yticks(y_axis_ticks)

        # Generate x_axis
        x_axis = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

        x_axis.sort()

        for datum in cw_l2_attack:
            values = list()
            data_dict = self._sort_dict(x_axis, datum[1])
            for key in data_dict:
                values.append(data_dict[key]["scatter_index"])

            # Append values to the plot
            line_name = datum[0].split("/"); line_name = line_name[-1]
            plt.plot(x_axis, values, color=ran_color_list[i], linestyle=self._random_linestyle(), label=line_name)
            i += 1

        plt.legend()
        plt.savefig(self.save_path + "/{}-cw_l2-sci-pertur_budget.png".format(self.plot_name))
        plt.close()

        # CW_Linf ATTACK
        plt.figure()
        plt.xlabel("Perturbation Budget"); plt.ylabel("Scatter Index (in %)")
        plt.title("Scatter Index as Perturbation Budget increases for CW_Linf attack")
        ran_color_list = self._random_color_picker(len(cw_linf_attack)); i = 0

        # Find maximum scatter index value
        scatter_values = list()
        for datum in cw_linf_attack:
            for key in datum[1]:
                scatter_values.append(datum[1][key]["scatter_index"])

        # Generate y_axis ticks; generate 10 ticks
        y_axis_ticks = np.linspace(0.0, float(math.ceil(max(scatter_values))), num=10)
        plt.yticks(y_axis_ticks)

        # Generate x_axis
        x_axis = list()
        for datum in cw_l2_attack:
            for key in datum[1]:
                if float(key) not in x_axis:
                    x_axis.append(float(key))

        x_axis.sort()

        for datum in cw_linf_attack:
            values = list()
            data_dict = self._sort_dict(x_axis, datum[1])
            for key in data_dict:
                values.append(data_dict[key]["scatter_index"])

            # Append values to the plot
            line_name = datum[0].split("/"); line_name = line_name[-1]
            plt.plot(x_axis, values, color=ran_color_list[i], linestyle=self._random_linestyle(), label=line_name)
            i += 1

        plt.legend()
        plt.savefig(self.save_path + "/{}-cw_linf-sci-perturb_budget.png".format(self.plot_name))
        plt.close()

    def _random_color_picker(self, num_of_categories: int) -> List[str]:
        """
        Internal method for randomly assigning hexadecimal colors to
        the categories of a plot.

        ### Parameters:
        :param num_of_categories: Number of categories that are going to be in the plot.

        ### Returns:
        :return: List of hexadecimal colors.
        """
        color_list = list()
        color_picker = MatplotColors()

        i = 0
        while i < num_of_categories:
            ran_color = color_picker.getrandomcolor()
            if ran_color not in color_list:
                color_list.append(ran_color); i += 1

        return color_list

    def _random_linestyle(self) -> str:
        """
        Internal method for selecting a random linestyle in Matplotlib

        ### Returns:
        :return: solid, dotted, dashed, or dashdot linestyle.
        """
        linestyles = ["solid", "dotted", "dashed", "dashdot"]
        i = random.randint(0, 3)
        return linestyles[i]

    def _sort_dict(self, keys: list, dict_to_sort: dict) -> dict:
        """
        Internal method for sorting the dictionary created by Jespipe

        ### Parameters:
        :param keys: A sorted list containing the keys of the dictionary.
        :param dict_to_sort: Dictionary to sort from least to greatest.

        ### Returns:
        :return: A sorted dictionary from least to greatest float values.
        """
        d = dict()
        for key in keys:
            if str(key) in dict_to_sort:
                d[str(key)] = dict_to_sort[str(key)]

        return d

if __name__ == "__main__":
    stage, parameters = start()

    # Execute code block based on passed stage from Jespipe
    if stage == "clean":
        plotter = RmseSiMae(parameters)
        plotter.plot()

    else:
        raise ValueError("Received invalid stage {}. Please only pass valid stages from Jespipe.".format(stage))
