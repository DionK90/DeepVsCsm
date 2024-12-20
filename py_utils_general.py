import matplotlib.figure
import pandas as pd
import numpy as np

from typing import List
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pickle
import os

import py_params
#################################################
### Utility Function for plotting predictions ###
#################################################

def plot_year_value(df_final_long: pd.DataFrame,
                    value:str,
                    row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                    col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                    hue_feature_name:str, hue_feature_values:List[int], 
                    types:List[str], 
                    years: List[int],
                    year_separators:List[int],
                    title_fig: str,
                    col_palette = None,
                    is_true_dotted = False,
                    is_fig_saved = False,
                    y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all years (x-axis).
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `year` that contains the year of the `log_mortality`
    - 1 column specified by the parameter `value` that contains the value to be plotted.

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        value (str): the name of the column to be used as the y-axis.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): 
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): 
        hue_feature_name (str): the name of the column to be plotted with different color in each subplot
        hue_feature_values (List[int]): the values to be considered when plotting multiple line with differnt colour (there will be one line with different colour for each value listed)
        types (List[int]): 
        years (List[int]): all the years to be plotted        
        year_separators (List[int]): a list of years where a straight vertical line will be drawn to separate the years into several intervals
        title_fig (str): the title of the figure
        col_palette:
        is_true_dotted (bool):
        is_fig_saved (bool):
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.
    """
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    if(col_palette is None):
        col_palette = sns.color_palette("viridis", len(hue_feature_values))   

    # Filter the given df according to the given years and types
    df_final_long = df_final_long.loc[(df_final_long[hue_feature_name].isin(hue_feature_values)) & 
                                      (df_final_long[row_feature_name].isin(row_feature_values)) & 
                                      (df_final_long[col_feature_name].isin(col_feature_values)) & 
                                      (df_final_long['type'].isin(types)) & 
                                      (df_final_long['year'].isin(years))]

    # Get all prediction types
    predicted_types = [predicted_type for predicted_type in list(df_final_long['type'].unique()) if predicted_type != 'true']

    # Create subplots
    fig, axs2d = plt.subplots(nrows=len(row_feature_values), ncols=len(col_feature_values), figsize=(18, 20), sharex=True, sharey=True)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()

    lines = []
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):             
             # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot different series based on its colour (all age groups)
            # for age_idx, age in enumerate(ages):
            for hue_idx, hue_val in enumerate(hue_feature_values):
                # Plot the true value
                df_curr = df_final_long.loc[(df_final_long[row_feature_name] == row_val) & 
                                            (df_final_long[col_feature_name] == col_val) & 
                                            (df_final_long[hue_feature_name] == hue_val) & 
                                            (df_final_long[py_params.COL_TYPE] == py_params.TYPE_TRUE)]            
                if(is_true_dotted):
                    curr_ax.scatter(df_curr['year'], df_curr[value], label=hue_val, color=col_palette[hue_idx], s=0.5)
                else:
                    curr_ax.plot(df_curr['year'], df_curr[value], label=hue_val, color=col_palette[hue_idx], 
                                 linewidth=1, linestyle='solid')                    
                
                # Plot the predicted value            
                for type_idx, prediction_type in enumerate(predicted_types):
                    df_curr = df_final_long.loc[(df_final_long[row_feature_name] == row_val) & 
                                                (df_final_long[col_feature_name] == col_val) & 
                                                (df_final_long[hue_feature_name] == hue_val) & 
                                                (df_final_long['type'] == prediction_type)]
                    if(is_true_dotted):
                        curr_ax.plot(df_curr['year'], df_curr[value], color=col_palette[hue_idx], 
                                 linewidth=1, linestyle="-")                           
                    else:
                        curr_ax.plot(df_curr['year'], df_curr[value], color=col_palette[hue_idx], 
                                     linewidth=1, linestyle=py_params.LINESTYLES[type_idx])
                
                
                    
                                    
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Vertical line separating training and test/forecast periods    
            if(year_separators != None):
                for year_separator in year_separators:
                    curr_ax.axvline(
                        year_separator,
                        color='black', linestyle='dashed', linewidth=0.7
                    )
            
            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6))                                          

    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    handles, labels = legend_ax.get_legend_handles_labels()    
    
    # Set the colorbar if hue includes many values
    if(len(hue_feature_values) > 10):
        num_hue_values = hue_feature_values
        if(hue_feature_name == "age"):
            num_hue_values = list()
            for age in hue_feature_values:
                num = [each for each in age if str(each).isdigit()]
                num = int(''.join(num))
                num_hue_values.append(num)

        norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
            cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.05)
        else:
            cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6)
        cbar.set_label(hue_feature_name, rotation=90)
        handles = []
        labels = []        

    
    
    # Add legends for each linestyle (type)
    if(len(types) != len(hue_feature_values) or all(types != hue_feature_values)):
        legend_linestyle = []
        if(is_true_dotted):
            legend_linestyle.append(mlines.Line2D([], [], linestyle='dotted', label=py_params.TYPE_TRUE))
            for type_idx, prediction_type in enumerate(predicted_types):    
                legend_linestyle.append(mlines.Line2D([], [], linestyle="-", label=prediction_type))        
        else:
            legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))
            for type_idx, prediction_type in enumerate(predicted_types):    
                legend_linestyle.append(mlines.Line2D([], [], linestyle=py_params.LINESTYLES[type_idx], label=prediction_type))        
        label_linestyle = [handle.get_label() for handle in legend_linestyle]

        # Combine automatic legend handles and custom legend handles
        all_handles = handles + legend_linestyle
        all_labels = labels + label_linestyle
    # Setting the legend in case where hue and linestyle are both determined by column `type`
    else:   
        all_handles = []     
        for idx, label in enumerate(labels):
            if(label == py_params.TYPE_TRUE):
                all_handles.append(mlines.Line2D([], [], linestyle='solid', color=col_palette[idx]))
            else:
                all_handles.append(mlines.Line2D([], [], linestyle=py_params.LINESTYLES[idx], color=col_palette[idx]))
        all_labels = labels        
    
    # Add legend with both automatic and custom legend entries
    ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    plt.show()   
    return fig

def plot_age_mortality_year(df_long: pd.DataFrame, 
                            row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                            col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                            hue_feature_name:str, hue_feature_values:List[int],                            
                            ages:List[str],
                            types:List[str], 
                            title_fig:str, 
                            value:str = 'log_mortality',
                            linestyles:dict = None,
                            is_fig_saved = False,
                            y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all ages or age groups (x-axis).
    This function can be used to plot various mortality-rate vs age groups 
    in various colours representing different categorical feature (such as years) 
    by using the `hue_feature_name` and `hue_feature_values` parameters.
    
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `age` that contains the age of the population 
    - 1 column named `log_mortality` that contains the log mortality rates

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): the labels for each feature values used to make subplots on the vertical axis
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): the labels for each feature values used to make subplots on the horizontal axis
        hue_feature_name (str): the name of the column to be plotted with different color in each subplot
        hue_feature_values (List[int]): the values to be considered when plotting multiple line with differnt colour (there will be one line with different colour for each value listed)
        ages (List[int]): all the ages to be plotted
        types (List[str]): all the prediction types (train, test, etc.) to be plotted 
        title_fig (str): the title of the figure
        linestyles (dict): a dictionary of dashes (in seaborn format) specifying the dash style for each value in the types parameter
        is_fig_saved (bool): a boolean to indicate whether the resulting plot is saved or not
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.        
    """   
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    col_palette = sns.color_palette("viridis", len(hue_feature_values))   
    
    # Filter the given df according to the given years and types
    df_long = df_long.loc[(df_long[hue_feature_name].isin(hue_feature_values)) & 
                          (df_long[row_feature_name].isin(row_feature_values)) & 
                          (df_long[col_feature_name].isin(col_feature_values)) & 
                          (df_long['type'].isin(types)) & 
                          (df_long['age'].isin(ages))]
    # df_long = df_long.loc[(df_long['type'].isin(types)) & (df_long[hue_feature_name].isin(hue_feature_values))]

    # Get all prediction types    
    predicted_types = [predicted_type for predicted_type in list(df_long['type'].unique()) if predicted_type != py_params.TYPE_TRUE]
    # Set the line style for each predicted type
    if(linestyles is None):
        linestyles = {}
        for idx_style, predicted_type in enumerate(predicted_types):
            linestyles.update({predicted_type: py_params.LINESTYLES[idx_style][1]})
        linestyles[py_params.TYPE_TRUE] = ''

    # Create subplots
    fig, axs2d = plt.subplots(len(row_feature_values), len(col_feature_values), figsize=(18, 18), sharex=True, sharey=True)
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()
    
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):
            # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot
            sns.lineplot(data=df_long.loc[(df_long[row_feature_name] == row_val) & (df_long[col_feature_name] == col_val)], 
                         x='age', y=value, hue=hue_feature_name, style='type', 
                         dashes=linestyles,
                         ax=curr_ax, palette=col_palette)
            
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_val}| {col_feature_name}: {col_val}', fontsize=12)

            # Remove the legend to create a separate collective legend for all plots
            curr_ax.get_legend().remove()            

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6)) 
            
    # Create a single legend for all the subplots
    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    all_handles, all_labels = legend_ax.get_legend_handles_labels()    
    
    # Set a colorbar if there are too many hue values, and make the legend for linestyle manually
    if(len(hue_feature_values) > 10):
        num_hue_values = hue_feature_values
        if(hue_feature_name == "age"):
            num_hue_values = list()
            for age in hue_feature_values:
                num = [each for each in age if str(each).isdigit()]
                num = int(''.join(num))
                num_hue_values.append(num)
        norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
            cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.1)
        else:
            cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6, fraction=0.1)
        cbar.set_label(hue_feature_name, rotation=90)
        handles = []
        labels = []        

    # Add legends for each linestyle (type)
    legend_linestyle = []
    if(py_params.TYPE_TRUE in types):
        legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))    
    for prediction_type in predicted_types:    
        legend_linestyle.append(mlines.Line2D([], [], linestyle=(0, linestyles[prediction_type]), label=prediction_type))
    label_linestyle = [handle.get_label() for handle in legend_linestyle]
    
    # Combine automatic legend handles and custom legend handles
    all_handles = handles + legend_linestyle
    all_labels = labels + label_linestyle
        
    # Add legend with both automatic and custom legend entries
    # ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    # print(legend_linestyle)
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    plt.show()   

def plot_age_mortality_model(df_long: pd.DataFrame, 
                             row_feature_name:str, row_feature_values:List[int], row_labels:dict,
                             col_feature_name:str, col_feature_values:List[int], col_labels:dict,
                             years:List[int],                       
                             ages:List[str],
                             types:List[str], 
                             title_fig:str, 
                             value:str='log_mortality',
                             fig:matplotlib.figure.Figure = None,
                             col_palette = None,
                             linestyles:dict = None,
                             is_fig_saved = False,
                             y_limit:list = None):
    """
    A function used to make a plot between the mortality rate (y-axis) of all ages or age groups (x-axis).
    This function can be used to plot various mortality-rate vs age groups 
    in various colours representing different categorical feature (such as years) 
    by using the `hue_feature_name` and `hue_feature_values` parameters.
    
    For the moment, the given DataFrame must be in long format with only 6 columns used as follows:
    - 1 column is used to determine the number of subplots in the vertical axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of subplots in the horizontal axis. If only 1 plot is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column is used to determine the number of lines made in each subplot, each with different colour. If only 1 line is desired, specify the name of the column and the specific value of that column as a one-element list.
    - 1 column named `type` that explains how the `log_mortality` is obtained, through training, validation, testing, or real value. If more than one type are given, each will be plotted with different type of linestroke.
    - 1 column named `age` that contains the age of the population 
    - 1 column named `log_mortality` that contains the log mortality rates

    Args:
        df_final_long (pd.DataFrame): a DataFrame containing the mortality data in a long format. The DataFrame must have columns 'year' and 'type' that indicate the year and the type (predicted, true, train, etc.) of the log_mortality. See the description of the function to see applicable constraints on the DataFrame.
        row_feature_name (str): the name of the column to be used to make subplots on the vertical axis
        row_feature_values (List[int]): the values to be considered when making subplots (there will be one row for each value listed)
        row_labels (dict): the labels for each feature values used to make subplots on the vertical axis
        col_feature_name (str): the name of the column to be used to make subplots on the horizontal axis
        col_feature_values (List[int]): the values to be considered when making subplots (there will be one column for each value listed)
        col_labels (dict): the labels for each feature values used to make subplots on the horizontal axis
        years (Lits[int]): all the years to be plotted
        ages (List[int]): all the ages to be plotted
        types (List[str]): all the prediction types (train, test, etc.) to be plotted 
        title_fig (str): the title of the figure
        linestyles (dict): a dictionary of dashes (in seaborn format) specifying the dash style for each value in the types parameter
        is_fig_saved (bool): a boolean to indicate whether the resulting plot is saved or not
        y_limit (List[int]): a list of two values indicating the minimum and maximum value of the y-axis respectively.        
    """   
    # Set the base font size of texts in the figure. 
    plt.rcParams['font.size'] = '21'

    # Specify the color pattern 
    # Specify the color pattern 
    if(col_palette is None):
        col_palette = sns.color_palette("nipy_spectral", len(types))   

    
    # Filter the given df according to the given years and types
    df_long = df_long.loc[(df_long[row_feature_name].isin(row_feature_values)) & 
                          (df_long[col_feature_name].isin(col_feature_values)) & 
                          (df_long['type'].isin(types)) & 
                          (df_long['year'].isin(years)) &
                          (df_long['age'].isin(ages))]
    # df_long = df_long.loc[(df_long['type'].isin(types)) & (df_long[hue_feature_name].isin(hue_feature_values))]

    # Get all prediction types    
    predicted_types = [predicted_type for predicted_type in list(df_long['type'].unique()) if predicted_type != py_params.TYPE_TRUE]
    # Set the line style for each predicted type
    if(linestyles is None):
        linestyles = {}
        for idx_style, predicted_type in enumerate(predicted_types):
            linestyles.update({predicted_type: py_params.LINESTYLES[idx_style][1]})
        linestyles[py_params.TYPE_TRUE] = ''

    # Create subplots
    if(fig is None):
        fig, axs2d = plt.subplots(len(row_feature_values), len(col_feature_values), figsize=(18, 18), sharex=True, sharey=True)
    else:
        axs2d = fig.subplots(len(row_feature_values), len(col_feature_values), sharex=True, sharey=True)
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        axs = axs2d.flatten()
    
    for row_idx, row_val in enumerate(row_feature_values):
        for col_idx, col_val in enumerate(col_feature_values):
            # Determine the axis to plot
            if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
                curr_ax = axs[row_idx * len(col_feature_values) + col_idx]
            else:
                curr_ax = axs

            # Plot
            sns.lineplot(data=df_long.loc[(df_long[row_feature_name] == row_val) & (df_long[col_feature_name] == col_val)], 
                         x='age', y=value, hue='type', style='type', 
                         dashes=linestyles,
                         ax=curr_ax, palette=col_palette)
            
            # Specify the y-axis limits
            if y_limit != None:
                curr_ax.set_ylim(y_limit[0], y_limit[1])  # Set y-axis limits

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_val}| {col_feature_name}: {col_val}', fontsize=12)

            # Remove the legend to create a separate collective legend for all plots
            curr_ax.get_legend().remove()            

            # Set the title of each subplots
            curr_ax.set_title(f'{row_feature_name}: {row_labels[row_val]} \n{col_feature_name}: {col_labels[col_val]}', fontsize=12)

            # Setting the x-axis labels
            curr_ax.tick_params(axis='x', rotation=45)
            curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6)) 
            
    # Create a single legend for all the subplots
    # Get the legend from the axes (for different hue colors)    
    if(len(row_feature_values) > 1 or len(col_feature_values) > 1):
        legend_ax = axs[len(col_feature_values)-1]        
    else:
        legend_ax = axs
    all_handles, all_labels = legend_ax.get_legend_handles_labels()    
    
    # Set a colorbar if there are too many hue values 
    # if(len(types) > 10):
    #     num_hue_values = len(types)
    #     if(hue_feature_name == "age"):
    #         num_hue_values = list()
    #         for age in hue_feature_values:
    #             num = [each for each in age if str(each).isdigit()]
    #             num = int(''.join(num))
    #             num_hue_values.append(num)
    #     norm = plt.Normalize(min(num_hue_values), max(num_hue_values))
    #     sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    #     sm.set_array([])
    #     if(len(row_feature_values) > 1 and len(col_feature_values) > 1):
    #         cbar = legend_ax.figure.colorbar(sm, ax=axs2d[:,:], shrink=0.6, fraction=0.1)
    #     else:
    #         cbar = legend_ax.figure.colorbar(sm, ax=legend_ax, shrink=0.6, fraction=0.1)
    #     cbar.set_label(hue_feature_name, rotation=90)
    #     handles = []
    #     labels = []        

    #     # Add legends for each linestyle (type)
    #     legend_linestyle = []
    #     if(py_params.TYPE_TRUE in types):
    #         legend_linestyle.append(mlines.Line2D([], [], linestyle='-', label=py_params.TYPE_TRUE))    
    #     for prediction_type in predicted_types:    
    #         legend_linestyle.append(mlines.Line2D([], [], linestyle=(0, linestyles[prediction_type]), label=prediction_type))
    #     label_linestyle = [handle.get_label() for handle in legend_linestyle]
        
    #     # Combine automatic legend handles and custom legend handles
    #     all_handles = handles + legend_linestyle
    #     all_labels = labels + label_linestyle
        
    # Add legend with both automatic and custom legend entries
    # ncol_legend = len(hue_feature_values)/2 if len(hue_feature_values) > 1 else 1

    # Set the legend
    # print(legend_linestyle)
    fig.legend(handles=all_handles, labels=all_labels, loc='upper right',
            fancybox=True, shadow=True)
    fig.suptitle(title_fig, fontsize=24)
    fig.subplots_adjust(top=0.9, right=0.8)
    if(is_fig_saved):
        plt.savefig(f"{title_fig}.png")
    return fig, axs

def load_all_files(folder, file_identifier):
    # Get the list of all fitted models in the folder
    files = os.listdir(folder)
    files = [file for file in files if file.find(file_identifier) >= 0]

    # For each file
    model_names = list()
    dfs = list()
    for file in files:    
        # Open the df
        with open(f"{folder}/{file}", "rb") as outfile:
            dfs.append(pickle.load(outfile))

        # Shortened some of the hyperparameters        
        # model_name = model_name[model_name.find('deep6')+6:]
        model_name = file[:file.find(".pickle")]
        model_name = model_name[model_name.find('deep6'):]
        model_name = model_name.replace('bootstrap', 'bs')
        model_name = model_name.replace('batch', 'b')

        # Determine whether it is lmxt or mxt    
        # input = ""
        # if model_name.find('lmxt') >= 0:
        #     input = 'lmxt'
        # elif model_name.find('mxt') >= 0:
        #     input = 'mxt'

        # post = ""
        # if model_name.find("post") >= 0:
        #     post = "_transfer"
        # if model_name.find("float") >= 0:
        #     post += "_float"

        # Get the age range
        ages = model_name[model_name.find('age') + 3:]
        idx_end = ages.find("_")
        if(idx_end >= 0):
            ages = ages[0:idx_end]
        else:
            ages = ages[0:]        
        if(len(ages) == 3):            
            ages = f"00_{ages[1:]}"
        else:
            ages = f"{ages[:2]}_{ages[2:]}"
        
        # Shorten the age range
        # model_name = f"{input}_{model_name[0:model_name.find('age')]}{ages}{post}"    
        # model_name = f"{model_name[0:model_name.find('age')]}{ages}_{model_name[model_name.find('age')+7:]}"
        model_name = f"{model_name[0:model_name.find('age')]}{ages}{model_name[model_name.find('age')+6:]}"
        # if(model_name.find("bs") >= 0):
        #     print(model_name)
        # model_name = f"{model_name[0:model_name.find('age')]}{ages}"
        # if(model_name.find("bs") >= 0):
        #     print(ages)
        #     print(model_name)   
        model_name = model_name.replace('epoch200_', "")
        
        # Add model names to a lists
        model_names.append(model_name)

    return [dfs, model_names]
