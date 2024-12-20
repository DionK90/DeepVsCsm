
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

import py_params
import py_hmd_data
from py_hmd_data import HmdMortalityData, HmdResidual, HmdError
import py_utils_general

import os
import pickle

##########################
### Parameter for data ###
##########################
AGE_START = 0
AGE_END = 99
MIN_YEAR_TRAIN = 1959
MAX_YEAR_TRAIN = 1999
MAX_YEAR_VALID = 1999
MAX_YEAR_TEST = 2017
# DEFAULT_MODELS = ['LC_SVD_00_99', 'APC_00_99', 'RH_00_99',
#                   'lmxt_64_bs2_b64_00_99', 'lmxt_128_bs2_b64_00_99', 
#                   'lmxt_64_bs0_b64_00_99', 'lmxt_128_bs0_b64_00_99', 
#                   'lmxt_256_bs2_b256_00_99']
# DEFAULT_MODELS = ['LC_SVD_00_99', 'APC_00_99', 'RH_00_99']
DEFAULT_MODELS = ['LC_SVD_00_99', 'APC_00_99', 'RH_00_99',
                  'deep6_128_Lenovo_wonorm_00_99_emb0_ret1_fr1_add2', # 4 wins over cause, original freeze
                  'deep6_128_Dell_00_99_emb0_ret0_fr1_add1', # only 1 win over cause, 11 wins over ages
                  'deep6_128_Dell_bs2_00_99', # 4 wins over cause, 1 win over ages
                  'deep6_128_HP_00_99_emb0_ret1_fr2_add2', # 3 wins over cause, 6 wins over cause
                  'deep6_128_Lenovo_wonorm_bs2_00_99'] # best training error, 25 wins over ages, original hyperparam
MAX_MODELS = 10
countries = [py_params.BIDICT_COUNTRY_HMD['USA']]
sexes = [0, 1]
causes = [0, 1, 2, 3, 4, 5]
years = list(range(1959, 2018))
ages = list(range(AGE_START, AGE_END))

#####################
### Streamlit App ###
#####################
if 'df_all' not in st.session_state:
    st.write("No data found. Please load the main page first.")
else:
    model_names = st.session_state.model_names
    df_all = st.session_state.df_all        

    st.title('Neural Networks vs Classical Stochastic Models')

    ###################
    ### Predictions ###
    ###################
    st.sidebar.markdown("# Control for Predictions")

    is_overall_log = st.sidebar.checkbox("Plot All Log-Rate", value=True)
    is_overall_rate = st.sidebar.checkbox("Plot All Rate", value=False)

    # Slider for selecting an age (0-99)
    age = st.sidebar.slider("Select age", min_value=0, max_value=99, value=20)

    # Select models to be plotted
    models = st.sidebar.multiselect(f"Select up to {MAX_MODELS} models for the combined plots:", 
                                    model_names,
                                    DEFAULT_MODELS)

    # Combobox for gender selection
    sex = st.sidebar.selectbox("Select gender", options=['Male', 'Female'])
    sex_code = py_params.BIDICT_SEX_1_2.inverse[sex.lower()]

    # Combobox for cause selection
    cause = st.sidebar.selectbox("Select cause", options=list(py_params.BIDICT_CAUSE_1_HMD.values())[1:])
    cause_code = py_params.BIDICT_CAUSE_1_HMD.inverse[cause.lower()]

    # Select model to focus on
    model = st.sidebar.selectbox("Select model to focus on", options = models)

    

    ### Year vs Log-Mortality ###
    #############################
    # Validate selection
    st.header("Plot Year vs Log-Rate", divider=True)
    if len(models) > MAX_MODELS:
        st.error(f"⚠️ You can only select up to {MAX_MODELS} options.")
    else:
        # Create figure title
        title = f"Year vs Log-Rate for "
        f"Year vs Log-Rate for various models (age: {age})",
            
        # Plot combined year vs log_mortality (all sexes and causes)
        df_filtered = df_all.loc[df_all.type.isin(models + [py_params.TYPE_TRUE])]
        y_min_log = df_filtered.loc[df_filtered.age == age]['log_mortality'].min()
        y_max_log = df_filtered.loc[df_filtered.age == age]['log_mortality'].max()

        st.subheader("Year vs Log-Rate All Sexes and Causes", divider = 'blue')
        if is_overall_log:
            fig = py_utils_general.plot_year_value(df_final_long=df_filtered.loc[df_filtered.age == age],
                                            value='log_mortality',
                                            row_feature_name='sex', row_feature_values=[1,2], row_labels=dict(py_params.BIDICT_SEX_1_2),
                                            col_feature_name='cause', col_feature_values=list(df_filtered['cause'].unique()), col_labels = dict(py_params.BIDICT_CAUSE_1_HMD),
                                            hue_feature_name='type', hue_feature_values=df_filtered.type.unique(),
                                            types=df_filtered.type.unique(), 
                                            years=range(MIN_YEAR_TRAIN, MAX_YEAR_TEST+1),
                                            year_separators=[MAX_YEAR_TRAIN],
                                            title_fig=title,
                                            is_fig_saved = False,
                                            is_true_dotted=False,                                    
                                            y_limit = [y_min_log, y_max_log],
                                            col_palette=sns.color_palette("tab10", len(df_filtered.type.unique())))
            
            st.pyplot(fig)
        else:
            st.write("Please check the checkbox on the left to plot this part.")

        st.subheader("Year vs Rate All Sexes and Causes", divider = 'blue')
        if is_overall_rate:
            fig = py_utils_general.plot_year_value(df_final_long=df_filtered.loc[df_filtered.age == age],
                                            value='mortality',
                                            row_feature_name='sex', row_feature_values=[1,2], row_labels=dict(py_params.BIDICT_SEX_1_2),
                                            col_feature_name='cause', col_feature_values=list(df_filtered['cause'].unique()), col_labels = dict(py_params.BIDICT_CAUSE_1_HMD),
                                            hue_feature_name='type', hue_feature_values=df_filtered.type.unique(),
                                            types=df_filtered.type.unique(), 
                                            years=range(MIN_YEAR_TRAIN, MAX_YEAR_TEST+1),
                                            year_separators=[MAX_YEAR_TRAIN],
                                            title_fig=title,
                                            is_fig_saved = False,
                                            is_true_dotted=False,                                    
                                            y_limit = [np.exp(y_min_log), np.exp(y_max_log)],
                                            col_palette=sns.color_palette("tab10", len(df_filtered.type.unique())))        
            st.pyplot(fig)
        else:
            st.write("Please check the checkbox on the left to plot this part.")

        # Plot combined year vs log_mortality (1 sex and 1 cause)
        st.subheader("Year for Specific Sex-Cause", divider = 'blue')
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig, ax = plt.subplots()
        fig.suptitle(f'Year vs Log-Rate {sex}-{cause} ({age})')
        sns.lineplot(data=df_filtered.loc[(df_filtered.sex == sex_code) & (df_filtered.cause==cause_code) & (df_filtered.age == age)], 
                    x='year', y='log_mortality', style='type', hue='type')
        plt.axvline(MAX_YEAR_TRAIN, color='gray', linestyle='--')
        ax.legend(prop={'size': 6})
        st.subheader(f"Year vs Log-Rate for {sex} and {cause}")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        fig.suptitle(f'Year vs Rate {sex}-{cause} ({age})')
        sns.lineplot(data=df_filtered.loc[(df_filtered.sex == sex_code) & (df_filtered.cause==cause_code) & (df_filtered.age == age)], 
                    x='year', y='mortality', style='type', hue='type')
        ax.legend(prop={'size': 6})
        plt.axvline(MAX_YEAR_TRAIN, color='gray', linestyle='--')
        st.subheader(f"Year vs Rate for {sex} and {cause}")
        st.pyplot(fig)

        # Plot focused age vs log_mortality
        # df_pred_focus = dict_pred[model]
        # df_pred_focus['log_mortality'] = np.log(df_pred_focus['mortality'])
        st.subheader("Year for Specific Sex-Cause-Model", divider = 'blue')
        df_pred_focus = df_filtered.loc[df_filtered.type == model]
        df_true = df_all.loc[df_all.type == py_params.TYPE_TRUE]
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig, ax = plt.subplots(ncols=2)  
        fig.suptitle(f"Year vs Log-Rate {sex}-{cause} for all ages ({model} vs Data)")
        ax[0].set_title(f"{model}")
        ax[1].set_title(f"Data")
        sns.lineplot(data=df_pred_focus.loc[(df_pred_focus.sex == sex_code) & (df_pred_focus.cause==cause_code)], 
                        x='year', y='log_mortality', hue='age', ax=ax[0], palette="viridis")
        sns.lineplot(data=df_true.loc[(df_true.sex == sex_code) & (df_true.cause==cause_code)], 
                        x='year', y='log_mortality', hue='age', ax=ax[1], palette="viridis")
        st.subheader(f"Year vs Log-Rate for {sex} and {cause} and {model} model only")
        st.pyplot(fig)
