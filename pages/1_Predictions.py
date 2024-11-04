
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

##########################################
### Functions to Adjust Mortality Data ###
##########################################

def adjust_age_sex_cause(df:pd.DataFrame,
                         model_name:str,
                         age_start:int = 0,                         
                         is_age_str:bool = True,
                         is_sex_zero:bool = True,
                         is_cause_zero:bool = True):
    """
    Adjust a mortality dataframe (inplace) as follows:
    1. convert age from string (mXX or eXX) to integer, then adjust the index from zero-based to the real age range
    2. convert sex and cause from zero-based index to one-based index (where 0 means all or total)
    3. add column `type` with value `model_name`
    """
    if(is_age_str):
        df.age = df.age.str[1:].astype(int)
    df.age = df.age + age_start
    if((model_name is not None) and (model_name.strip() != "")):
        df['type'] = model_name
    if(is_sex_zero):
        df['sex'] = df['sex'] + 1
    if(is_cause_zero):
        df['cause'] = df['cause'] + 1
        

#########################################
### Function to Load Data and Results ###
#########################################

@st.cache_data
def load_data(folder):
    # Dataset log_mortality
    PATH = f"{folder}HMD_COD_LVL_1_ROUGH (single).csv"
    df_lm = pd.read_csv(PATH)
    # Remove sex 3 (total)
    df_lm = df_lm.loc[df_lm.sex != 3,:]
    # Set sex, cause, and countries values to 0-based
    # df_lm.sex = df_lm.sex-1
    # df_lm.cause = df_lm.cause-1
    df_lm.country = df_lm.country.map(py_params.BIDICT_COUNTRY_HMD)
    # Sort th data frame based on country, year, sex, cause
    df_lm.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
    # Rename and filter the log_mortality data
    df_lm.drop(columns=[col for col in df_lm.columns if (col[0] == 'm' and col not in features_m)], inplace=True)
    df_lm.rename(columns={old:new for old,new in zip(features_m, features_lm)}, inplace=True)

    # Dataset for exposure
    PATH = f"{folder}HMD_COD_LVL_1_EXP (single).csv"
    df_e = pd.read_csv(PATH)
    # Remove sex 3 (total)
    df_e = df_e.loc[df_e.sex != 3,:]
    # Set sex and country values to 0-based
    # df_e.sex = df_e.sex - 1
    df_e.country = df_e.country.map(py_params.BIDICT_COUNTRY_HMD)
    # Sort th data frame based on country, year, sex, cause    
    df_e.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
    df_e = df_e.groupby(by=['country', 'year', 'sex']).min().reset_index().drop(columns=['cause'])
    # Filter the exposure data to contain only ages in the specified range
    df_e.drop(columns=[col for col in df_e.columns if (col[0] == 'e' and col not in features_e)], inplace=True)

    # Dataset for mortality
    df_m = df_lm.copy()
    df_m.loc[:, features_lm] = np.exp(df_m.loc[:, features_lm].to_numpy())
    df_m.rename(columns={old:new for old, new in zip(features_lm, features_m)}, inplace=True)

    # Dataset in HmdData format (for flexibility)
    data_hmd = HmdMortalityData(df_m, df_e)
    df_m_long = data_hmd.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    df_m_long['age'] = df_m_long['age'].str[1:].astype(int)

    return [df_lm, df_m, df_e, df_m_long, data_hmd]

@st.cache_data
def load_pred_csm(folder):        
    csm_model_names = ['LC_SVD_00_99', 'LC_SVD_00_85', 'LC_SVD_20_99', 'LC_SVD_20_85',                       
                       'APC_00_99', 'APC_00_85', 'APC_20_99', 'APC_20_85',
                       'RH_00_99', 'RH_00_85', 'RH_20_99', 'RH_20_85',]

    # Load the LC predictions for various age ranges
    with open(f"{folder}data_hmd_lc_svd_0_99.pickle", "rb") as outfile:
        hmd_pred_lc_svd_0_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_lc_svd_20_99.pickle", "rb") as outfile:
        hmd_pred_lc_svd_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_lc_svd_0_85.pickle", "rb") as outfile:
        hmd_pred_lc_svd_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_lc_svd_20_85.pickle", "rb") as outfile:
        hmd_pred_lc_svd_20_85 = pickle.load(outfile)

    # Load the APC predictions for various age ranges
    with open(f"{folder}data_hmd_apc_0_99.pickle", "rb") as outfile:
        hmd_pred_apc_0_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_apc_20_99.pickle", "rb") as outfile:
        hmd_pred_apc_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_apc_0_85.pickle", "rb") as outfile:
        hmd_pred_apc_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_apc_20_85.pickle", "rb") as outfile:
        hmd_pred_apc_20_85 = pickle.load(outfile)

    # Load the RH predictions for various age ranges
    with open(f"{folder}data_hmd_rh_0_99.pickle", "rb") as outfile:
        hmd_pred_rh_0_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_rh_20_99.pickle", "rb") as outfile:
        hmd_pred_rh_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_rh_0_85.pickle", "rb") as outfile:
        hmd_pred_rh_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_rh_20_85.pickle", "rb") as outfile:
        hmd_pred_rh_20_85 = pickle.load(outfile)

    # Extract the prediction dataframe for the LC_SVD models
    df_pred_lc_svd_0_99 = hmd_pred_lc_svd_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_0_99, csm_model_names[0])
    df_pred_lc_svd_0_85 = hmd_pred_lc_svd_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_0_85, csm_model_names[1])
    df_pred_lc_svd_20_99 = hmd_pred_lc_svd_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_20_99, csm_model_names[2])
    df_pred_lc_svd_20_85 = hmd_pred_lc_svd_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_20_85, csm_model_names[3])

    # Extract the prediction dataframe for the APC models
    df_pred_apc_0_99 = hmd_pred_apc_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_0_99, csm_model_names[4])
    df_pred_apc_0_85 = hmd_pred_apc_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_0_85, csm_model_names[5])
    df_pred_apc_20_99 = hmd_pred_apc_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_20_99, csm_model_names[6])
    df_pred_apc_20_85 = hmd_pred_apc_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_20_85, csm_model_names[7])

    # Extract the prediction dataframe for the RH models
    df_pred_rh_0_99 = hmd_pred_rh_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_0_99, csm_model_names[8], is_cause_zero=False, is_sex_zero=False)
    df_pred_rh_0_85 = hmd_pred_rh_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_0_85, csm_model_names[9], is_cause_zero=False, is_sex_zero=False)
    df_pred_rh_20_99 = hmd_pred_rh_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_20_99, csm_model_names[10], is_cause_zero=False, is_sex_zero=False)
    df_pred_rh_20_85 = hmd_pred_rh_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_20_85, csm_model_names[11], is_cause_zero=False, is_sex_zero=False)
    
    dict_csm = {key:value for key, value in zip(csm_model_names, 
                                                [df_pred_lc_svd_0_99, df_pred_lc_svd_0_85, df_pred_lc_svd_20_99, df_pred_lc_svd_20_85,
                                                 df_pred_apc_0_99, df_pred_apc_0_85, df_pred_apc_20_99, df_pred_apc_20_85, 
                                                 df_pred_rh_0_99, df_pred_rh_0_85, df_pred_rh_20_99, df_pred_rh_20_85])}

    return dict_csm, csm_model_names


@st.cache_data
def load_pred_deep(folder):
    # Load the first best deep model for various age ranges
    with open(f"{folder}df_all_fcnn_hmd_cod_1_mxt_scaled_sig_256.pickle", "rb") as outfile:
        df_deep_1_0_99 = pickle.load(outfile)

    # Load the second best deep model for various age ranges
    with open(f"{folder}df_all_fcnn_hmd_cod_2_lmxt_scaled_sig_256_1.pickle", "rb") as outfile:
        df_deep_2_0_99 = pickle.load(outfile)

    # Load the third best deep model for various age ranges
    with open(f"{folder}df_all_fcnn_hmd_cod_17_lmxt_scaled_sig_4096.pickle", "rb") as outfile:
        df_deep_3_0_99 = pickle.load(outfile)
    
    # Naming the above models
    deep_model_names = ['deep6_1_mxt_256_00_99', 'deep6_1_lmxt_256_00_99', 'deep6_17_lmxt_4096_00_99']
    adjust_age_sex_cause(df_deep_1_0_99, deep_model_names[0], is_age_str=False)
    adjust_age_sex_cause(df_deep_2_0_99, deep_model_names[1], is_age_str=False)
    adjust_age_sex_cause(df_deep_3_0_99, deep_model_names[2], is_age_str=False)
    dfs = [df_deep_1_0_99, df_deep_2_0_99, df_deep_3_0_99]
    
    # Get all immediate subfolders
    subfolders = [f.name for f in os.scandir(folder) if f.is_dir()]

    # Model from HEC    
    if("HEC" in subfolders):        
        # Load the files
        folder_name = f"{folder}HEC/"

        st.write(subfolders)
        st.write(folder_name)
        
        dfs_hec, model_names_hec = py_utils_general.load_all_dfs(folder_name, 'df')

        # Adjust the data type and ranges
        for idx in range(0, len(dfs_hec)):
            adjust_age_sex_cause(dfs_hec[idx], model_names_hec[idx], is_age_str=False)        

        # Add them to the collections to be returned
        deep_model_names = deep_model_names + model_names_hec
        dfs = dfs + dfs_hec

    # Model from DELL
    if("Dell" in subfolders):
        # Load the files
        folder_name = f"{folder}Dell/"
        dfs_dell, model_names_dell = py_utils_general.load_all_dfs(folder_name, 'df')

        # Adjust the data type and ranges
        for idx in range(0, len(dfs_dell)):
            adjust_age_sex_cause(dfs_dell[idx], model_names_dell[idx], is_age_str=False)        
        
        # Add them to the collections to be returned
        deep_model_names = deep_model_names + model_names_dell
        dfs = dfs + dfs_dell
    
    dict_deep = {key:value for key, value in zip(deep_model_names, dfs)}

    return dict_deep, deep_model_names


##########################
### Parameter for data ###
##########################
AGE_START = 0
AGE_END = 99
MIN_YEAR_TRAIN = 1959
MAX_YEAR_TRAIN = 1999
MAX_YEAR_VALID = 1999
MAX_YEAR_TEST = 2017
countries = [py_params.BIDICT_COUNTRY_HMD['USA']]
sexes = [0, 1]
causes = [0, 1, 2, 3, 4, 5]
years = list(range(1959, 2018))
ages = list(range(AGE_START, AGE_END))

# Determine which columns contain the mortality rates and which columns contain categorical features
features_m = [f"m{age}" for age in range(AGE_START, AGE_END+1)]
features_lm = [f"lm{age}" for age in range(AGE_START, AGE_END+1)]
features_e = [f"e{age}" for age in range(AGE_START, AGE_END+1)]

# Determine which columns contain the categorical features
features_cat_wide = ['cause', 'sex']
features_cat_long = ['age', 'cause', 'sex']

####################################
### Load Dataset and Predictions ###
####################################
folder_data = "./"
folder_csm = "./Results Intermediate/CSM models/"
folder_deep = "./Results Intermediate/DEEP models/"

df_lm, df_m, df_e, df_m_long, data_hmd = load_data(folder_data)
dict_pred, csm_model_names =  load_pred_csm(folder_csm)
dict_pred_deep, deep_model_names =  load_pred_deep(folder_deep)
dict_pred.update(dict_pred_deep)

model_names = csm_model_names + deep_model_names

df_all = pd.concat([df_m_long.assign(type=py_params.TYPE_TRUE)] + list(dict_pred.values()))
df_all['log_mortality'] = np.log(df_all['mortality'])
y_min = df_all['log_mortality'].min()
y_max = df_all['log_mortality'].max()

#####################
### Streamlit App ###
#####################
st.title('Neural Networks vs Classical Stochastic Models')

###################
### Predictions ###
###################
st.sidebar.markdown("# Control for Predictions")

# Slider for selecting a year range (1959 to 1999)
year = st.sidebar.slider("Select year", min_value=1959, max_value=1999, value=1970)

# Select models to be plotted
models = st.sidebar.multiselect("Select up to 6 models for the combined plots:", 
                                model_names,
                                [csm_model for csm_model in csm_model_names if csm_model.find("00_99") >= 0] + deep_model_names[0:3])

# Combobox for gender selection
sex = st.sidebar.selectbox("Select gender", options=['Male', 'Female'])
sex_code = py_params.BIDICT_SEX_1_2.inverse[sex.lower()]

# Combobox for cause selection
cause = st.sidebar.selectbox("Select cause", options=list(py_params.BIDICT_CAUSE_1_HMD.values())[1:])
cause_code = py_params.BIDICT_CAUSE_1_HMD.inverse[cause.lower()]

# Select model to focus on
model = st.sidebar.selectbox("Select model to focus on", options = csm_model_names + deep_model_names)

# Slider for selecting an age (0-99)
age = st.sidebar.slider("Select age", min_value=0, max_value=99, value=20)

### Age vs Log-Mortality ###
############################

st.header("Plot Age vs Log-Rate", divider=True)

# Validate selection
if len(models) > 6:
    st.error(f"⚠️ You can only select up to 6 options.")
else:    
    # Plot combined age vs log_mortality (all sexes and causes)    
    df_all = df_all.loc[df_all.type.isin(models + [py_params.TYPE_TRUE])]
    fig, _ = py_utils_general.plot_age_mortality_model(df_all,
                                            row_feature_name='sex', row_feature_values=[1,2], 
                                            row_labels=py_params.BIDICT_SEX_1_2,
                                            col_feature_name='cause', 
                                            col_feature_values=df_all['cause'].unique(),
                                            col_labels=dict(py_params.BIDICT_CAUSE_1_HMD),
                                            value='log_mortality',
                                            years=[year],
                                            types=df_all.type.unique(),
                                            title_fig = f"Age vs Log-Rate for various models ({year} - {'train' if year <= MAX_YEAR_TRAIN else 'test'})",
                                            is_fig_saved=False,
                                            ages=list(df_all['age'].unique()), y_limit=[y_min, y_max],
                                            col_palette=sns.color_palette("tab10", len(df_all.type.unique())))
    st.subheader("Age vs Log-Rate All Sexes and Causes")
    st.pyplot(fig)

        
    # Plot combined age vs log_mortality (1 sex and 1 cause)    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, ax = plt.subplots()
    fig.suptitle(f'Age vs Log-Rate {sex}-{cause} for various models')
    sns.lineplot(data=df_all.loc[(df_all.sex == sex_code) & (df_all.cause==cause_code) & (df_all.year == year)], 
                 x='age', y='log_mortality', style='type', hue='type')
    st.subheader(f"Age vs Log-Rate for {sex} and {cause}")
    st.pyplot(fig)

    # Plot focused age vs log_mortality
    # df_pred_focus = dict_pred[model]
    # df_pred_focus['log_mortality'] = np.log(df_pred_focus['mortality'])    
    df_pred_focus = df_all.loc[df_all.type == model]    
    df_m_long['log_mortality'] = np.log(df_m_long['mortality'])
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_title(f"{model}")
    ax[1].set_title(f"Data")
    fig.suptitle(f"Age vs Log-Rate {sex}-{cause} for all ages ({model} vs Data)")
    sns.lineplot(data=df_pred_focus.loc[(df_pred_focus.sex == sex_code) & (df_pred_focus.cause==cause_code)], 
                       x='age', y='log_mortality', hue='year', ax=ax[0], palette="viridis")
    sns.lineplot(data=df_m_long.loc[(df_m_long.sex == sex_code) & (df_m_long.cause==cause_code)], 
                       x='age', y='log_mortality', hue='year', ax=ax[1], palette="viridis")
    st.subheader(f"Age vs Log-Rate for {sex} and {cause} and {model} model only")
    st.pyplot(fig)

### Year vs Log-Mortality ###
#############################
# Validate selection
st.header("Plot Year vs Log-Rate", divider=True)
if len(models) > 6:
    st.error(f"⚠️ You can only select up to 6 options.")
else:
    # Create figure title
    title = f"Year vs Log-Rate for "
    f"Year vs Log-Rate for various models (age: {age})",
        
    # Plot combined year vs log_mortality (all sexes and causes)
    df_all = df_all.loc[df_all.type.isin(models + [py_params.TYPE_TRUE])]
    fig = py_utils_general.plot_year_value(df_final_long=df_all.loc[df_all.age == age],
                                    value='log_mortality',
                                    row_feature_name='sex', row_feature_values=[1,2], row_labels=dict(py_params.BIDICT_SEX_1_2),
                                    col_feature_name='cause', col_feature_values=list(df_all['cause'].unique()), col_labels = dict(py_params.BIDICT_CAUSE_1_HMD),
                                    hue_feature_name='type', hue_feature_values=df_all.type.unique(),
                                    types=df_all.type.unique(), 
                                    years=range(MIN_YEAR_TRAIN, MAX_YEAR_TEST+1),
                                    year_separators=[MAX_YEAR_TRAIN],
                                    title_fig=title,
                                    is_fig_saved = False,
                                    is_true_dotted=False,                                    
                                    y_limit = [y_min, y_max],
                                    col_palette=sns.color_palette("tab10", len(df_all.type.unique())))
    st.subheader("Year vs Log-Rate All Sexes and Causes")
    st.pyplot(fig)

    # Plot combined year vs log_mortality (1 sex and 1 cause)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, ax = plt.subplots()
    fig.suptitle(f'Year vs Log-Rate {sex}-{cause} for various models')
    sns.lineplot(data=df_all.loc[(df_all.sex == sex_code) & (df_all.cause==cause_code) & (df_all.age == age)], 
                 x='year', y='log_mortality', style='type', hue='type')
    st.subheader(f"Age vs Log-Rate for {sex} and {cause}")
    st.pyplot(fig)

    # Plot focused age vs log_mortality
    # df_pred_focus = dict_pred[model]
    # df_pred_focus['log_mortality'] = np.log(df_pred_focus['mortality'])
    df_pred_focus = df_all.loc[df_all.type == model]
    df_m_long['log_mortality'] = np.log(df_m_long['mortality'])
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    fig, ax = plt.subplots(ncols=2)  
    fig.suptitle(f"Year vs Log-Rate {sex}-{cause} for all ages ({model} vs Data)")
    ax[0].set_title(f"{model}")
    ax[1].set_title(f"Data")
    sns.lineplot(data=df_pred_focus.loc[(df_pred_focus.sex == sex_code) & (df_pred_focus.cause==cause_code)], 
                       x='year', y='log_mortality', hue='age', ax=ax[0], palette="viridis")
    sns.lineplot(data=df_m_long.loc[(df_m_long.sex == sex_code) & (df_m_long.cause==cause_code)], 
                       x='year', y='log_mortality', hue='age', ax=ax[1], palette="viridis")
    st.subheader(f"Year vs Log-Rate for {sex} and {cause} and {model} model only")
    st.pyplot(fig)
