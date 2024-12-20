import asizeof
import pandas as pd
import numpy as np

import streamlit as st
import pickle 
import os 

import py_utils_general
import py_params
import py_hmd_data

from py_hmd_data import HmdMortalityData, HmdResidual, HmdError

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
    # csm_model_names = ['LC_SVD_00_99', 'LC_SVD_00_85', 'LC_SVD_20_99', 'LC_SVD_20_85',                       
    #                    'APC_00_99', 'APC_00_85', 'APC_20_99', 'APC_20_85',
    #                    'RH_00_99', 'RH_00_85', 'RH_20_99', 'RH_20_85',]
    # csm_model_names = ['LC_SVD_00_99',
    #                    'APC_00_99', 
    #                    'RH_00_99']
    csm_model_names = ['LC_SVD_00_99', 'LC_SVD_00_85', 'LC_SVD_20_85',
                       'APC_00_99', 'APC_00_85', 'APC_20_85',
                       'RH_00_99', 'RH_00_85', 'RH_20_85',]

    # Load the LC predictions for various age ranges
    with open(f"{folder}data_hmd_lc_svd_0_99.pickle", "rb") as outfile:
        hmd_pred_lc_svd_0_99 = pickle.load(outfile)
    # with open(f"{folder}data_hmd_lc_svd_20_99.pickle", "rb") as outfile:
    #     hmd_pred_lc_svd_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_lc_svd_0_85.pickle", "rb") as outfile:
        hmd_pred_lc_svd_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_lc_svd_20_85.pickle", "rb") as outfile:
        hmd_pred_lc_svd_20_85 = pickle.load(outfile)

    # Load the APC predictions for various age ranges
    with open(f"{folder}data_hmd_apc_0_99.pickle", "rb") as outfile:
        hmd_pred_apc_0_99 = pickle.load(outfile)
    # with open(f"{folder}data_hmd_apc_20_99.pickle", "rb") as outfile:
    #     hmd_pred_apc_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_apc_0_85.pickle", "rb") as outfile:
        hmd_pred_apc_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_apc_20_85.pickle", "rb") as outfile:
        hmd_pred_apc_20_85 = pickle.load(outfile)

    # Load the RH predictions for various age ranges
    with open(f"{folder}data_hmd_rh_0_99.pickle", "rb") as outfile:
        hmd_pred_rh_0_99 = pickle.load(outfile)
    # with open(f"{folder}data_hmd_rh_20_99.pickle", "rb") as outfile:
    #     hmd_pred_rh_20_99 = pickle.load(outfile)
    with open(f"{folder}data_hmd_rh_0_85.pickle", "rb") as outfile:
        hmd_pred_rh_0_85 = pickle.load(outfile)
    with open(f"{folder}data_hmd_rh_20_85.pickle", "rb") as outfile:
        hmd_pred_rh_20_85 = pickle.load(outfile)

    # Extract the prediction dataframe for the LC_SVD models
    df_pred_lc_svd_0_99 = hmd_pred_lc_svd_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_0_99, 'LC_SVD_00_99')
    df_pred_lc_svd_0_85 = hmd_pred_lc_svd_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_0_85, 'LC_SVD_00_85')
    # df_pred_lc_svd_20_99 = hmd_pred_lc_svd_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    # adjust_age_sex_cause(df_pred_lc_svd_20_99, 'LC_SVD_20_99')
    df_pred_lc_svd_20_85 = hmd_pred_lc_svd_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_lc_svd_20_85, 'LC_SVD_20_85')

    # Extract the prediction dataframe for the APC models
    df_pred_apc_0_99 = hmd_pred_apc_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_0_99, 'APC_00_99')
    df_pred_apc_0_85 = hmd_pred_apc_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_0_85, 'APC_00_85')
    # df_pred_apc_20_99 = hmd_pred_apc_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    # adjust_age_sex_cause(df_pred_apc_20_99, 'APC_20_99')
    df_pred_apc_20_85 = hmd_pred_apc_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_apc_20_85, 'APC_20_85')

    # Extract the prediction dataframe for the RH models
    df_pred_rh_0_99 = hmd_pred_rh_0_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_0_99, 'RH_00_99', is_cause_zero=False, is_sex_zero=False)    
    df_pred_rh_0_85 = hmd_pred_rh_0_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_0_85, 'RH_00_85', is_cause_zero=False, is_sex_zero=False)
    # df_pred_rh_20_99 = hmd_pred_rh_20_99.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    # adjust_age_sex_cause(df_pred_rh_20_99, 'RH_20_99', is_cause_zero=False, is_sex_zero=False)
    df_pred_rh_20_85 = hmd_pred_rh_20_85.to_long(data_type=py_hmd_data.TYPE_DATA_M)
    adjust_age_sex_cause(df_pred_rh_20_85, 'RH_20_85', is_cause_zero=False, is_sex_zero=False)
        
    # Make a dictionary of predictions df to be returned
    # dict_csm = {key:value for key, value in zip(csm_model_names, 
    #                                             [df_pred_lc_svd_0_99, df_pred_lc_svd_0_85, df_pred_lc_svd_20_99, df_pred_lc_svd_20_85,
    #                                              df_pred_apc_0_99, df_pred_apc_0_85, df_pred_apc_20_99, df_pred_apc_20_85, 
    #                                              df_pred_rh_0_99, df_pred_rh_0_85, df_pred_rh_20_99, df_pred_rh_20_85])}
    # dict_csm = {key:value for key, value in zip(csm_model_names, 
    #                                             [df_pred_lc_svd_0_99,
    #                                              df_pred_apc_0_99,
    #                                              df_pred_rh_0_99,])}
    dict_csm = {key:value for key, value in zip(csm_model_names, 
                                                [df_pred_lc_svd_0_99, df_pred_lc_svd_0_85, df_pred_lc_svd_20_85,
                                                 df_pred_apc_0_99, df_pred_apc_0_85, df_pred_apc_20_85, 
                                                 df_pred_rh_0_99, df_pred_rh_0_85, df_pred_rh_20_85])}

    # Minimize the memory consumption of each df
    for key in dict_csm.keys():
        if 'country' in dict_csm[key].columns:
            cols_drop = ['country']
        else:
            cols_drop = []
        dict_csm[key] = py_hmd_data.minimize_df(dict_csm[key], cols_drop=cols_drop, cols_int=['age', 'sex', 'cause', 'year'])

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

    # Model from Best 
    if("Best" in subfolders):        
        # Load the files
        folder_name = f"{folder}Best/"

        dfs_best, model_names_best = py_utils_general.load_all_files(folder_name, 'df')

        # Adjust the data type and ranges
        for idx in range(0, len(dfs_best)):
            adjust_age_sex_cause(dfs_best[idx], model_names_best[idx], is_age_str=False)        

        # Add them to the collections to be returned
        deep_model_names = deep_model_names + model_names_best
        dfs = dfs + dfs_best

    # # Model from HEC
    # if("HEC" in subfolders):        
    #     # Load the files
    #     folder_name = f"{folder}HEC/"

    #     dfs_hec, model_names_hec = py_utils_general.load_all_files(folder_name, 'df')

    #     # Adjust the data type and ranges
    #     for idx in range(0, len(dfs_hec)):
    #         adjust_age_sex_cause(dfs_hec[idx], model_names_hec[idx], is_age_str=False)        

    #     # Add them to the collections to be returned
    #     deep_model_names = deep_model_names + model_names_hec
    #     dfs = dfs + dfs_hec

    # # Model from DELL
    # if("Dell" in subfolders):
    #     # Load the files
    #     folder_name = f"{folder}Dell/"
    #     dfs_dell, model_names_dell = py_utils_general.load_all_files(folder_name, 'df')

    #     # Adjust the data type and ranges
    #     for idx in range(0, len(dfs_dell)):
    #         adjust_age_sex_cause(dfs_dell[idx], model_names_dell[idx], is_age_str=False)        
        
    #     # Add them to the collections to be returned
    #     deep_model_names = deep_model_names + model_names_dell
    #     dfs = dfs + dfs_dell
    
    dict_deep = {key:value for key, value in zip(deep_model_names, dfs)}
    # Minimize the memory consumption of each df
    for key in dict_deep.keys():        
        if 'country' in dict_deep[key].columns:
            cols_drop = ['country']
        else:
            cols_drop = []
        dict_deep[key] = py_hmd_data.minimize_df(dict_deep[key], cols_drop=cols_drop, cols_int=['age', 'sex', 'cause', 'year'])

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

st.set_page_config(
    page_title="Results",
    page_icon="👋",
)

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a page below.")

st.markdown(
    """
    Select Predictions to see plot for predictions from various models.
    Select Errors to compare overall errors among various models
    """
)


####################################
### Load Dataset and Predictions ###
####################################
if 'df_all' not in st.session_state:
    folder_data = "./"
    folder_csm = "./Results Intermediate/CSM models/"
    folder_deep = "./Results Intermediate/DEEP models/"

    _, _, _, df_m_long, _ = load_data(folder_data)
    dict_pred, csm_model_names =  load_pred_csm(folder_csm)
    dict_pred_deep, deep_model_names =  load_pred_deep(folder_deep)
    dict_pred.update(dict_pred_deep)

    print("CSM df preds size: ",asizeof.asizeof(dict_pred)/1000000, " MB")
    print("Deep df preds size: ",asizeof.asizeof(dict_pred_deep)/1000000, " MB")

    model_names = csm_model_names + deep_model_names

    df_all = pd.concat([df_m_long.assign(type=py_params.TYPE_TRUE)] + list(dict_pred.values()))
    df_all['log_mortality'] = np.log(df_all['mortality'])
    y_min_log = df_all['log_mortality'].min()
    y_max_log = df_all['log_mortality'].max()    

    st.session_state.df_all = df_all
    st.session_state.model_names = model_names
    st.session_state.y_min_log = y_min_log
    st.session_state.y_max_log = y_max_log
    st.write("Data loaded successfully")
else:
    st.write("Data has been loaded previously")