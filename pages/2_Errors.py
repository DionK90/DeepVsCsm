
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import bidict

import py_params
import py_hmd_data
from py_hmd_data import HmdMortalityData, HmdResidual, HmdError

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
def load_res_csm(folder):    
    csm_model_names = ['LC_SVD_00_99', 'LC_SVD_00_85', 'LC_SVD_20_99', 'LC_SVD_20_85',                       
                       'APC_00_99', 'APC_00_85', 'APC_20_99', 'APC_20_85',
                       'RH_00_99', 'RH_00_85', 'RH_20_99', 'RH_20_85',]

    # Load the LC residuals for various age range 
    with open(f"{folder}res_lc_svd_0_99.pickle", "rb") as outfile:
        res_lc_svd_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_lc_svd_0_99.df_true, model_name=None)
        adjust_age_sex_cause(res_lc_svd_0_99.df_res, model_name=None)
        adjust_age_sex_cause(res_lc_svd_0_99.df_pred, model_name=None)
    with open(f"{folder}res_lc_svd_20_99.pickle", "rb") as outfile:
        res_lc_svd_20_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_lc_svd_20_99.df_true, model_name=None)
        adjust_age_sex_cause(res_lc_svd_20_99.df_res, model_name=None)
        adjust_age_sex_cause(res_lc_svd_20_99.df_pred, model_name=None)    
    with open(f"{folder}res_lc_svd_0_85.pickle", "rb") as outfile:
        res_lc_svd_0_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_lc_svd_0_85.df_true, model_name=None)
        adjust_age_sex_cause(res_lc_svd_0_85.df_res, model_name=None)
        adjust_age_sex_cause(res_lc_svd_0_85.df_pred, model_name=None)    
    with open(f"{folder}res_lc_svd_20_85.pickle", "rb") as outfile:
        res_lc_svd_20_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_lc_svd_20_85.df_true, model_name=None)
        adjust_age_sex_cause(res_lc_svd_20_85.df_res, model_name=None)
        adjust_age_sex_cause(res_lc_svd_20_85.df_pred, model_name=None)    

    # Load the RH residuals for various age range
    with open(f"{folder}res_rh_0_99.pickle", "rb") as outfile:
        res_rh_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_rh_0_99.df_true, model_name=None)
        adjust_age_sex_cause(res_rh_0_99.df_res, model_name=None)
        adjust_age_sex_cause(res_rh_0_99.df_pred, model_name=None)    
    with open(f"{folder}res_rh_20_99.pickle", "rb") as outfile:
        res_rh_20_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_rh_20_99.df_true, model_name=None)
        adjust_age_sex_cause(res_rh_20_99.df_res, model_name=None)
        adjust_age_sex_cause(res_rh_20_99.df_pred, model_name=None)
    with open(f"{folder}res_rh_0_85.pickle", "rb") as outfile:
        res_rh_0_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_rh_0_85.df_true, model_name=None)
        adjust_age_sex_cause(res_rh_0_85.df_res, model_name=None)
        adjust_age_sex_cause(res_rh_0_85.df_pred, model_name=None)
    with open(f"{folder}res_rh_20_85.pickle", "rb") as outfile:
        res_rh_20_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_rh_20_85.df_true, model_name=None)
        adjust_age_sex_cause(res_rh_20_85.df_res, model_name=None)
        adjust_age_sex_cause(res_rh_20_85.df_pred, model_name=None)

    # Load the APC residuals for various age range
    with open(f"{folder}res_apc_0_99.pickle", "rb") as outfile:
        res_apc_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_apc_0_99.df_true, model_name=None)
        adjust_age_sex_cause(res_apc_0_99.df_res, model_name=None)
        adjust_age_sex_cause(res_apc_0_99.df_pred, model_name=None)
    with open(f"{folder}res_apc_20_99.pickle", "rb") as outfile:
        res_apc_20_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_apc_20_99.df_true, model_name=None)
        adjust_age_sex_cause(res_apc_20_99.df_res, model_name=None)
        adjust_age_sex_cause(res_apc_20_99.df_pred, model_name=None)
    with open(f"{folder}res_apc_0_85.pickle", "rb") as outfile:
        res_apc_0_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_apc_0_85.df_true, model_name=None)
        adjust_age_sex_cause(res_apc_0_85.df_res, model_name=None)
        adjust_age_sex_cause(res_apc_0_85.df_pred, model_name=None)
    with open(f"{folder}res_apc_20_85.pickle", "rb") as outfile:
        res_apc_20_85 = pickle.load(outfile)
        adjust_age_sex_cause(res_apc_20_85.df_true, model_name=None)
        adjust_age_sex_cause(res_apc_20_85.df_res, model_name=None)
        adjust_age_sex_cause(res_apc_20_85.df_pred, model_name=None)

    dict_csm = {key:value for key, value in zip(csm_model_names, 
                                                [res_lc_svd_0_99, res_lc_svd_0_85, res_lc_svd_20_99, res_lc_svd_20_85,
                                                 res_apc_0_99, res_apc_0_85, res_apc_20_99, res_apc_20_85, 
                                                 res_rh_0_99, res_rh_0_85, res_rh_20_99, res_rh_20_85])}
    return dict_csm, csm_model_names

@st.cache_data
def load_res_deep(folder):    
    # Load the first best deep model for various age ranges
    with open(f"{folder}res_deep6_hmd_cod_1_lmxt_scaled_sig_256.pickle", "rb") as outfile:
        res_deep_1_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_deep_1_0_99.df_res, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_1_0_99.df_true, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_1_0_99.df_pred, model_name=None, is_age_str=False)

    # Load the second best deep model for various age ranges
    with open(f"{folder}res_deep6_hmd_cod_2_lmxt_scaled_sig_256_1.pickle", "rb") as outfile:
        res_deep_2_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_deep_2_0_99.df_res, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_2_0_99.df_true, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_2_0_99.df_pred, model_name=None, is_age_str=False)

    # Load the third best deep model for various age ranges
    with open(f"{folder}res_deep6_hmd_cod_17_lmxt_scaled_sig_4096.pickle", "rb") as outfile:
        res_deep_3_0_99 = pickle.load(outfile)
        adjust_age_sex_cause(res_deep_3_0_99.df_res, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_3_0_99.df_true, model_name=None, is_age_str=False)
        adjust_age_sex_cause(res_deep_3_0_99.df_pred, model_name=None, is_age_str=False)

        
    deep_model_names = ['deep6_1_mxt_256_00_99', 'deep6_1_lmxt_256_00_99', 'deep6_17_lmxt_4096_00_99']    
    dict_deep = {key:value for key, value in zip(deep_model_names, [res_deep_1_0_99, res_deep_2_0_99, res_deep_3_0_99])}
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
res_csm, csm_model_names = load_res_csm(folder_csm)
res_deep, deep_model_names = load_res_deep(folder_deep)
res = res_csm.copy()
res.update(res_deep)

model_names = csm_model_names + deep_model_names

#####################
### Streamlit App ###
#####################
st.title('Neural Networks vs Classical Stochastic Models')

st.sidebar.markdown("# Control for Errors")

# Select models for the errors
models_err = st.sidebar.multiselect("Select models for the errors", 
                                    model_names,
                                    [model_name for model_name in model_names if model_name.find("00_99") >= 0])
st.write(f"Selected models: {models_err}")
cause = st.sidebar.selectbox("Select cause for larger comparison", options=list(py_params.BIDICT_CAUSE_1_HMD.values())[1:])
cause_code = py_params.BIDICT_CAUSE_1_HMD.inverse[cause.lower()]

##############
### Errors ###
##############
# Create various errors
error_types = [py_hmd_data.TYPE_ERROR_MSE]
res_curr = [res[key] for key in models_err]
errors = py_hmd_data.aggregate_errors(residuals=res_curr, res_names=models_err,
                                           by=["sex", "type"],
                                           error_types=error_types)
errors_year = py_hmd_data.aggregate_errors(residuals=res_curr, res_names=models_err,
                                                by=["sex", "year", "type"],
                                                error_types=error_types)
errors_age = py_hmd_data.aggregate_errors(residuals=res_curr, res_names=models_err,
                                               by=["sex", "age", "type"],
                                               error_types=error_types)
errors_cause = py_hmd_data.aggregate_errors(residuals=res_curr, res_names=models_err,
                                                 by=["sex", "cause", "type"],
                                                 error_types=error_types)

st.header("Overall MSE", divider=True)
fig = errors.barplot_sex_type(x='measure', hue='model')
st.pyplot(fig)

st.header("Overall MSE over the age", divider=True)
fig = errors_age.lineplot_sex_type(x='age', hue='model', style='model', filter_cols_str=['measure'], filter_values_str=['mse'])
st.pyplot(fig)

st.header("Overall MSE per cause", divider=True)
fig = errors_cause.barplot_sex_type(x='cause', hue='model', filter_cols_str=['measure'], filter_values_str=['mse'])
st.pyplot(fig)

# Plot comparison for a specific cause
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(12,12), sharex=True, sharey=True)
ax[0][0].set_title(f"Test Male")
ax[0][1].set_title(f"Test Female")
ax[1][0].set_title(f"Train Male")
ax[1][1].set_title(f"Train Female")
fig.suptitle(f"Overall MSE for {cause}")

# st.table(errors_cause.df_error)

for idx_type, each_type in enumerate(['test', 'train']):
    for idx_sex, sex in enumerate(['Male', 'Female']):
        ax[idx_type][idx_sex].tick_params(axis='x', rotation=45)
        is_legend = False
        if(each_type == 'train'):
            is_legend = True
        sns.barplot(x='model', y=errors_cause._col_err, hue='model', 
                    data=errors_cause.df_error.loc[(errors_cause.df_error.type == each_type) & 
                                                   (errors_cause.df_error.sex == idx_sex+1) & 
                                                   (errors_cause.df_error.cause == cause_code),:],
                                                    ax=ax[idx_type][idx_sex], errorbar=None, legend=is_legend)
st.pyplot(fig)

st.header("Overall MSE over the year", divider=True)
fig = errors_year.lineplot_sex_type(x='year', hue='model', style='model', filter_cols_str=['measure'], filter_values_str=['mse'])
st.pyplot(fig)

st.header("DEEP vs CSM Table", divider=True)
st.write(
    """
    Tables to compare the perfomance of various deep models with various classical stochastic models. 
    Comparison should be made for each two lines to see how many times the deep model beat the stochastic models among the 12 subpopulations (all combinations of sex & cause).
    """)
cols_res = ['type', 'sex', 'cause']
dict_df_mse = dict()
age_ranges = ["00_99", "00_85", "20_99", "20_85"]


# For each deep model
for deep_name in deep_model_names:
    # Extract the age range in the deep model
    which_age = [deep_name.find(age_range) >= 0 for age_range in age_ranges]
    age_range = [age_range for (age_range, idx) in zip(age_ranges, which_age) if idx]

    # Extract the CSM models with the same age range
    which_csms = [csm_model_name for csm_model_name in csm_model_names if csm_model_name.find(age_range[0]) >= 0]

    df_sum =  pd.DataFrame(columns=['model', 'mean MSE', 'median MSE', 'best'])
    # Compare with each CSM model
    for csm in which_csms:
        _, df_mse_sum = py_hmd_data.compare_test_error([res[csm], res[deep_name]],
                                                            [csm, deep_name],
                                                            cols_res)
        df_sum = pd.concat([df_sum, df_mse_sum]).reset_index(drop=True)
    df_sum['mean MSE'] = df_sum['mean MSE']*10**4
    df_sum['median MSE'] = df_sum['median MSE']*10**4
    df_sum.rename(columns={'mean MSE': 'mean MSE (10^4)', 'median MSE': 'median MSE (10^4)'})
    dict_df_mse[deep_name] = df_sum

    # Display the table
    st.subheader(deep_name)
    st.table(dict_df_mse[deep_name])