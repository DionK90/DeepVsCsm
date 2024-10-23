import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from typing import List
from typing import Type
from typing import Callable
import warnings

import py_params

TYPE_DATA_DEATH = 'death'
TYPE_DATA_EXP = 'exposure'
TYPE_DATA_M = 'mortality'
TYPE_DATA_LM = 'log_mortality'

COL_RES = 'res'
COL_ERR = 'err'

TYPE_ERROR_MSE = 'mse'
TYPE_ERROR_MAPE = 'mape'
TYPE_ERROR_MPE = 'mpe'
TYPE_ERROR_MAE = 'mae'
TYPE_ERROR_R2 = "'R2'"
AVAIL_ERRORS = [TYPE_ERROR_MSE, TYPE_ERROR_MAE, TYPE_ERROR_MPE, TYPE_ERROR_MAPE, TYPE_ERROR_R2]

TYPE_RES_DF = 'df'
TYPE_RES_NP = 'np'

class HmdMortalityData():
    """
    A class used to apply various operations to the HMD COD dataset. The dataframe should have the following columns:
    - 'year': the column containing the year when the mortality rates are applied
    - 'sex': the column describing the gender (1 for male, 2 for female, 3 for both) of whom the mortality rates are applied
    - 'cause': the column describing the underlying cause for the mortality rates
    ' 'mXX': the column containing the mortality rates for age XX. The mortality columns in the DataFrame should be sorted from youngest to oldest.

    Some available operations are:
    - get the mortality rates as a numpy ndarray with ages for the rows and years for the columns
    - get the exposures as a numpy ndarray with ages for the rows and years for the columns
    - get the rates, death counts, log rates, or exposures dataframe in a wide format
    - get the rates, death counts, log rates, or exposures dataframe in a long format

    Notes: 
    - the object of this class should be initialized by giving HMD datasets in the form of mortality rates and exposure, both in a wide format
    - there should not be other columns whose name starts with "m" or "e" for rates and exposures dataframe respectively
    """
    def __init__(self, df_hmd_rates, df_hmd_exp):
        """
        Initialize an HmdMortalitydata from a given hmd rates and exposures dataframe
        Args:
            - df_hmd_rates (pd.DataFrame): a pandas DataFrame containing the cause-of-death mortality rates from hmd in wide format for ages.
            - df_hmd_exp (pd.DataFrame): a pandas DataFrame containing the exposure from hmd in wide format for ages. Note that the exposure does not have the 'cause' feature (column)
        """
        # Store the mortality dataframe in a sorted manner
        self.df_hmd_rates = df_hmd_rates.sort_values(by=py_params.COL_CO_YR_SX_CA)
                
        # Extract all unique years in the mortality dataset
        self.years = df_hmd_rates['year'].unique()
        
        # Extract all unique ages or age-groups in the mortality dataset
        self.col_m = [column for column in df_hmd_rates.columns if column.find('m') == 0]
        self.ages = [int(age[1:]) for age in self.col_m]
        self.col_e = [f"e{age}" for age in self.ages]                

        # Determine whether the data is subdivided into age-group or single-year of age
        self.is_age_group = False
        for i in range(1, len(self.ages)):
            if(self.ages[i] - self.ages[i-1] > 1):
                self.is_age_group = True

        # Filter, sort, and store the exposure dataset
        self.df_hmd_exp = df_hmd_exp.loc[df_hmd_exp['year'].isin(self.years)]
        self.df_hmd_exp.sort_values(by=py_params.COL_CO_YR_SX, inplace=True)

        # Extract all other columns (features) aside from the mortality rates (and exposures)
        self.col_others_m = [column for column in df_hmd_rates.columns if column.find('m') != 0]
        self.col_others_e = [column for column in df_hmd_exp.columns if column.find('e') != 0]
    
    def get_mxt(self, year_start, year_end, age_start, age_end, sex, cause, country):
        """
        A function to prepare the data from hmd format to a suitable format for traditional stochastic models in py_model_st module.
        The function will prepare a numpy array object based on the specified years, ages, sex, cause, and country
        Args:
            year_start: the first year of the extracted mortality rates
            year_end: the last year of the extracted mortality rates
            age_start: the first age of the extracted mortality rates
            age_end: the last age of the extracted mortality rates
            sex: the gender of the extracted mortality rates
            cause: the underlying cause of death of the extracted mortality rates
            country: the country of the extracted mortality rates
        Returns:
            a list of 3 elements:
                - a numpy array of the mortality rates with ages in rows and years in columns
                - a numpy array of all ages considered
                - a numpy array of all years considered
        """
        if year_start not in self.years:
            raise ValueError("First year is not in the dataset")
        if year_end not in self.years:
            raise ValueError("Last year is not in the dataset")
        if age_start not in self.ages:
            raise ValueError("First age is not in the dataset")
        if age_end not in self.ages:
            raise ValueError("Last age is not in the dataset")
        
        df_temp =  self.df_hmd_rates.loc[(self.df_hmd_rates['sex'] == sex) & 
                                         (self.df_hmd_rates['cause'] == cause) & 
                                         (self.df_hmd_rates['country'] == country) &
                                         (self.df_hmd_rates['year'] >= year_start) & 
                                         (self.df_hmd_rates['year'] <= year_end),].set_index('year')
        df_temp = df_temp.loc[:,[col for col in self.col_m if int(col[1:]) >= age_start and int(col[1:]) <= age_end]]        
        return [df_temp.to_numpy().T,
                df_temp.index.to_numpy(),
                df_temp.columns.to_numpy()]        
    
    def get_ext(self, year_start, year_end, age_start, age_end, sex, country):
        """
        A function to prepare the data from hmd format to a suitable format for traditional stochastic models in py_model_st module.
        The function will prepare a numpy array object based on the specified years, ages, sex, cause, and country
        Args:
            year_start: the first year of the extracted exposures
            year_end: the last year of the extracted exposures
            age_start: the first age of the extracted exposures
            age_end: the last age of the extracted exposures
            sex: the gender of the extracted exposures
            country: the country of the extracted exposures
        Returns:
            a list of 3 elements:
                - a numpy array of the mortality rates with ages in rows and years in columns
                - a numpy array of all ages considered
                - a numpy array of all years considered
        """
        if year_start not in self.years:
            raise ValueError("First year is not in the dataset")
        if year_end not in self.years:
            raise ValueError("Last year is not in the dataset")
        if age_start not in self.ages:
            raise ValueError("First age is not in the dataset")
        if age_end not in self.ages:
            raise ValueError("Last age is not in the dataset")
        
        df_temp = self.df_hmd_exp.loc[(self.df_hmd_exp['sex'] == sex) & 
                                      (self.df_hmd_exp['country'] == country) &
                                      (self.df_hmd_exp['year'] >= year_start) &
                                      (self.df_hmd_exp['year'] <= year_end)].set_index('year')
        df_temp = df_temp.loc[:,[col for col in df_temp.columns if col[0] == 'e' and int(col[1:]) >= age_start and int(col[1:]) <= age_end]]
        return [df_temp.to_numpy().T,
                df_temp.index.to_numpy(),
                df_temp.columns.to_numpy()]

    def to_wide(self, data_type:str, countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
                years:List[int]=None, ages:List[int]=None, 
                start_year:int=None, end_year:int=None,
                start_age:int=None, end_age:int=None):
        """
        Args:
            data_type (str): the type of data to be extracted ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None
        Return:
            pd.DataFrame: a dataframe filtered by the given lists in a wide format for the ages (there will be 1 column containing the 'type' data for each age)
        """
        # Check whether ranges or the first and last year (or age) are given
        years = self._check_ranges(years, start_year, end_year)
        ages = self._check_ranges(ages, start_age, end_age)                
       
        # If exposure is requested
        if(data_type == TYPE_DATA_EXP):            
            return self._filter(TYPE_DATA_EXP, countries, sexes, causes, years, ages).copy()
        
        # Filter the dataframe according to the given lists
        df_returned = self._filter(TYPE_DATA_M, countries, sexes, causes, years, ages).copy()

        # Filter the ages for further processing (converting to other forms of info)
        filtered_col_m = self.col_m        
        filtered_col_e = self.col_e        
        if(ages != None):
            filtered_col_m = [f"m{age}" for age in ages]
            filtered_col_e = [f"e{age}" for age in ages]
        else:
            ages = self.ages

        # If mortality rates is requested
        if(data_type == TYPE_DATA_M):
            return df_returned
        # If log mortality is requested
        elif(data_type == TYPE_DATA_LM):            
            df_returned.loc[:, filtered_col_m] = np.log(df_returned.loc[:, filtered_col_m])
            df_returned.rename(columns={old:new for (old, new) in zip(filtered_col_m, [f"lm{age}" for age in ages])}, inplace=True)
            return df_returned
        # If the number of deaths is requested
        elif(data_type == TYPE_DATA_DEATH):        
            # Need to ensure that the mortality rates and exposures correspond to each other (both are for the same countries, years, sexes, causes, and ages)    
            # For each country, sex, and cause
            countries = df_returned['country'].unique()
            sexes = df_returned['sex'].unique()
            causes = df_returned['cause'].unique()
            years = df_returned['year'].unique()
            for idx_country in countries:
                for idx_sex in sexes:
                    for idx_cause in causes:                                                
                        df_returned.loc[(df_returned['country'] == idx_country) & 
                                        (df_returned['sex'] == idx_sex) & 
                                        (df_returned['cause'] == idx_cause), filtered_col_m] = df_returned.loc[(df_returned['country'] == idx_country) & 
                                                                                                           (df_returned['sex'] == idx_sex) & 
                                                                                                           (df_returned['cause'] == idx_cause), filtered_col_m].values * self.df_hmd_exp.loc[(self.df_hmd_exp['country'] == idx_country) & 
                                                                                                                                                                                         (self.df_hmd_exp['sex'] == idx_sex), filtered_col_e].values                        
            # rename the columns to d to signifies death
            df_returned.rename(columns={old:new for (old, new) in zip(filtered_col_m, [f"d{age}" for age in ages])}, inplace=True)
            return df_returned
        else:
            return None

    def to_long(self, data_type:str, countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
                years:List[int]=None, ages:List[int]=None, 
                start_year:int=None, end_year:int=None,
                start_age:int=None, end_age:int=None):
        """
        Args:
            data_type (str): the type of data to be extracted ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None
        Return:
            pd.DataFrame: a dataframe filtered by the given lists in a long format
        """
        # Use to_wide to filter and get the dataframe correspond to each data type (exposure, mortality, log mortality, deaths)
        df_returned = self.to_wide(data_type=data_type, countries=countries, sexes=sexes, causes=causes, years=years, ages=ages,
                                   start_year=start_year, end_year=end_year, start_age=start_age, end_age=end_age)
        
        # Determine the name of the column depending on the data type requested
        value_name = data_type        
            
        # Melt the dataframe into long format
        if df_returned is not None:
            return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name=value_name)
        else:
            return None
        
        # # Check whether ranges or the first and last year (or age) are given
        # years = self._check_ranges(years, start_year, end_year)
        # ages = self._check_ranges(ages, start_age, end_age)

        # # If the exposure is requested
        # if(data_type == TYPE_EXP):            
        #     return self._filter(TYPE_EXP, countries, sexes, causes, years, ages).melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='exposure')
        
        # df_returned = self._filter(TYPE_M, countries, sexes, causes, years, ages).copy()
        
        # # If the mortality rate is requested
        # if(data_type == TYPE_M):
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='mortality')
        # # If the log mortality is requested
        # elif(data_type == TYPE_LM):            
        #     df_returned.loc[:, self.col_m] = np.log(df_returned.loc[:, self.col_m])
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='log_mortality')
        # # If the number of deaths is requested
        # elif(data_type == TYPE_DEATH):     
        #     # Need to ensure that the mortality rates and exposures correspond to each other (both are for the same countries, years, sexes, causes, and ages)    
                   
        #     df_returned.loc[:, self.col_m] = self.df_hmd_rates[self.col_m].values * self.df_hmd_exp[self.col_e].values
        #     df_returned.rename(columns={old:new for (old, new) in zip(self.col_m, [f"d{age}" for age in self.ages])}, inplace=True)
        #     return df_returned.melt(id_vars=['country', 'sex', 'cause', 'year'],var_name='age', value_name='death')
        # else:
        #     return None

    def _filter(self, data_type:str, countries:List[str] = None, 
                sexes:List[int] = None, causes:List[int] = None, 
                years:List[int] = None, ages:List[int] = None):
        """
        Function to filter the mortality dataframes based on the given values for each column. Set None if no filter should be applied.
        Args:
            data_type (str): either 'mortality' or 'exposure' to specify which dataframe to be filtered
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included
        Raises:
            ValueError: if one value in any of the five lists do not exist in the original data frame given

        Returns:
            pd.DataFrame: a dataframe filtered by the given lists
        """

        if(data_type==TYPE_DATA_EXP and causes is not None):
            raise ValueError("Exposure data does not have cause.")
        
        if(countries == None):
            countries = self.df_hmd_rates['country'].unique()
        if(sexes == None):
            sexes = self.df_hmd_rates['sex'].unique()            
        if(causes == None):
            causes = self.df_hmd_rates['cause'].unique()
        if(years == None):
            years = self.years
        if(ages == None):
            ages = self.ages
            
        # Check whether the given feature values are all contained in the dataframe
        if(not all(elem in self.ages for elem in ages)):
            raise ValueError("Some ages given are not in the mortality dataframe.")
        if(not all(elem in self.years for elem in years)):
            raise ValueError("Some years given are not in the mortality dataframe.")
        if(not all(elem in self.df_hmd_rates.country.unique() for elem in countries)):
            raise ValueError("Some countries given are not in the mortality dataframe.")
        if(not all(elem in self.df_hmd_rates.cause.unique() for elem in causes)):
            raise ValueError("Some causes given are not in the mortality dataframe.")
        
        # Make columns for the given ages
        if(data_type == TYPE_DATA_M):            
            col_m = [f'm{age}' for age in ages]
            return self.df_hmd_rates.loc[(self.df_hmd_rates['country'].isin(countries)) & 
                                        (self.df_hmd_rates['sex'].isin(sexes)) & 
                                        (self.df_hmd_rates['cause'].isin(causes)) &
                                        (self.df_hmd_rates['year'].isin(years)), self.col_others_m + col_m]
        elif(data_type == TYPE_DATA_EXP):
            col_e = [f'e{age}' for age in ages]
            return self.df_hmd_exp.loc[(self.df_hmd_exp['country'].isin(countries)) & 
                                        (self.df_hmd_exp['sex'].isin(sexes)) & 
                                        (self.df_hmd_exp['year'].isin(years)), self.col_others_e + col_e]
        else:
            raise ValueError("Type can only either 'mortality' or 'exposure'.")
            
    def diff(self, data_other: 'HmdMortalityData', data_type:str,
             countries:List[str]=None, sexes:List[int]=None, causes:List[int]=None, 
             years:List[int]=None, ages:List[int]=None, 
             start_year:int=None, end_year:int=None,
             start_age:int=None, end_age:int=None):
        """
        Find the difference of value (specified through the 'type' parameter) between this and other HmdMortalityData.
        The function include some optional parameter to filter which country, gender, cause, years, and ages to be included in the calculation.

        If the given HmdMortalityData contains different feature values (country, gender, cause, years, and ages), only the difference for features contained in both data is calculated.

        Args:
            data_other (HmdMortalityData): another object of HmdMortalityData
            data_type (str): the type of data to be calculated ('death' for total number of death, 'exposure' for exposure, 'mortality' for mortality rates, 'log_mortality' for log mortality rates). Use provided constants in this module.
            country (List[str]): a list of countries 
            sexes (List[int]): a list of integer indicating the genders to be included (1 for male, 2 for female, 3 for both) 
            causes (List[int]): a list of integer indicating the cause of death to be included
            years (List[int]): a list of integer specifying the years to be included
            ages (List[int]): a list of integer specifying the ages to be included            
            start_year (int): the first year to be included. Only applied when years = None
            end_year (int): the last year to be included. Only applied when years = None
            start_age (int): the first age to be included. Only applied when ages = None
            end_age (int): the last age to be included. Only applied when ages = None

        Returns:
        """        
        # Check whether ranges or the first and last year (or age) are given
        years = self._check_ranges(years, start_year, end_year)
        ages = self._check_ranges(ages, start_age, end_age)

        # Find the smallest set of unfiltered feature values (features with None arguments) between the two dataframes
        if(countries == None):
            this_values = self.df_hmd_rates['country'].unique()
            other_values = data_other.df_hmd_rates['country'].unique()
            countries = list(set(this_values) & set(other_values))
        if(sexes == None):
            this_values = self.df_hmd_rates['sex'].unique()
            other_values = data_other.df_hmd_rates['sex'].unique()
            sexes = list(set(this_values) & set(other_values))          
        if(causes == None):
            this_values = self.df_hmd_rates['cause'].unique()
            other_values = data_other.df_hmd_rates['cause'].unique()
            causes = list(set(this_values) & set(other_values))    
        if(years == None):
            this_values = self.years
            other_values = data_other.years
            years = list(set(this_values) & set(other_values))    
        if(ages == None):
            this_values = self.ages
            other_values = data_other.ages
            ages = list(set(this_values) & set(other_values))    
            
        
        # Filter this data and to ensure only feature values contained in this data is requested            
        df_returned = self._filter(data_type=data_type, countries=countries, sexes=sexes, 
                                causes=causes, years=years, ages=ages).copy()            

        # Filter so that both data include the same features values (country, sex, cause, year, and age)
        df_other = data_other._filter(data_type=data_type,
                                        countries=countries, sexes=sexes, 
                                        causes=causes, years=years, ages=ages).copy()   
                        
        # Sort both data so both have the same order (and thus they correspond to each other)
        df_returned.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        df_other.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
        
        # Take columns that contains the mortality (or exposure) data only
        cols = [col for col in df_returned.columns if col not in self.col_others_m]
        
        # Substract df_returned (this dataframe) with the df_other (the given df)
        df_returned.loc[:, cols] = df_returned.loc[:, cols].values() -  df_other.loc[:, cols].values()
        return df_returned                    
        
    def _check_ranges(self, values:List[int], start_range:int, end_range:int):
        """
        A function to check whether a range or only the starting and ending values are given
        Args:
            arange (List[int]): a list of values in integer
            start_range (int): the starting value of a range
            end_range (int): the ending value of a range
        Returns:
            the argument range if given, a range of values from start_range to end_range if both are given, or None
        """
        # If ranges is not given, but the start and end of the range is given
        if(values is None and start_range is not None and end_range is not None):
            if(start_range > end_range):
                raise ValueError('The beginning of the range should be smalled than the end.')            
            return list(range(start_range, end_range+1))
        # If ranges is given
        elif(values is not None):
            return values
        # If nothing is given
        else:
            return None
        
class HmdResidual():
    """
    A class to calculate and store the residuals (differences) from a given true values and predicted values
    of a HMD dataset. The main purpose of this class is to store the residuals, 
    and provides some functions to convert the residuals into some performance measurements.
    There is a also a function to draw the residual heatmap.
    """
    def __init__(self, df_true_long:pd.DataFrame, df_pred_long:pd.DataFrame, 
                 data_type:str,                 
                 year_train_end:int, year_val_end:int = None, 
                 features_cat:List[str] = py_params.COL_CO_YR_SX_CA):
        """
        Residual is defined as true values - predicted values. This class will help in calculating errors between the true dataframe and a prediction dataframe.
        The two columns must have the same columns, else only the columns in the true dataframe will be kept. Moreover, this function relies on sorting the two dataframe based on some categorical features, then subtract the mortality values between the two.
        Therefore, categorical features that uniquely specify the two dataframes must be given (and ensured that these features exist in both dataframe)
        Args:
            df_true_long(pd.DataFrame): a dataframe with the same structure as HMD-COD mortality dataset containing the true values of mortality (or death or log mortality) in a long format
            df_pred_long(pd.DataFrame): a dataframe with the same structure as HMD-COD mortality dataset containing the predicted values of mortality (or death or log mortality) in a long format
            data_type(str): a string describing the data type contained (see available constants in this module, such as death, mortality, or log mortality) 
            year_train_end (int)
            year_val_end (int)
            features_cat (List[str]): a list of columns' name used to uniquely identify mortality rates in the two dataframes. This features will be used to match the mortality rates between the two dataframes.
        """
        self.df_true = df_true_long
        self.df_pred = df_pred_long
        self.data_type = data_type        
        self._col_res = f"{COL_RES}_{self.data_type}"

        # Calculate the residuals (by sorting first, then applying a matrix operation)
        # self.df_res = self.df_true.copy()
        # self.df_res.join(self.df_pred, on=features_cat)
        # self.df_res[self._col_res] = self.df_res[f"{data_type}_x"] - self.df_res[f"{data_type}_y"]
        # self.df_res.drop(columns=[f"{data_type}_x", f"{data_type}_y"])
        self.df_true.sort_values(by=features_cat, inplace=True)
        self.df_pred.sort_values(by=features_cat, inplace=True)
        self.df_res = self.df_true.copy()        
        self.df_res.loc[:,self._col_res] = self.df_true.loc[:,data_type].values - self.df_pred.loc[:,data_type].values
        self.df_res.drop(columns=[data_type], inplace=True)

        # Add info on which ones are train, valid (if specified) and test data
        self.year_train_end = year_train_end
        self.df_res['type'] = py_params.TYPE_TRAIN
        if(year_val_end is not None):
            self.df_res.loc[(self.df_res.year > year_train_end) & (self.df_res.year <= year_val_end), 'type'] = py_params.TYPE_VAL
            self.df_res.loc[(self.df_res.year > year_val_end), 'type'] = py_params.TYPE_TEST
        else:
            self.df_res.loc[(self.df_res.year > year_train_end), 'type'] = py_params.TYPE_TEST

        

    def residual(self, operation:Callable=None):
        """
        Get the residual dataframe with an option to apply an operation to each residual. 
        Args:
            operation(callable): operation to be done to each residual, such as square or absolute. Give none to get the raw residual.
        Returns:
            A dataframe with HMD-COD variables in a long format containing the residuals.
        """
        df_returned = self.df_res.copy()

        # If no operation is applied, return a copy of the residual dataframe
        if operation is None:
            return df_returned    
        
        # Apply the operation to all residuals
        df_returned.loc[:, self._col_res] = operation(df_returned.loc[:, self._col_res])
        return df_returned
            
    def error(self, by:List[str], error_type:str = TYPE_ERROR_MSE, benchmark:pd.DataFrame=None,
              year_val_end:int = None) -> pd.DataFrame:
        """
        Get the mean errors, grouped by the given features (columns) in the `by` parameters.
        Supported type of errors can be seen in the AVAIL_ERRORS constant in this module.

        When error_type == TYPE_ERROR_R2, the 
        Get the explanation ratio (Euthum et al., 2024) for the given true and predicted values
        during the initialization.
        1 - num / denum,
        num = sum_x_t{ (log(mxt) - log(mxt_hat))^2 }
        denum = sum_x_t{ (log(mxt) - avg_over_t(x))^2 }
        avg_over_t(x) = sum_t(log(mxt))/T

        Args:
            - by(List[str])
            - error_type(str)
            - benchmark(pd.DataFrame)
        Returns:
            A dataframe containing the error, grouped (averaged) according to the given `by` parameters.
        """        
        # Check the error_type        
        if error_type not in AVAIL_ERRORS:
            raise ValueError("Error type is not yet supported. See available constants in the module.")

        # Check whether the "by" parameters match with the columns in the HMD dataset
        if(by is not None):
            if(not(set(by).issubset(self.df_res.columns))):
                warnings.warn(f"Warning: The 'by' parameters must be from the columns of the HMD dataset: {self.df_res.columns}")
            # Remove duplicates
            by = list(set(by))
            # Error when the number of samples after grouping is too small.        
            num_samples = self.df_res.groupby(by=by).count().min().iloc[0]
            if num_samples < 30:
                raise ValueError(f"Too many categorical features, the number of samples per group ({num_samples}) may not be reliable.") 

        # Calculate the errors for each sex and cause
        df_returned = None
        # Case for MSE
        if(error_type == TYPE_ERROR_MSE):
            df_returned = self.residual(np.square)
        # Case for MAE
        elif(error_type == TYPE_ERROR_MAE):
            df_returned = self.residual(np.abs)
        # Case for R2
        elif(error_type == TYPE_ERROR_R2):
            df_returned = self.residual(np.square)
            
            # If benchmark is None, then use average over years as benchmark predictions
            if(benchmark is None):
                # Averaging the true values based on country, sex, cause, and age
                benchmark = self.df_true.copy()
                benchmark = benchmark.groupby(by=['country','sex','cause','age'])[self.data_type].mean().reset_index()                
                benchmark = pd.merge(self.df_true.drop(columns=[self.data_type]), benchmark, how="inner", on=["country", "sex", "cause", "age"])                
            
            # Sort the benchmark so that the log mortality with the stored true values 
            benchmark.sort_values(by=py_params.COL_CO_YR_SX_CA, inplace=True)
            
            # Find the "variation" explained by the benchmark and by the model (the predictions in initialization)
            # This is raw squared residuals for each country, sex, cause, and each age and year
            denum = np.square(self.df_true[self.data_type].values - benchmark[self.data_type].values)
            num = self.residual(np.square)[self._col_res].values

            # Store it inside the dataframe to be returned
            df_returned = self.residual()
            df_returned.loc[:, "num"] = num
            df_returned.loc[:, "denum"] = denum

            # Calculate the "R2" and return.
            # Needs different pattern because this measure is not simply averaging (like the other M**)
            # R2 = 1 - sum(num)/sum(denum)
            # if by is None:
            #     return 1 - df_returned.loc[:, "num"].sum() / df_returned.loc[:, "denum"].sum()
            # else:
            #     df_returned = df_returned.groupby(by=by)[['num', 'denum']].sum().reset_index()
            #     df_returned.loc[:, self._col_res] = 1 - df_returned.num.values/df_returned.denum.values                
            #     df_returned.drop(columns=['num', 'denum'], inplace=True)
            #     return df_returned                                        

        # Case for MPE and MAPE 
        else:        
            df_returned = self.residual()
            df_returned.loc[:, self._col_res] = df_returned.loc[:, self._col_res].values / self.df_true.loc[:, self.data_type].values            
            # special case for MAPE, which is MPE with absolute
            if(error_type == TYPE_ERROR_MAPE):
                df_returned.loc[:, self._col_res] = np.abs(df_returned.loc[:, self._col_res].values)                   
        
        # Separate validation and test set if necessary
        if(year_val_end is not None):        
            df_returned.loc[(df_returned.year > self.year_train_end) & (df_returned.year <= year_val_end), 'type'] = py_params.TYPE_VAL
        else:
            df_returned.loc[(df_returned.year > self.year_train_end), 'type'] = py_params.TYPE_TEST

        # Check if "by" is None (calculate the mean error of the dataframe)
        if(by is None):
            if(error_type == TYPE_ERROR_R2):
                return 1 - df_returned.loc[:, "num"].sum() / df_returned.loc[:, "denum"].sum()
            else:
                return df_returned.loc[:, self._col_res].mean()               

        # group by the squared residuals according to the "by" parameter, and calculate the average for each group
        if(error_type == TYPE_ERROR_R2):
            df_returned = df_returned.groupby(by=by)[['num', 'denum']].sum().reset_index()            
            df_returned.loc[:, self._col_res] = 1 - df_returned.num.values/df_returned.denum.values                            
            df_returned.drop(columns=['num', 'denum'], inplace=True)
            return df_returned
            
        
        return df_returned.groupby(by=by)[self._col_res].mean().reset_index()

    def heatmap(self, sex, cause, ax = None, title=None, cmap=None):   
        

        # Generate some example data
        data = np.random.randn(10, 10)  # Random data between -3 and 3

        if(cmap is None):
            cmap="RdYlBu"
        sns.heatmap(data = self.df_res.loc[(self.df_res.sex == sex) & (self.df_res.cause==cause), ['year', 'age', self._col_res]].pivot(index='age', values=self._col_res, columns='year'),
                    cmap=cmap, ax=ax, center=0)
        if (title is None):
            title = f"Residual {py_params.BIDICT_SEX_1_2[sex]}-{py_params.BIDICT_CAUSE_1_HMD[cause]}"
        if(ax is None):
            plt.title(title,fontsize=10)
            plt.show()     
        else:
            ax.set_title(title, fontsize=10)
           
class HmdError():
    """
    A class to store mean errors from forecasts on HMD dataset using various models.
    The main purpose of the class is to ease collecting forecasting errors from various models,
    and to easily plot the various models for visualization.

    The class is to be used with residuals dataframe generated from HmrResidual class to match with the naming convention of the columns.
    """
    def __init__(self, df_error:pd.DataFrame, model_name:str, measure_name:str, 
                 data_type:str):
        """
        Args:
            df_error (pd.DataFrame):
            model_name (str):
            measure_name (str):
            data_type (str):

        """
        # Error checking for the two parameters
        # if(df_error is None and columns is None):
        #     raise ValueError("One of 'df_error' or 'columns' parameter must be given to determine the required information for the errors.")
        # if(df_error is not None and columns is not None):
        #     if(df_error.columns != columns):
        #         raise ValueError("Both 'df_error' and 'columns' parameter must contain the same columns to determine the required information for the errors.")
        
        # Store the error
        # if(df_error is not None):
        #     self.df_error = df_error
        # else:
        #     self.df_error = pd.DataFrame()
        if(model_name is not None):
            df_error['model'] = model_name
        if(measure_name is not None):
            df_error['measure'] = measure_name
        self.df_error = df_error        
        self.data_type = data_type

        # set the column name for error values
        self._col_err = f"{COL_ERR}_{self.data_type}"
        self.df_error.rename(columns={f"{COL_RES}_{self.data_type}": self._col_err},inplace=True)
        self.columns = list(self.df_error.columns)
        

    def add_model(self, df_error:pd.DataFrame, model_name:str, measure_name:str):
        # Add model information on the given dataframe
        if(model_name is not None):
            df_error['model'] = model_name
        # Add measure (performance measure or mean error type) on the given dataframe
        if(measure_name is not None):
            df_error['measure'] = measure_name

        # Concatenate the given error dataframe to this object
        self.df_error = pd.concat((self.df_error, df_error.rename(columns={f"{COL_RES}_{self.data_type}": self._col_err})), ignore_index=True)
    
    def remove_model(self, model_name:List[str]):
        self.df_error = self.df_error.drop(self.df_error.loc[(self.df_error.model.isin(model_name))].index)

    def lineplot_sex_type(self, x:str, hue:str, style:str,
                          filter_cols_str:List[str]=None, filter_values_str:List[str]=None,
                          filter_cols_num:List[int]=None, filter_values_num:List[int]=None):
        """
        """
        # Create a figure    
        sexes = self.df_error.sex.unique()
        types = ['test', 'train']
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig, axs = plt.subplots(ncols=len(sexes), nrows=len(types), figsize=(12, 10))
        
        # Add a title
        title = f"Errors Across {x}s"
        if(filter_values_str is not None):
            title = title + f"{[filter_str for filter_str in filter_values_str]}"
        if(filter_values_num is not None):
            title = title + f"{[filter_num for filter_num in filter_values_num]}"
        fig.suptitle(title)
        
        # Filter the data according to given filter
        df_temp = self.df_error
        if(filter_cols_str is not None):
            # Check the filter first
            if(len(filter_cols_str) != len(filter_values_str)):
                raise ValueError("The number of values does not match the number of column to filter.")
            for idx, col in enumerate(filter_cols_str):
                df_temp = df_temp.loc[df_temp[col] == filter_values_str[idx], :]
        if(filter_cols_num is not None):
            # Check the filter first
            if(len(filter_cols_num) != len(filter_values_num)):
                raise ValueError("The number of values does not match the number of column to filter.")
            for idx, col in enumerate(filter_cols_num):
                df_temp = df_temp.loc[df_temp[col] == filter_values_num[idx], :]

        # Draw separate plot for each selected cause
        for idx_type, each_type in enumerate(types):
            for idx_sex, each_sex in enumerate(sexes):    
                # Plot the barplot
                if(each_type == "train"):
                    is_legend = True
                else:
                    is_legend = False
                
                # Plot the barplot
                curr_ax = axs[idx_type, idx_sex]
                sns.lineplot(x=x, y=self._col_err, hue=hue, style=style,
                             data=df_temp.loc[(df_temp.type == each_type) & 
                                              (df_temp.sex==each_sex),:],
                             ax=curr_ax, errorbar=None, legend=is_legend)

                # Put information
                curr_ax.set_title(f"{each_type}-{py_params.BIDICT_SEX_1_2[each_sex]}")       
                curr_ax.xaxis.set_major_locator(plt.MaxNLocator(6)) 
        plt.show();
        return fig
    
    def barplot_sex_type(self, x:str, hue:str,
                         filter_cols_str:List[str]=None, filter_values_str:List[str]=None,
                         filter_cols_num:List[int]=None, filter_values_num:List[int]=None):
        """
        Args:
        Returns:
        """
        # Create a figure    
        sexes = self.df_error.sex.unique()
        types = ['test', 'train']
        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        fig, axs = plt.subplots(ncols=len(sexes), nrows=len(types), figsize=(12, 10))
        # Add a title
        title = f"Errors Across {x}s"
        if(filter_values_str is not None):
            title = title + f"{[filter_str for filter_str in filter_values_str]}"
        if(filter_values_num is not None):
            title = title + f"{[filter_num for filter_num in filter_values_num]}"
        fig.suptitle(title)
        
        # Filter the data according to given filter
        df_temp = self.df_error
        if(filter_cols_str is not None):
            # Check the filter first
            if(len(filter_cols_str) != len(filter_values_str)):
                raise ValueError("The number of values does not match the number of column to filter.")
            for idx, col in enumerate(filter_cols_str):
                df_temp = df_temp.loc[df_temp[col] == filter_values_str[idx], :]
        if(filter_cols_num is not None):
            # Check the filter first
            if(len(filter_cols_num) != len(filter_values_num)):
                raise ValueError("The number of values does not match the number of column to filter.")
            for idx, col in enumerate(filter_cols_num):
                df_temp = df_temp.loc[df_temp[col] == filter_values_num[idx], :]

        # Draw separate plot for each selected cause
        for idx_type, each_type in enumerate(types):
            for idx_sex, each_sex in enumerate(sexes):    
                # Plot the barplot
                if(each_type == "train"):
                    is_legend = True
                else:
                    is_legend = False
                curr_ax = axs[idx_type, idx_sex]
                sns.barplot(x=x, y=self._col_err, hue=hue, 
                            data=df_temp.loc[(df_temp.type == each_type) & 
                                             (df_temp.sex==each_sex),:],
                            ax=curr_ax, errorbar=None, legend=is_legend)

                # Put information
                curr_ax.set_title(f"{each_type}-{py_params.BIDICT_SEX_1_2[each_sex]}")       

        plt.show();
        return fig


def aggregate_errors(residuals:List[HmdResidual],res_names:List[str],
                    error_types:List[str], by:List[str]):
    errors = None    
    for error_type in error_types:
        for  idx_res, res in enumerate(residuals):
            if(errors is None):     
                errors = HmdError(res.error(by=by, error_type=error_type),
                                  res_names[idx_res], error_type, data_type=res.data_type)
            else:
                errors.add_model(res.error(by=by, error_type=error_type),
                                 model_name=res_names[idx_res], measure_name=error_type)
            # errors.add_model(res.error(by=by, error_type=error_type),
            #                  model_name=res_names[idx_res], measure_name=error_type)
    return errors

def compare_test_error(residuals:list[HmdResidual],
                   model_names:list[str], cols_res:list[str]):
    # Should assert that all residuals have the same columns, including  _col_res  

    # Get the MSE dataframes
    dfs = [residual.error(by=cols_res) for residual in residuals]
    dfs = [df.loc[df.type == "test"].rename(columns={'res_mortality':f'res_mortality_{model_names[idx]}'}).set_index(cols_res) for idx, df in enumerate(dfs)]
    
    # Combine all MSE dataframes
    df_mse = pd.concat(dfs, axis=1, join="inner").reset_index()

    # Determine the best (smallest) MSE in each record (sub-population)
    col_res = residuals[0]._col_res    
    df_mse['best'] = df_mse[[f"{col_res}_{name}" for name in model_names]].idxmin(axis=1)
    df_mse['best'] = df_mse['best'].str[len(col_res)+1:]
    
    ls_mean = [df_mse[f"{col_res}_{name}"].mean() for name in model_names]
    ls_median = [df_mse[f"{col_res}_{name}"].median() for name in model_names]
    dict_best = df_mse['best'].value_counts().to_dict()
    ls_best = [dict_best[key] for key in model_names]
    
    df_mse_sum = pd.DataFrame({'model': model_names, 'mean MSE': ls_mean,
                               'median MSE': ls_median, 'best': ls_best})
    return df_mse, df_mse_sum
