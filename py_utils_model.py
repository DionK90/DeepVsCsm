import matplotlib.figure
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from typing import List
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import torch
import pytorch_lightning as pl
from torchviz import make_dot

import os

import py_params
import py_model
import py_hmd_data
"""
The mortality data used here needs to have the following format:
* Ages (or age group) must be in a wide format.
* Columns can only contain the following:
    1. `year` for the year when the death is recorded in `int`.
    2. `country` for multi-population setting in `int` using the code given by WHO.
    3. `cause` for the underlying cause of death in `int`. The code for this part is intended to be obtained from grouping the various CoD into a smaller category, coded using an integer [1,...]. The code `0` is specifically used for 'all-cause' mortality.
    4. `sex` for the gender in `int`, where `1` is male, `2` is female, and `3` is total.
    5. The rest of the columns must indicate ages or age-groups, sorted from youngest to oldest. All the columns will be used in the modeling, thus filtered must be done in the data preprocessing steps. These columns will contain the **mortality rate, not the death counts**.
"""
class Scaler:
    """
    A class intended to be a collection of scalers fitted to the training set of mortality data.
    The class contains all scalers used in this project, such as:
    1. MinMaxScaler fitted to the mortality data according to sex can cause (based on Adachi Mamiya's Master Thesis)    
    """   
    
    def __init__(self, df: pd.DataFrame,
                 max_year_train: int,
                 mortality_features: List[str]):
        """
        Args:
            df (pd.DataFrame): the mortality data in the format as explained in the beginning of this file.
            max_year_train (int): an integer indicating the maximum year used as training set (assuming the minimum year of the data as the starting point of the training set)
            mortality_features: a list of column names for the log mortality rate, each column specifying the mortality rate of a particular age or age-group
        """
        self.df = df
        self.dict_mamiya_scaler = {}
        for sex in df.sex.unique():
            for cause in df['cause'].unique():
                key = f"{sex}-{cause}"
                scaler = MinMaxScaler()
                self.dict_mamiya_scaler[key] = scaler.fit(df.loc[(df.sex == sex) & 
                                                                 (df.cause == cause) & 
                                                                 (df.year <= max_year_train), 
                                                                 mortality_features].to_numpy().reshape(-1,1))


    def mamiya_transform(self, sex: int, cause:int, data):
        return self.dict_mamiya_scaler[f"{sex}-{cause}"].transform(data)
    def mamiya_inverse_transform(self, sex: int, cause:int, data):
        return self.dict_mamiya_scaler[f"{sex}-{cause}"].inverse_transform(data)
    
    @staticmethod
    def is_mamiya(transform:callable):
        if(transform == None):
            return False
        
        return ((transform.__name__ == Scaler.mamiya_transform.__name__) or 
                (transform.__name__ == Scaler.mamiya_inverse_transform.__name__))

TYPE_RESULT_RES = "res"
TYPE_RESULT_ERR = "err"
TYPE_RESULT_PRED = "pred"
FOLDER_CLASSICAL = "Results Intermediate/Classical_Stoch"

################################################################
### Utility Function to combine predictions into a dataframe ###
################################################################

def multi_channels_recursive_prediction_to_long_df(predictions, loader, 
                                                   features_cat, 
                                                   features_m,
                                                   label_inverse_transform, horizon,
                                                   df_dataset=None,                          
                                                   is_include_true_values=False):
    """
    A function to process multi-channels predictions (i.e. all age-groups simultaneously) into a long DataFrame complete with all its categorical inputs.

    Args:
        predictions (torch.Tensor): predictions made by an ANN in a list of 3D tensor (a list of size total_batches containing batch_size x age_group x horizon)
        loader (DataLoader): DataLoader to load the batch used to make predictions. The loader is used to take (categorical) inputs fed for making the predictions.
        features_cat (List[str]): a list of string specifying the column names for the categorical features.
        df_dataset (pd.DataFrame): a DataFrame containing the true values. If `is_include_true_values` equals True, then data in `df_dataset` will be added to the returned DataFrame as the true value (with a flag variable `type` = 'true')        
        horizon (int): the prediction horizon (how many years ahead are predicted)
        is_include_true_values (bool): a flag indicating whether records with true values should be added to the returned DataFrame
        label_inverse_transform (callable): 
    
    Return:
        A DataFrame object in a long format, where each record contains all categorical variables (including age_group), a `type` column to indicate true or predicted values, and the log_mortality
    """
    predictions_transformed = []
    df_cats = pd.DataFrame(columns=['year'] + features_cat)
    for batch_idx, batch in enumerate(loader):
        for i in range(0, batch['year'].shape[0]):
            # Take the categorical value of each record
            features_cat_values = []
            for cat in features_cat:
                features_cat_values.append(batch[cat][i])
            year =  batch['year'][i]

            # Construct a dictionary
            curr_dict = {'year': year}
            for cat_idx in range(len(features_cat)):
                curr_dict[features_cat[cat_idx]] = features_cat_values[cat_idx]

            # Construct a DataFrame for each record
            df_cats = pd.concat([df_cats, pd.DataFrame.from_dict(curr_dict)], ignore_index=True)
            
            # Inverse transform the predictions if scaled
            if(Scaler.is_mamiya(label_inverse_transform)):
                predictions_transformed.append(label_inverse_transform(sex = batch['sex'][i].numpy()[0],
                                                                       cause = batch['cause'][i].numpy()[0],
                                                                       data = predictions[batch_idx][i].float().numpy()))
            else:
                predictions_transformed.append(predictions[batch_idx][i].float().numpy())

    # Construct a DataFrame for the predictions, 
    # flattening the predictions from 2D of size (22, HORIZON) into 1D of size (22*HORIZON)
    predictions_transformed = np.array(predictions_transformed)
    df_preds = pd.DataFrame(predictions_transformed.reshape(-1, np.multiply(*predictions_transformed[0].shape)))

    # join the predictions and the categorical features
    df_join = df_cats.join(df_preds)

    # Reformat the constructed DataFrame into a long format
    df_preds_long = pd.DataFrame()
    # For each row
    for i in df_join.index:
        # Get the starting year
        predicted_year = df_join.iloc[i]['year']

        # Rename the prediction columns into AGE_GROUP_YEAR
        col_names = [age_group + "_" + str(predicted_year + i) for age_group in features_m for i in range(0, horizon)]
        df_curr_row = df_join.iloc[i].to_frame().transpose()
        df_curr_row.columns = list(df_curr_row.columns[0:3]) + col_names

        # Add the column type to mark this is a prediction made from `predicted_year`
        df_curr_row['type'] = 'predicted_' + str(predicted_year)

        # Melt the dataframe into long format
        df_curr_row = df_curr_row.melt(id_vars=list(df_curr_row.columns[0:3]) + ['type'],
                        value_name='log_mortality')
        
        # Replace the `year` column with the value of YEAR
        # Separate the AGE_GROUP_YEAR into AGE_GROUP and YEAR   
        df_curr_row.drop(columns=['year'], inplace=True)
        df_curr_row = df_curr_row.join(df_curr_row['variable'].str.split('_', n=1, expand=True)
                        .rename(columns={0: "age",1: "year"}))
        df_curr_row.drop(columns=['variable'], inplace=True)
        
        # Combine into one long DataFrame
        df_preds_long = pd.concat([df_preds_long, df_curr_row], ignore_index=True, axis=0)    

    df_preds_long['year'] = df_preds_long['year'].astype(int)

    # Add the original data to the long-formatted dataframe
    if (is_include_true_values) and (type(df_dataset) != None):
        df_true = df_dataset.copy().drop(columns=['country'])

        df_true = df_true.melt(id_vars=['year'] + features_cat,
                               value_name='log_mortality',
                               var_name='age')
        df_true['type'] = 'true'

        df_preds_long = pd.concat([df_true, df_preds_long], axis=0, ignore_index=True)
    
    return df_preds_long

def single_channel_prediction_to_long_df(predictions, loader, 
                                           features_cat, 
                                           label_inverse_transform, 
                                           prediction_type,
                                           col_value_name='log_mortality',
                                           age_start:int = 0,                                           
                                           year_inverse_transform = None,
                                           df_dataset=None,                          
                                           is_include_true_values=False):
    """
    A function to process single-channel non-recursive predictions (i.e. single age-groups without historical data) into a long DataFrame complete with all its categorical inputs.

    Args:
        predictions (torch.Tensor): predictions made by an ANN to the entire loader in a 3D tensor (total_batches x batch_size x 1)
        loader (DataLoader): DataLoader to load the batch used to make predictions. The loader is used to take (categorical) inputs fed for making the predictions.
        features_cat (List[str]): a list of string specifying the column names for the categorical features.
        df_dataset (pd.DataFrame): a DataFrame containing the true values. If `is_include_true_values` equals True, then data in `df_dataset` will be added to the returned DataFrame as the true value (with a flag variable `type` = 'true')        
        is_include_true_values (bool): a flag indicating whether records with true values should be added to the returned DataFrame
        label_inverse_transform (callable): 
    
    Return:
        A DataFrame object in a long format, where each record contains all categorical variables (including age_group), a `type` column to indicate true or predicted values, and the log_mortality
    """

    
    
    df_preds_long = pd.DataFrame(columns=['year'] + features_cat)
    for batch_idx, batch in enumerate(loader):  
        curr_df = pd.DataFrame(columns=['year'] + features_cat)
        # Add info from each categorical column
        for col in features_cat:
            curr_df[col] = batch[col].numpy().squeeze()
        
        # Check whether the year is transformed or not
        if(year_inverse_transform is not None):
            year =  np.int32(np.round(year_inverse_transform(batch['year']).squeeze()))
        else:
            year =  batch['year'].numpy().squeeze()
        curr_df['year'] = year
        
        # Inverse transform the predictions if scaled
        predictions_transformed = list()
        if(Scaler.is_mamiya(label_inverse_transform)):
            for i in range(0, batch['year'].shape[0]):
                predictions_transformed.append(label_inverse_transform(sex = batch['sex'][i].numpy()[0],
                                                                       cause = batch['cause'][i].numpy()[0],
                                                                       data = predictions[batch_idx][i].reshape(-1, 1).float().numpy()).squeeze().astype(np.float32))
            predictions_transformed = np.array(predictions_transformed)
        elif(label_inverse_transform is not None):
            predictions_transformed = label_inverse_transform(predictions[batch_idx]).squeeze()
        else:
            predictions_transformed = predictions[batch_idx].numpy().squeeze()
        curr_df[col_value_name] = predictions_transformed

        df_preds_long = (curr_df.copy() if df_preds_long.empty 
                   else pd.concat([df_preds_long, curr_df], ignore_index=True))
                                    
    df_preds_long[py_params.COL_TYPE] = prediction_type

    # Adjust the age into the real age range (instead of zero-based index)
    adjust_age_to_real(df_preds_long, age_start)

    # Add the predictions, give it a column name 'log_mortality'
    # df_preds_long[col_value_name] = np.array(predictions_transformed).squeeze()
    

    # df_preds_lond = pd.DataFrame(np.array(predictions_transformed).squeeze(), columns=[col_value_name])

    # Join the predictions and the categorical features (since there is only one prediction, the resulting DataFrame is already in the long format)
    # df_preds_long = df_cats.join(df_preds)

    # Add information on what kind of predictions (model or which split is used)
    # df_preds_long[py_params.COL_TYPE] = prediction_type

    # Add the original data to the long-formatted dataframe
    if (is_include_true_values) and (df_dataset is not None):
        df_true = df_dataset.copy()

        df_true = df_true.melt(id_vars=['year'] + features_cat,
                               value_name=col_value_name,
                               var_name=py_params.COL_TYPE)
        df_true['type'] = 'true'

        df_preds_long = pd.concat([df_true, df_preds_long], axis=0, ignore_index=True)
    
    return df_preds_long

######################################################################
### Utility Function for various reasons specific to FCNN ###
######################################################################
# def get_df_predictions_fcnn(model:py_model.Fcnn, model_name:str,
#                             features_cat:List[str],
#                             dm:py_dataset.MortalityUcodLongDataModule,
#                             trainer:pl.Trainer,
#                             seed:int,
#                             label_inverse_transform:callable = None):
#     """
#     A function used to extract the mortality rates predicted by the given model into a dataframe format
#     along with the other categorical variates. The returned dataframe will have predictions for training, validation, and test set.

#     Args:
#         model(py_model.Fcnn): a model of class py_model.Fcnn whose predictions want to be extracted
#         model_name(str): the name of the model (additional information put in a new column called 'type')
#         dm:py_dataset.MortalityUcodLongDataModule: the data loader used to prepare the dataset to be predicted
#         trainer:pl.Trainer: the trainer object used to make predictions on the data loader object
#         seed: the seed used for reproducability
#     """
#     model.eval()

#     ##########################
#     # Get df for training data    
#     # Set seed to make sure the loader has the same randomness
#     seed_everything(seed, True)
#     loader = dm.train_dataloader()

#     # Get the predictions on the training set (set seed to make sure the training data order does not change during enumerating)
#     seed_everything(seed, True)
#     predictions = trainer.predict(model, loader)

#     # Combine the predicted rates with the categorical values (set seed to make sure both info match each other)
#     seed_everything(seed, True)
#     df_long_train = single_channel_prediction_to_long_df(predictions, loader, 
#                                                           features_cat, 
#                                                           label_inverse_transform, 
#                                                           model_name,
#                                                           # f"{model_name}_{py_params.TYPE_TRAIN}",
#                                                           df_dataset=None,                          
#                                                           is_include_true_values=False)
#     ############################
#     # Get df for validation data    
#     # Set seed to make sure the loader has the same randomness
#     seed_everything(seed, True)
#     loader = dm.val_dataloader()

#     # Get the predictions on the training set (set seed to make sure the training data order does not change during enumerating)
#     seed_everything(seed, True)
#     predictions = trainer.predict(model, loader)

#     # Combine the predicted rates with the categorical values (set seed to make sure both info match each other)
#     seed_everything(seed, True)
#     df_long_val = single_channel_prediction_to_long_df(predictions, loader, 
#                                                           features_cat, 
#                                                           label_inverse_transform, 
#                                                           model_name,
#                                                           # f"{model_name}_{py_params.TYPE_VAL}",
#                                                           df_dataset=None,                          
#                                                           is_include_true_values=False)
    
#     ######################
#     # Get df for test data    
#     # Set seed to make sure the loader has the same randomness
#     seed_everything(seed, True)
#     loader = dm.test_dataloader()

#     # Get the predictions on the training set (set seed to make sure the training data order does not change during enumerating)
#     seed_everything(seed, True)
#     predictions = trainer.predict(model, loader)

#     # Combine the predicted rates with the categorical values (set seed to make sure both info match each other)
#     seed_everything(seed, True)
#     df_long_test = single_channel_prediction_to_long_df(predictions, loader, 
#                                                           features_cat, 
#                                                           label_inverse_transform, 
#                                                           model_name, 
#                                                           # f"{model_name}_{py_params.TYPE_TEST}",
#                                                           df_dataset=None,                          
#                                                           is_include_true_values=False)    
#     return [df_long_train, df_long_val, df_long_test]

######################################################################
### Utility Function for various reasons specific to FCNN ###
######################################################################

def calculate_manual_mse(df_true_long:pd.DataFrame, df_pred_long:pd.DataFrame,
                         col_value_name:str='log_mortality',
                         features_cat=py_params.COL_YR_SX_CA_AG) -> np.float64:
    """
    A function used to calculate the MSE of the two DataFrame from the given column `col_value_name`
    Both DataFrame must contain the columns (year, sex, cause, age, log_mortality).

    Args:
        df_true_long (pd.DataFrame): a DataFrame that contains the real log_mortality
        df_pred_long (pd.DataFrame): a DataFrame that contains the predicted log_mortality
        col_value_name (str): a string indicating the name of the column which contains the value from which MSE is calculated
    Returns:
        a num
    """
    min_year = df_pred_long.year.min()
    max_year = df_pred_long.year.max()

    df_pred_long = df_pred_long.sort_values(by=features_cat)
    df_true_long = df_true_long.sort_values(by=features_cat)

    pred_m = df_pred_long.loc[:, col_value_name].reset_index(drop=True)
    true_m = df_true_long.loc[(df_true_long['year'] >= min_year) &
                          (df_true_long['year'] <= max_year),  col_value_name].reset_index(drop=True)    
    return np.mean(np.square(true_m.to_numpy() - pred_m.to_numpy()))

def render_model_architecture(model:pl.LightningModule, model_name:str,
                            dm: pl.LightningDataModule):
    """
    A function to render the model's architecture into a png.
    The image will be saved to the current working directory with the name "{model_name}_architecture.png".

    Args:
        model (py_model.Fcnn): the model whose architecture is to be rendered
        model_name (str): the name of the model
        dm (py_dataset.MortalityUcodLongDataModule): a DataModule object to generate the model's computational graph
    """
    # Set the LightningDataModule
    dm.setup(None)
    loader = dm.test_dataloader()
    batch = iter(loader).__next__()

    # Pass the dummy input through the model to generate a computational graph
    output = model(batch)

    # Generate a visualization of the computational graph
    dot = make_dot(output, params=dict(model.named_parameters()))

    # Save the visualization to a file or display it
    dot.render(f"{model_name}_architecture", format="png", cleanup=True)

def fcnn_load_evaluate_residuals(ckpt_path:str, 
                                 loader:torch.utils.data.DataLoader,
                                 col_value_name:str,
                                 model_name:str,                 
                                 max_year_train:int, max_year_valid:int,
                                 type_error:str,           
                                 label_inverse_transform:callable,
                                 year_inverse_transform:callable,
                                 is_res_mortality:bool=False):
    """
    Return:
        A list of objects as follows:
            - The loaded Torch (FCNN) model
            - A dataframe containing the predictions made by the loaded model
            - A HmdResidual object containing residuals between predictions and true values
            - A dictionary entry for this model with its error in train, validation, and test set
    """
    # Load the model
    loaded_model = py_model.RwFcnn.load_from_checkpoint(ckpt_path)
    
    # Prepare the Trainer 
    trainer = pl.Trainer()
        
    # Get the predictions 
    predictions = trainer.predict(loaded_model, loader)    
    df_pred = single_channel_prediction_to_long_df(predictions=predictions, 
                                                   loader=loader,
                                                   features_cat=loader.dataset.categorical_features.copy(), 
                                                   prediction_type=model_name,
                                                   col_value_name=col_value_name,
                                                   label_inverse_transform=label_inverse_transform,
                                                   year_inverse_transform=year_inverse_transform,
                                                   df_dataset=None, is_include_true_values=False)     
                                                           
    # Get the residuals object    
    df_true = loader.dataset.df_long.rename(columns={py_params.FEATURES_LABEL_NAME: col_value_name})    
    df_true['type'] = py_params.TYPE_TRUE
    if(year_inverse_transform is not None):
        df_true.loc[:,'year'] = (year_inverse_transform(df_true['year'].values[:, None])).round().squeeze()
        df_true['year'] = df_true['year'].astype(int)

    # Make sure both dataframe (predictions and true values) have the same columns
    # return [df_pred, df_true]
    df_pred = standardize_columns(df=df_pred, df_ref=df_true)

    # Check if residuals should be built on mortality or not
    if(is_res_mortality):
        res_value_name = py_hmd_data.TYPE_DATA_M
        if(col_value_name == py_hmd_data.TYPE_DATA_LM):
            df_true['mortality'] = np.exp(df_true['log_mortality'])
            df_true.drop(columns=['log_mortality'], inplace=True)
            df_pred['mortality'] = np.exp(df_pred['log_mortality'])
            df_pred.drop(columns=['log_mortality'], inplace=True)            
    else:
        res_value_name = col_value_name

    res = py_hmd_data.HmdResidual(df_true_long=df_true,
                                  df_pred_long=df_pred,
                                  year_train_end=max_year_train, year_val_end=max_year_valid,
                                  data_type=res_value_name,
                                  features_cat=loader.dataset.categorical_features.copy() + ['year'])
    
    # Get the entry for comparison purpose    
    df_error = res.error(by=['type'], error_type=type_error, year_val_end=max_year_valid)
    df_error = df_error.set_index('type').T    
    df_error.columns.name = None
    df_error.reset_index(inplace=True, drop=True)
    if(max_year_valid is not None):
        df_error = df_error[['train', 'val', 'test']]
    else:
        df_error = df_error[['train', 'test']]
    df_error.insert(0, 'model', model_name)
    return [loaded_model, df_pred, res, df_error]

# def fcnn_predict_residuals_runs(models:list[py_model.RwFcnn], loader:torch.utils.data.DataLoader,
def fcnn_predict_residuals_runs(models:list[pl.LightningModule], loader:torch.utils.data.DataLoader,
                                col_value_name:str,
                                model_name:str,
                                max_year_train:int, max_year_valid:int,
                                label_inverse_transform:callable, year_inverse_transform:callable,
                                type_error:str,
                                age_start:int=0,
                                is_res_mortality:bool=False):
    """
    Args: 

    Return:
        A list of objects as follows:
            - A dataframe containing the predictions made by the loaded model
            - A HmdResidual object containing residuals between predictions and true values
            - A dictionary entry for this model with its error in train, validation, and test set
    """    
    # Get the predictions 
    df_pred = fcnn_predict_runs(models=models, loader=loader,
                                model_name=model_name,
                                col_value_name=col_value_name,
                                age_start=age_start,
                                label_inverse_transform=label_inverse_transform,
                                year_inverse_transform=year_inverse_transform)     
                                                           
    # Process the dataframe with the true values
    df_true = loader.dataset.df_long.rename(columns={py_params.FEATURES_LABEL_NAME: col_value_name})    
    df_true['type'] = py_params.TYPE_TRUE
    adjust_age_to_real(df_true, age_start)
    if(year_inverse_transform is not None):
        df_true.loc[:,'year'] = (year_inverse_transform(df_true['year'].values[:, None])).round().squeeze()
        df_true['year'] = df_true['year'].astype(int)

    # Make sure both dataframe (predictions and true values) have the same columns
    # return [df_pred, df_true]
    df_pred = standardize_columns(df=df_pred, df_ref=df_true)

    # Check if residuals should be built on mortality or not
    if(is_res_mortality):
        res_value_name = py_hmd_data.TYPE_DATA_M
        if(col_value_name == py_hmd_data.TYPE_DATA_LM):
            df_true['mortality'] = np.exp(df_true['log_mortality'])
            df_true.drop(columns=['log_mortality'], inplace=True)
            df_pred['mortality'] = np.exp(df_pred['log_mortality'])
            df_pred.drop(columns=['log_mortality'], inplace=True)            
    else:
        res_value_name = col_value_name

    res = py_hmd_data.HmdResidual(df_true_long=df_true,
                                  df_pred_long=df_pred,
                                  year_train_end=max_year_train, year_val_end=max_year_valid,
                                  data_type=res_value_name,
                                  features_cat=loader.dataset.categorical_features.copy() + ['year'])
    
    # Get the entry for comparison purpose    
    df_error = res.error(by=['type'], error_type=type_error, year_val_end=max_year_valid)
    df_error = df_error.set_index('type').T    
    df_error.columns.name = None
    df_error.reset_index(inplace=True, drop=True)
    if(max_year_valid is not None):
        df_error = df_error[['train', 'val', 'test']]
    else:
        df_error = df_error[['train', 'test']]
    df_error.insert(0, 'model', model_name)
    return [df_pred, res, df_error]

# def fcnn_predict_runs(models:list[py_model.RwFcnn], loader:torch.utils.data.DataLoader,
def fcnn_predict_runs(models:list[pl.LightningModule], loader:torch.utils.data.DataLoader,
                      col_value_name:str,
                      model_name:str,
                      label_inverse_transform:callable, year_inverse_transform:callable,
                      age_start:int = 0):
    """
    A method to make and aggregate predictions made from the given models. The prediction from all models will be averaged.
    Args:
        models(list[py_model.RwFcnn]):
        loader(DataLoader): an object of dataloader which was created using py_dataset.MortalityUcodLongDataset
        col_value_name(str):
        model_name(str):        
        label_invers_transform(callable):
        year_invers_transform(callable):
    """
    # Initialize Trainer
    trainer = pl.Trainer()

    # Get the categorical features from the loader (which contains the dataset)
    features_cat = loader.dataset.categorical_features.copy()

    #For each iteration
    ls_curr_df = list()
    for idx, model in enumerate(models):
        # Make predictions
        predictions = trainer.predict(model, loader)

        # Combine the predicted rates with the categorical values (set seed to make sure both info match each other)
        ls_curr_df.append(single_channel_prediction_to_long_df(predictions=predictions, loader=loader, 
                                                               features_cat=features_cat,
                                                               prediction_type=model_name,
                                                               col_value_name=col_value_name,
                                                               age_start=age_start,
                                                               label_inverse_transform=label_inverse_transform, 
                                                               year_inverse_transform=year_inverse_transform,
                                                               df_dataset=None, is_include_true_values=False)) 
        ls_curr_df[idx]['run']= idx
    df_aggregated = pd.concat(ls_curr_df, ignore_index=True)
    return df_aggregated.groupby(by=features_cat + ['year', 'type']).agg('mean').reset_index().drop(columns='run')

def standardize_columns(df:pd.DataFrame, df_ref:pd.DataFrame):
    """
    Standardize the columns of the first dataframe to match with the referenced dataframe.
    The total number of rows for both dataframe must be the same, and 
    referenced dataframe's columns must be the superset of dataframe's columns
    Args:
        df (pd.DataFrame): a DataFrame object whose columns are a subset of df_ref's
        df_ref (pd.DataFrame): a DataFrame object whose columns are a superset of df's
    Return:
        DataFrame df with additional columns from df_ref, including its values. 
        The values are assigned by first sorting both DataFrame according to intersected columns from both df.
    """
    cols = list(df.columns)
    cols_ref = list(df_ref.columns)

    df = df.sort_values(by=cols)
    df_ref = df_ref.sort_values(by=cols)

    for col in [col for col in cols_ref if col not in cols]:
        df[col] = df_ref[col]
    return df

########################################################
### Utility Function for converting age in the dataframe
########################################################

def adjust_age_to_zero(df:pd.DataFrame,
                       age_start:int):
    df['age'] = df['age']-age_start
    
def adjust_age_to_real(df:pd.DataFrame,
                       age_start:int):
    df['age'] = df['age']+age_start

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

###########################################################
### Utility Function to make loading checkpoints easier ###
###########################################################
def get_models_name(model_names_core:list[str], input_types:list[str],
                    activations:list[str], is_scaled:bool, is_ind:bool):
    """
    Construct model names from the given base(s) `model_names_core` according to the given input_types, activations, whether it is scaled and whether independet CoD is used.
    This function is made only to help naming the model automatically, and will probably be updated depending on what variants are considered in the research.
    Args:
        model_names_core (list[str]): the base(s) of the model name (usually to signify its architecture)
        input_types (list[str]): whether mortality (mxt) or log_mortality (lmxt)
        activations (list[str]): the last activation layer, whether it is sigmoid (sig) or linear (lin)
        is_scaled (bool): whether the target variable (mxt or lmxt) is scaled or not
        is_ind (bool): whether CoDs are considered independently during the modelling or not

    Return:
        A list of model names with additional information on the input_type, the last activation function, whether it is scaled and whether CoDs are considered independently.
    """
    models_name = list()
    # for each activation function
    for activation in activations:
        # for each input type 
        for input_type in input_types:
            # for each model name given
            for model_name_core in model_names_core:
                # Determine whether the model uses scaling for its target variable
                scaled = ""
                if(is_scaled):
                    scaled = "scaled_"
                
                # Determine whether the model is built using independent cause of death
                ind = ""
                if(is_ind):
                    ind = "i_"
                
                # Construct the model name
                models_name.append(f"{model_name_core}_{ind}{input_type}_{scaled}{activation}")
    return models_name

def get_best_ckpts(folder_logs:str, model_names:list[str], version_name:str = None):
    """
    """
    ckpts_best = list()
    # Get the best and last checkpoints for the models given
    for model_name in model_names:    
        # Get the latest version if not given
        if(version_name is None):
            folder_model = f"{folder_logs}/{model_name}/"
            version_name = os.listdir(folder_model)
            version_name = [os.path.join(folder_model, f) for f in version_name] # add path to each file            
            version_name.sort(key=lambda x: os.path.getmtime(x))            
            version_name = version_name[-1][version_name[-1].rindex("/")+1:]            
                
        # Get the second last (the best and the last) checkpoint
        folder_ckpts = f"{folder_logs}/{model_name}/{version_name}/checkpoints/"        
        ckpts = os.listdir(folder_ckpts)
        ckpts.remove('last.ckpt')
        ckpts = [os.path.join(folder_ckpts, f) for f in ckpts] # add path to each file
        ckpts.sort(key=lambda x: os.path.getmtime(x))
        ckpts_best.append(ckpts[-1])
    
    return ckpts_best

def get_best_ckpts_ind(folder_logs:str, model_names:list[str], num_causes:int,
                       version_names:str = None):
    """
    """
    ckpts_best = list()
    # Get the best and last checkpoints for the models given
    for model_name in model_names:    
        # Get the latest version if not given
        if(version_names is None):
            folder_model = f"{folder_logs}/{model_name}/"
            version_names = os.listdir(folder_model)
            version_names = [os.path.join(folder_model, f) for f in version_names] # add path to each file            
            version_names.sort(key=lambda x: os.path.getmtime(x))            
            version_names = [version_name[version_name.rindex("/")+1:] for version_name in version_names[-num_causes:]]     
                
        # Get the second last (the best and the last) checkpoint
        for version_name in version_names:
            folder_ckpts = f"{folder_logs}/{model_name}/{version_name}/checkpoints/"        
            ckpts = os.listdir(folder_ckpts)
            ckpts.remove('last.ckpt')
            ckpts = [os.path.join(folder_ckpts, f) for f in ckpts] # add path to each file
            ckpts.sort(key=lambda x: os.path.getmtime(x))
            ckpts_best.append(ckpts[-1])
    
    return ckpts_best

