
import streamlit as st
import pickle 
import os 

def load_all_dfs(folder):
    # Get the list of all fitted models in the folder
    files = os.listdir(folder)
    files = [file for file in files if file.find('df') >= 0]

    # For each file
    model_names = list()
    dfs = list()
    for file in files:    
        # Open the df
        with open(f"{folder}/{file}", "rb") as outfile:
            dfs.append(pickle.load(outfile))

        # Shortened some of the hyperparameters
        model_name = file
        model_name = model_name[model_name.find('deep6')+6:]
        model_name = model_name.replace('bootstrap', 'bs')
        model_name = model_name.replace('batch', 'b')

        # Determine whether it is lmxt or mxt    
        input = ""
        if model_name.find('lmxt') >= 0:
            input = 'lmxt'
        elif model_name.find('mxt') >= 0:
            input = 'mxt'

        # Get the age range
        ages = model_name[model_name.find('age') + 3:]
        ages = ages[0:ages.find("_")]
        if(len(ages) == 3):
            ages = f"00_{ages[1:]}"
        else:
            ages = f"{ages[:2]}_{ages[2:]}"
        
        # Shorten the age range
        model_name = f"{input}_{model_name[0:model_name.find('age')]}{ages}"    
        model_name = model_name.replace('epoch200_', "")
        
        # Add model names to a list
        model_names.append(model_name)

    return [dfs, model_names]


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