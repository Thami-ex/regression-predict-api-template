"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    # Load the data
    riders = pd.read_csv("./regression data/Riders.csv")
    test = pd.read_csv("./regression data/Test.csv")
    train = pd.read_csv("./regression data/Train.csv")
    variableDefinitions= pd.read_csv("./regression data/VariableDefinitions.csv")
    
    # Checking missing values
    train.isnull().sum()
    
    # Filling the missing values in values Temperature column with the on both the train and test data
    train['Temperature'] = train['Temperature'].fillna( train['Temperature'].mean())
    test['Temperature'] = test['Temperature'].fillna( test['Temperature'].mean())
    
    # Proportion of missing values in the Precipitation in millimeters column
    missing_vals = train['Precipitation in millimeters'].isnull().sum()
    round((missing_vals/len(train.index))*100,0)
    
    precipitation = train['Precipitation in millimeters'].copy()
    precipitation.dropna(inplace = True)

    # We want to check if whether the available records we have contain Zeros for when 
    # there was no rainfall/precipitation at the time of the delivery.
    precipitation[precipitation==0].count()
    
    # Fillinh missing values in Precipitation column with 0 on both train and test data
    train['Precipitation in millimeters'] = train['Precipitation in millimeters'].fillna(0)
    test['Precipitation in millimeters'] = test['Precipitation in millimeters'].fillna(0)
    
    # Checking if all missing values have been handled
    train.isnull().sum()
    
    train = train[['Order No', 'User Id', 'Vehicle Type', 'Platform Type',
       'Personal or Business', 'Placement - Day of Month',
       'Placement - Weekday (Mo = 1)', 'Placement - Time',
       'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
       'Confirmation - Time', 'Arrival at Pickup - Day of Month',
       'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
       'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time',
       'Distance (KM)', 'Temperature', 'Precipitation in millimeters',
       'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'Rider Id','Time from Pickup to Arrival']]
    
    # Build a dictionary of correlation coefficients and p-values
    dict_cp = {}

    column_titles = [col for col in corrs.index if col!= 'Time from Pickup to Arrival']
    for col in column_titles:
        p_val = round(pearsonr(train[col], train['Time from Pickup to Arrival'])[1],6)
        dict_cp[col] = {'Correlation_Coefficient':corrs[col],
                        'P_Value':p_val}

    df_cp = pd.DataFrame(dict_cp).T
    df_cp_sorted = df_cp.sort_values('P_Value')
    df_cp_sorted[df_cp_sorted['P_Value']<0.1]
    
    #dropping highly correlated predictors and the ones that were not selected above
    train = train.drop(['Placement - Weekday (Mo = 1)', 'Placement - Weekday (Mo = 1)','Confirmation - Day of Month',
                        'Confirmation - Weekday (Mo = 1)','Arrival at Pickup - Day of Month','Arrival at Pickup - Weekday (Mo = 1)',
                        'Pickup - Day of Month','Pickup - Weekday (Mo = 1)'], axis = 1)

    test = test.drop(['Placement - Weekday (Mo = 1)', 'Placement - Weekday (Mo = 1)','Confirmation - Day of Month',
                      'Confirmation - Weekday (Mo = 1)','Arrival at Pickup - Day of Month','Arrival at Pickup - Weekday (Mo = 1)',
                      'Pickup - Day of Month','Pickup - Weekday (Mo = 1)'], axis = 1)

    #dropping the irrelevant columns 
    train = train.drop(['User Id','Vehicle Type','Rider Id', 'Confirmation - Time', ], axis = 1)

    test = test.drop(['User Id','Vehicle Type','Rider Id', 'Confirmation - Time'], axis = 1)
    
    train.drop(['Placement - Time','Arrival at Pickup - Time','Pickup - Time'], axis = 1, inplace = True)
    test.drop(['Placement - Time','Arrival at Pickup - Time','Pickup - Time'], axis = 1, inplace = True)
    
    test_df = pd.get_dummies(test.iloc[:,1:], drop_first= True)
    train_df = pd.get_dummies(train.iloc[:,1:], drop_first=True)

    
    predict_vector = train_df
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
