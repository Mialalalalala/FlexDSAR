import copy
import os
import re
from pathlib import Path
import pickle
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to perform the retrieval using XGBoost
def run_retrieval(case):
    site = case['landcover']
    folder_path = Path("test_data")
    filename = folder_path.joinpath(f"{site}_45_test.csv")
    datacube = pd.read_csv(filename)
    # print(datacube.columns)

    final_dataframes = datacube[['a','b','c','vwc']]  
    y = datacube[['a','b','c','vwc']] 

    pp = [] 

    for freq in case['frequencies']:
            pol_string = case['polarizations'][freq]
            inc_list = case['angles'][freq]
            noise = case['noise']
            for inc in inc_list:
                 pol_list = [f"{freq} {pol} {inc}" for pol in pol_string.split('/')]

                 if site in ['Grassland', 'Shrub']:
                     pol_list = [item for item in pol_list if "HV" not in item]
                 
                 print('pol_list',pol_list)

                 pp = pp + pol_list

                 filename = folder_path.joinpath(f"{site}_{inc}_test.csv")
                #  filename = f"{site}_{inc}_test.csv"
                 if Path(filename).exists():
                      datacube = pd.read_csv(filename)
                 else:
                      raise FileNotFoundError(f"Test dataset {filename} not found!")
                 # print('datacube',datacube.columns)
                 selected_df = datacube[pol_list]  
                # print('selected_df',selected_df)
                 final_dataframes = pd.concat([final_dataframes, selected_df], axis=1)

    if site in ['Grassland', 'Shrub']:
         final_dataframes_1 = final_dataframes
         y = y
        #  print('final_dataframes_1 grass',final_dataframes_1)
    elif site in ['Deciduous','Evergreen']:
        hv_cols = [col for col in final_dataframes.columns if "HV" in col]
        final_dataframes_ = final_dataframes[~(final_dataframes[hv_cols] < -45).any(axis=1)]
        y = y[~(final_dataframes[hv_cols] < -45).any(axis=1)]

        final_dataframes_1 = final_dataframes_[(final_dataframes_['vwc'] > 9) & (final_dataframes_['vwc'] < 15)]
        y = y[(final_dataframes_['vwc'] > 9) & (final_dataframes_['vwc'] < 15)]

    np.random.seed(0)

    X_train, X_test_, y_train, y_test_ = train_test_split(final_dataframes_1, y, test_size=0.2, random_state=42)
    X_train = final_dataframes_1
    y_train = y
    
    if site in ['Grassland', 'Shrub']:
        X_test = X_test_
        y_test = y_test_

    if site in ['Deciduous']:
        X_test = X_test_#[(X_test_['vwc'] > 9) & (X_test_['vwc'] < 15)]
        y_test = y_test_#[(X_test_['vwc'] > 9) & (X_test_['vwc'] < 15)]

    if site in ['Evergreen']:
        X_test = X_test_#[(X_test_['vwc'] > 5) & (X_test_['vwc'] < 20)]
        y_test = y_test_#[(X_test_['vwc'] > 5) & (X_test_['vwc'] < 20)]
    
    # add noise
    print('noise',noise)
    print('pp',pp)
    if noise != 0:
        for col in pp:
            gaussian_noise = np.random.normal(loc=0.0, scale=noise, size=X_test.shape[0])
            # Add this noise directly to the column
            X_test[col] = X_test[col] + gaussian_noise
    print('X_test',X_test)

# ############################################################
#     num_est = 30
#     # num_max_depth = 10

    # Load the model
    subfolder = "models"  # Change to your actual subfolder name
    model_filename = f"{site}_{pp}_{noise}_a.pkl"  
    model_filename_a = os.path.join(subfolder, model_filename)  # Construct full path

    if os.path.exists(model_filename_a):
        with open(model_filename_a, "rb") as file:
            model_a = pickle.load(file)
    else:
        # Create an XGBoost regressor model
        raise FileNotFoundError(f"Model {model_filename} not found!")
        # model_a = xgb.XGBRegressor(objective='reg:squarederror')#n_estimators=num_est, max_depth = num_max_depth
        # # Train the model
        # model_a.fit(X_train[pp], y_train['a'])
        # joblib.dump(model_a, model_filename)

    # Load the model
    model_filename = f"{site}_{pp}_{noise}_b.pkl" 
    model_filename_b = os.path.join(subfolder, model_filename)  # Construct full path 
    if os.path.exists(model_filename_b):
        with open(model_filename_b, "rb") as file:
            model_b = pickle.load(file)
    else:
        # Create an XGBoost regressor model
        raise FileNotFoundError(f"Model {model_filename} not found!")
        # model_b = xgb.XGBRegressor(objective='reg:squarederror')#n_estimators=num_est, max_depth = num_max_depth
        # # Train the model
        # model_b.fit(X_train[pp], y_train['b'])
        # joblib.dump(model_b, model_filename)

    # Load the model
    model_filename = f"{site}_{pp}_{noise}_c.pkl" 
    model_filename_c = os.path.join(subfolder, model_filename)  # Construct full path  
    if os.path.exists(model_filename_c):
        with open(model_filename_c, "rb") as file:
            model_c = pickle.load(file)
    else:
        # # Create an XGBoost regressor model
        raise FileNotFoundError(f"Model {model_filename} not found!")
        # model_c = xgb.XGBRegressor(objective='reg:squarederror')#n_estimators=num_est, max_depth = num_max_depth
        # # Train the model
        # model_c.fit(X_train[pp], y_train['c']) 
        # joblib.dump(model_c, model_filename)

    # Load the model
    model_filename = f"{site}_{pp}_{noise}_vwc.pkl"  
    model_filename_vwc = os.path.join(subfolder, model_filename)  # Construct full path
    if os.path.exists(model_filename_vwc):
        with open(model_filename_vwc, "rb") as file:
            model_vwc = pickle.load(file)
    else:
        # # Create an XGBoost regressor model
        raise FileNotFoundError(f"Model {model_filename} not found!")
        # model_vwc = xgb.XGBRegressor(objective='reg:squarederror')#n_estimators=num_est, max_depth = num_max_depth
        # # Train the model
        # model_vwc.fit(X_train[pp], y_train['vwc'])
        # joblib.dump(model_vwc, model_filename)

# ############################################################
    # # Make predictions
    y_pred_a = model_a.predict(X_test[pp])
    # # Evaluate the model
    rmse_a = root_mean_squared_error(y_test['a'], y_pred_a)
    # # Make predictions
    y_pred_b = model_b.predict(X_test[pp])
    # # Evaluate the model
    rmse_b = root_mean_squared_error(y_test['b'], y_pred_b)
    # # Make predictions
    y_pred_c = model_c.predict(X_test[pp])
    # # Evaluate the model
    rmse_c = root_mean_squared_error(y_test['c'], y_pred_c)
    #  # Make predictions
    y_pred_vwc = model_vwc.predict(X_test[pp])
    # # Evaluate the model
    rmse_vwc = root_mean_squared_error(y_test['vwc'], y_pred_vwc)

    print('rmse_a',rmse_a,'rmse_b',rmse_b,'rmse_c',rmse_c)

# ##########################second order poly soil moisture profile##################################
    X_test['sm0'] = X_test['c']
    X_test['sm10'] = X_test['a']*0.1*0.1+X_test['b']*0.1+X_test['c']
    X_test['sm20'] = X_test['a']*0.2*0.2+X_test['b']*0.2+X_test['c']
    X_test['sm30'] = X_test['a']*0.3*0.3+X_test['b']*0.3+X_test['c']
    X_test['sm40'] = X_test['a']*0.4*0.4+X_test['b']*0.4+X_test['c']
    X_test['sm50'] = X_test['a']*0.5*0.5+X_test['b']*0.5+X_test['c']
    
    X_test['sm0_re'] = y_pred_c
    X_test['sm10_re'] = y_pred_a*0.1*0.1+y_pred_b*0.1+y_pred_c
    X_test['sm20_re'] = y_pred_a*0.2*0.2+y_pred_b*0.2+y_pred_c
    X_test['sm30_re'] = y_pred_a*0.3*0.3+y_pred_b*0.3+y_pred_c
    X_test['sm40_re'] = y_pred_a*0.4*0.4+y_pred_b*0.4+y_pred_c
    X_test['sm50_re'] = y_pred_a*0.5*0.5+y_pred_b*0.5+y_pred_c
    X_test['vwc_re'] = y_pred_vwc
    
    X_test_screen = X_test[(X_test['sm0_re'] > 0) &(X_test['sm10_re'] > 0) & (X_test['sm20_re'] > 0) & (X_test['sm50_re'] >0) & (X_test['vwc_re'] >0) &
                           (X_test['sm0_re'] < 0.5) &(X_test['sm10_re'] < 0.5) & (X_test['sm20_re'] <0.5) & (X_test['sm50_re'] <0.5)]
    # print(df_valid)

    rmse_sm = root_mean_squared_error(np.array([X_test['sm0'],X_test['sm10'],X_test['sm30'],X_test['sm50']]).flatten(), 
                                     np.array([X_test['sm0_re'],X_test['sm10_re'],X_test['sm30_re'],X_test['sm50_re']]).flatten())
    rmse_sm0 = root_mean_squared_error(X_test_screen['sm0'], 
                                         X_test_screen['sm0_re'])
    rmse_sm1 = root_mean_squared_error(X_test_screen['sm10'], 
                                         X_test_screen['sm10_re'])
    rmse_sm2 = root_mean_squared_error(X_test_screen['sm20'], 
                                         X_test_screen['sm20_re'])
    rmse_sm3 = root_mean_squared_error(X_test_screen['sm30'], 
                                         X_test_screen['sm30_re'])
    rmse_sm4 = root_mean_squared_error(X_test_screen['sm40'], 
                                         X_test_screen['sm40_re'])
    rmse_sm5 = root_mean_squared_error(X_test_screen['sm50'], 
                                         X_test_screen['sm50_re'])
    rmse_vwc = root_mean_squared_error(X_test_screen['vwc'], 
                                        X_test_screen['vwc_re'])
    
#     # print('X-test',X_test)
#     print('rmse_sm1,rmse_sm2,rmse_sm3,rmse_vwc',rmse_sm1,rmse_sm2,rmse_sm3,rmse_vwc)
    
    return rmse_sm0,rmse_sm1,rmse_sm3,rmse_sm5,rmse_sm,rmse_vwc,#rmse_sm2,rmse_sm4,rmse_a,rmse_b,rmse_c