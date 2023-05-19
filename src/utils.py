import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logger import logging


def data_processing(df, drop_columns,target_column ):
        try:
            input_feature_df = df.drop(columns=drop_columns, axis=1)

            input_feature_df.dropna(subset=['multiple_deliveries', 'Time_Orderd'], inplace=True)

            input_feature_df['Order_Date'] = pd.to_datetime(input_feature_df['Order_Date'], dayfirst=True)
            input_feature_df['Order_Day'] = input_feature_df['Order_Date'].dt.dayofweek
            input_feature_df['Order_Month'] = input_feature_df['Order_Date'].dt.month
            input_feature_df['Order_hour'] = input_feature_df['Time_Orderd'].str.split(':').str[0]

            # Dropping order date and time as per EDA
            input_feature_df.drop(['Order_Date','Time_Orderd'], axis=1, inplace=True)
            # Order_hour has some incorrect decimal values. Dropping these values.
            input_feature_df.drop(input_feature_df[input_feature_df['Order_hour'].astype(float) < 1 ].index, inplace=True)

            input_feature_df['Order_hour'] = input_feature_df['Order_hour'].astype(int)

            target_feature_df = input_feature_df[target_column]
            input_feature_df.drop(target_column, axis=1, inplace=True)

            return input_feature_df, target_feature_df
        
        except Exception as e:
            logging.info('Error occurred in data processing function..')
            raise CustomException(e,sys)
        
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
