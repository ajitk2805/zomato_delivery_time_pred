import sys
import os

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import data_processing, save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')

            # Define columns to be ordinally encoded and scaled
            cat_cols = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle', 'Festival', 'City']
            num_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition','multiple_deliveries', 
                        'Order_Day', 'Order_Month', 'Order_hour']

            # Defining ranking for each ordinal variable
            traffic_map = ['Low', 'Medium', 'High', 'Jam']
            weather_map = ['Sunny', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Fog']
            festival_map = ['No', 'Yes']
            city_map = ['Semi-Urban', 'Urban', 'Metropolitian']
            vehicle_map = ['scooter', 'electric_scooter', 'motorcycle']

            logging.info("Pipeline initiated")
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OrdinalEncoder(categories=[weather_map, traffic_map, vehicle_map, festival_map, city_map])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, num_cols),
                ('cat_pipeline', cat_pipeline, cat_cols)
                ]

            )

            logging.info('Pipeline completed.')

            return preprocessor            

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading of train test data completed')
            logging.info(f'Train df head: \n{train_df.head().to_string}')
            logging.info(f'Test df head: \n{test_df.head().to_string}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()
            target_column = 'Time_taken (min)'
            drop_columns = ['ID', 'Delivery_person_ID', 'Time_Order_picked','Restaurant_latitude','Restaurant_longitude',
                'Delivery_location_latitude','Delivery_location_longitude', 'Type_of_order']
            
            # Updating the test and train dataframes as per EDA by using data_processing function (from utils)
            input_feature_train_df, target_feature_train_df = data_processing(train_df,drop_columns, target_column)
            input_feature_test_df, target_feature_test_df = data_processing(test_df,drop_columns, target_column)

            ## Data Tranaformation using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.info('Error in initiating data transformation')
            raise CustomException(e,sys)
        


