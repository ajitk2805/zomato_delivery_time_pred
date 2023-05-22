import sys, os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __ini__(self):
        pass

    def predict(self, features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl') 
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred

        except Exception as e:
            logging.info('Error occured at prediction')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Delivery_person_Age,
                 Delivery_person_Ratings,
                 Vehicle_condition,
                 multiple_deliveries,
                 Order_Day,
                 Order_Month,
                 Order_hour,
                 Weather_conditions,
                 Road_traffic_density,
                 Type_of_vehicle,
                 Festival,
                 City
                 ):
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Order_Day = Order_Day
        self.Order_Month = Order_Month
        self.Order_hour = Order_hour
        self.Weather_conditions = Weather_conditions
        self.Road_traffic_density = Road_traffic_density
        self.Type_of_vehicle = Type_of_vehicle
        self.Festival = Festival
        self.City = City

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age': [self.Delivery_person_Age],
                'Delivery_person_Ratings': [self.Delivery_person_Ratings],
                'Vehicle_condition': [self.Vehicle_condition],
                'multiple_deliveries': [self.multiple_deliveries],
                'Order_Day': [self.Order_Day],
                'Order_Month': [self.Order_Month],
                'Order_hour': [self.Order_hour],
                'Weather_conditions': [self.Weather_conditions],
                'Road_traffic_density': [self.Road_traffic_density],
                'Type_of_vehicle': [self.Type_of_vehicle],
                'Festival': [self.Festival],
                'City': [self.City]                  
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe gathered')
            return df
        except Exception as e:
            logging.info('Error occurred at prediction pipeline')
            raise CustomException(e,sys)

        