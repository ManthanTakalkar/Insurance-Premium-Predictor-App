import os
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler
from InsurancePremium.DataPreProcessing.FeatureEngineering import PreProcessor
from InsurancePremium.Logging.Logs import log_class


class Scaler:

    """
    Class_Name : Scaler
    Description: This Class is used to split the Dataset into dependent and independent features and then scaling the dependent features.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Scaling_log.txt'
        self.loader = PreProcessor()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def splitfeatures(self):
        """
        Method_Name : splitfeatures()
        Description : This method is used to split the dataset into dependent and independent features.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Dataset to be splitted into Dependent and Independent Features')
            x = self.loader.renamefeatures().drop('expenses', axis=1)
            y = self.loader.renamefeatures()['expenses']
            self.log_object.create_log_file('Dataset splitted into Dependent and Independent Features')
            return x, y
        except Exception as e:
            self.log_object.create_log_file('The error is :- ' + str(e))
            raise e

    def scalefeatures(self):
        """
        Method_Name : scalefeatures()
        Description : This method is used to scale the dependent features.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Dependent features to be scaled down using RobustScaler')
            rob = RobustScaler()
            x, y = self.splitfeatures()
            x_scaled = pd.DataFrame(rob.fit_transform(x), columns=x.columns)
            self.log_object.create_log_file('Dependent features scaled down using RobustScaler')
            print(x_scaled)
            return x_scaled, y, rob
        except Exception as e:
            self.log_object.create_log_file('The Error is :- ' + str(e))
            raise e

    def save_scalar(self):
        """
        Method_Name : scalefeatures()
        Description : This function is used to save Robust Scaler .
        output      : Pickle format
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Saving the RobustScaler')
            a, b, rob = self.scalefeatures()
            pickle.dump(rob, open('../BestModel/RobustScaler000.pkl', 'wb'))
            self.log_object.create_log_file('RobustScaler Saved in pickle format')

        except Exception as e:
            self.log_object.create_log_file('The Error is :- ' + str(e))
            raise e

