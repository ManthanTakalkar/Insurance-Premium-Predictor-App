import os
import numpy as np
from InsurancePremium.Logging.Logs import log_class
from InsurancePremium.DataSplitting.splitting import Split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV


class Model:

    """
    Class_Name : Model
    Description: This Class is used to train the model.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'ModelBuilding_log.txt'
        self.loader = Split()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def data_access(self):
        """
        Method_Name : data_access
        Description : This method is used to access the training and testing data
        Output      : Dataframe
        On_Failure  : Raise Exception
        Written By  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """

        try:
            self.log_object.create_log_file("Accessing Training Dataset And Testing Dataset")
            x_train, x_test, y_train, y_test = self.loader.splitdata()
            self.log_object.create_log_file("Accessed Training Dataset And Testing Dataset")
            return x_train, x_test, y_train, y_test
        except Exception as e:
            self.log_object.create_log_file("The error occurred is:" + str(e))
            raise e

    def pre_tuning_model(self):
        """
        Method_Name : pre_tuning_model
        Description : This method is used to train the machine learning model
        Output      : Model
        On_Failure  : Raise Exception
        Written By  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('started Model Training using Gradient Boosting Regressor...')
            x_train, x_test, y_train, y_test = self.data_access()
            base_model = GradientBoostingRegressor()
            base_model.fit(x_train, y_train)
            self.log_object.create_log_file('Completed Model Training!!!')
            train_accu = base_model.score(x_train, y_train)
            print('Training Accuracy is:- ', train_accu)
            self.log_object.create_log_file(f"R2 Score on Training Data: {str(train_accu)}")
            test_accu = base_model.score(x_test, y_test)
            print(f"Testing accuracy:- {test_accu}")
            self.log_object.create_log_file(f"R2 Score on Testing Data: {str(test_accu)}")
            return base_model
        except Exception as e:
            self.log_object.create_log_file("The error occurred is:" + str(e))
            raise e
