import os
import numpy as np
from InsurancePremium.Logging.Logs import log_class
from sklearn.ensemble import GradientBoostingRegressor
from InsurancePremium.DataSplitting.splitting import Split
from math import sqrt
import pickle


class SavingModel:
    """
    Class_Name : SavingModel
    Description: This Class is used for saving the best model.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.s = Split()
        self.folder = '../LogFiles/'
        self.filename = 'final_model_log.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def final_model(self):
        """
        Method_Name : final_model()
        Description : This method is used to create final model by using best parameters obtained while performing Hyperparameter Tuning.
        output      : model
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Creating the final model by using best hyper parameters')
            x_train, x_test, y_train, y_test = self.s.splitdata()
            gbr = GradientBoostingRegressor(n_estimators=2585, min_samples_split=15, min_samples_leaf=32,
                                            max_features=None, max_depth=3, loss='huber', learning_rate=0.3,
                                            alpha=0.7089473684210525)
            gbr.fit(x_train, y_train)
            y_pred = gbr.predict(x_test)

            training_score = gbr.score(x_train, y_train)
            print('Training_accu:-', training_score)
            testing_score = gbr.score(x_test, y_test)
            print('Testing_accu:-', testing_score)

            from sklearn import metrics
            mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
            print(f'MAE:- {mae}')
            rmse = round(sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
            print(f'RMSE:- {rmse}')
            self.log_object.create_log_file('Model Building completed!!!')
            return gbr
        except Exception as e:
            print('Error occurred is: ' + str(e))
            raise e

    def save_model(self):
        """
        Method_Name : save_model()
        Description : This method is used to to save the final model for future prediction.
        output      : Pickle format
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Saving the model for future predictions')
            gbr = self.final_model()
            with open('../BestModel/FinalModel_ForPrediction.pkl', 'wb') as f:
                pickle.dump(gbr, f)
        except Exception as e:
            print('Error occurred is: ' + str(e))
            raise e

