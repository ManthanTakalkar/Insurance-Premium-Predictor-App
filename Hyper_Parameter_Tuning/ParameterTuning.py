import os
import numpy as np
from sklearn import metrics
from math import sqrt
from InsurancePremium.Logging.Logs import log_class
from InsurancePremium.DataSplitting.splitting import Split
from InsurancePremium.Model.model_building import Model
from sklearn.model_selection import RandomizedSearchCV


class Tuning:

    """
    Class_Name : Tuning
    Description: This Class is used for finding the best parameters of the model.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.loader = Model()
        self.splited = Split()
        self.folder = '../LogFiles/'
        self.filename = 'Tuning_log.txt'
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def random_search(self):
        """
        Method_Name : random_search()
        Description : This method is used to search best parameters for Gradient Boosting Regressor.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Tuning Gradient Boosting using Randomized Search cv...')
            x_train, x_test, y_train, y_test = self.splited.splitdata()
            model = self.loader.pre_tuning_model()
            param_grid = {
                'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1],
                'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                'max_features': ['auto', 'sqrt', None],
                'min_samples_leaf': [i for i in range(1, 50, 1)],
                'min_samples_split': [i for i in range(2, 50, 1)],
                'n_estimators': [int(i) for i in np.linspace(100, 3000, 22)],
                'alpha': [a for a in np.linspace(0.1, 0.99, 20)],
                'loss': ['ls', 'lad', 'huber', 'quantile']
            }
            rscv_gbr = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=100, cv=5, n_jobs=7,
                                          random_state=90, verbose=2)
            rscv_gbr.fit(x_train, y_train)
            self.log_object.create_log_file('Tuning Gradient Boosting using Randomized Search CV Completed!!!')

            best_parameters = rscv_gbr.best_params_
            print('Best_Hyperparameters:-\n', best_parameters)
            self.log_object.create_log_file(f"Best Parameters obtained by performing RandomizedSearch_cv:\n {str(best_parameters)}")

            gbr_best = rscv_gbr.best_estimator_
            y_pred = gbr_best.predict(x_test)

            training_score = gbr_best.score(x_train, y_train)
            print('Training_accu:-', training_score )
            self.log_object.create_log_file(f"R2 Score on Training Data: {str(training_score)}")
            testing_score = gbr_best.score(x_test, y_test)
            print('Testing_accu:-', testing_score )
            self.log_object.create_log_file(f"R2 Score on Testing Data: {str(testing_score)}")

            mae = round(metrics.mean_absolute_error(y_test, y_pred), 4)
            print(f'MAE:- {mae}')
            self.log_object.create_log_file(f"Mean Absolute Error(mae) on Testing Data: {str(mae)}")
            rmse = round(sqrt(metrics.mean_squared_error(y_test, y_pred)), 4)
            print(f'RMSE:- {rmse}')
            self.log_object.create_log_file(f"Root Mean Square Error(rmse) on Testing Data: {str(rmse)}")

            return rscv_gbr
        except Exception as e:
            print('Error occurred is: ' + str(e))
            raise e


h = Tuning()
h.random_search()
