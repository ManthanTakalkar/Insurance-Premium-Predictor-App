import os
from sklearn.model_selection import train_test_split
from InsurancePremium.Logging.Logs import log_class
from InsurancePremium.DataScaling.scaling import Scaler


class Split:

    """
    Class_Name : Split
    Description: This Class is used to split the Dataset into dependent and independent features and then scaling the dependent features.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Splitter_log.txt'
        self.loader = Scaler()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def splitdata(self):
        """
        Method_Name : splitdata()
        Description : This method is used to split Dataset into Training Dataset And Testing Dataset.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            x_scaled, y, rob = self.loader.scalefeatures()
            self.log_object.create_log_file('Dataset to be split into Training and Testing Dataset')
            x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=75)
            self.log_object.create_log_file('Dataset splits into Training and Testing Dataset')
            return x_train, x_test, y_train, y_test
        except Exception as e:
            self.log_object.create_log_file('The error occurred is :' + str(e))
            raise e

