import pandas as pd
from InsurancePremium.Logging.Logs import log_class
import os


class LoadingRaw:
    """
    ClassName  : LoadingRaw
    Description: This class is used to acquire/access data that is stored in the csv file
    Written by : Adityaraj Hemant Chaudhari, Manthan Takalkar
    Version    : 0.1
    Revisions  : 0
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'DataAcquisition_log.txt'

        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def get_data(self):
        """
        Method_Name: get_data
        Description: This method is used to acquire the data from the data source
        Output: PandaDataFrame
        On_Failure: RaiseExceptions
        Written by: Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version: 0.1
        Revisions: 0
        """
        try:
            self.log_object.create_log_file("Loading training data set from the local source into pandas DataFrame")
            df = pd.read_csv('../Dataset/insurance_dataset.csv')
            return df
        except Exception as e:
            self.log_object.create_log_file("The error is :- " + str(e))
            raise e



















