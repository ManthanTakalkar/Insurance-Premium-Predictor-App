import pandas as pd
from InsurancePremium.Logging.Logs import log_class
from InsurancePremium.DataIngestion.Data_Acquisition import LoadingRaw
import os


class DataInfo:
    """
    Class Name : DataInfo
    Description: This class is used to figure out size and shape and overall info of data we loaded previously
    written by : Adityaraj Hemant Chaudhari, Manthan Takalkar
    version    : 0.1
    revisions  : None
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'info_log.txt'
        self.loader = LoadingRaw()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def get_shape(self):
        """
        Method_Name : get_shape()
        Description : This method is used to get the shape of the dataset loaded previously
        output      : 2-D array
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Finding the dataset shape')
            shape = self.loader.get_data().shape
            self.log_object.create_log_file('Dataset shape Found')
            return shape
        except Exception as e:
            self.log_object.create_log_file('The Error is :- ' + str(e))
            raise e

    def get_info(self):
        """
        Method_Name : get_shape()
        Description : This method is used to get the info of the features in the dataset loaded previously
        output      : 2-D array
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Finding the Info of Features from the dataset')
            info = self.loader.get_data().info
            self.log_object.create_log_file('Dataset info found')
            return info
        except Exception as e:
            self.log_object.create_log_file('The Error is :- ' + str(e))
            raise e

    def get_size(self):
        """
        Method_Name : get_size()
        Description : This method is used to get the size of the dataset loaded previously
        output      : 2-D array
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """
        try:
            self.log_object.create_log_file('Finding the dataset size')
            size = self.loader.get_data().size
            self.log_object.create_log_file('Dataset size found')
            return size
        except Exception as e:
            self.log_object.create_log_file('The Error is :- ' + str(e))
            raise e

