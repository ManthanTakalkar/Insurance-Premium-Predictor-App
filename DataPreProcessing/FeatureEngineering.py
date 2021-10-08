import pandas as pd
import os
from InsurancePremium.Logging.Logs import log_class
from InsurancePremium.DataIngestion.Data_Acquisition import LoadingRaw
pd.set_option('display.max_columns', None)


class PreProcessor:

    """
    Class_Name : PreProcessor
    Description: This Class is used to Perform Feature Engineering.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self):
        self.folder = '../LogFiles/'
        self.filename = 'Preprocessing_log.txt'
        self.loader = LoadingRaw()
        if not os.path.isdir(self.folder):
            os.mkdir(self.folder)
        self.log_object = log_class(self.folder, self.filename)

    def featureencoding(self):

        """
        Method_Name : featureencoding()
        Description : This method is used to transform the categorical values into numerical values using One Hot Encoding
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """

        try:
            self.log_object.create_log_file('Converting Categorical Values from Feature Column to Numerical values.')
            encoded_data = pd.get_dummies(self.loader.get_data()[['sex', 'smoker', 'region']], drop_first=True)
            self.log_object.create_log_file('Converted Categorical Values from Feature Column to Numerical values.')
            data = pd.concat((self.loader.get_data(), encoded_data), axis=1)
            return data
        except Exception as e:
            self.log_object.create_log_file('The error is :- ' + str(e))
            raise e

    def dropfeatures(self):

        """
        Method_Name : dropfeatues()
        Description : This method is used to drop features which have been converted to numerical type.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """

        try:
            self.log_object.create_log_file('Dropping some of the categorical features')
            data = self.featureencoding()
            data = data.drop(['sex', 'smoker', 'region'], axis=1)
            self.log_object.create_log_file('Features with Categorical values dropped!')
            return data
        except Exception as e:
            self.log_object.create_log_file('The error is :- ' + str(e))
            raise e

    def renamefeatures(self):

        """
        Method_Name : renamefeatures()
        Description : This method is used to give appropriate name to some features.
        output      : DataFrame
        on failure  : raise exception
        Written by  : Adityaraj Hemant Chaudhari, Manthan Takalkar
        Version     : 0.1
        Revisions   : None
        """

        try:
            self.log_object.create_log_file('Renaming column names.')
            data = self.dropfeatures()
            data = data.rename({'sex_male': 'gender', 'smoker_yes': 'smoker', 'region_northwest': 'northwest',
                                'region_southeast': 'southeast', 'region_southwest': 'southwest'}, axis=1)
            self.log_object.create_log_file('Columns Renamed!')
            return data
        except Exception as e:
            self.log_object.create_log_file('The error is:- ' + str(e))
            raise e



