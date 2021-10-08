from datetime import datetime


class log_class:
    """
    Class_Name : log_class
    Description: This Class is used to create log files.
    Written By : Adityaraj Hemant Chaudhari , Manthan Takalkar
    Version    : 0.1
    Revisions  : None
    """

    def __init__(self, folder_path, file_name):
        self.folder_path = folder_path
        self.file_name = file_name

    def create_log_file(self, log_message):
        """
        Method: create_log_file
        input: log_message
        output: save log_message in file
        on error: raise error message
        """
        try:
            self.now = datetime.now()
            self.date = self.now.date()
            self.current_time = self.now.strftime("%H:%M:%S")

            with open(self.folder_path + self.file_name, 'a') as file:
                file.write(str(self.date) + "\t" + str(self.current_time) + "\t\t" + log_message + "\n")
        except Exception as e:
            raise e



