import traceback
import sys

# Custom exception class that captures the stack trace and provides a custom message + Pre-defined Exceptions
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message) # If error exists it will print pre-defined errors
        self.error_message = self.get_detailed_error_message(error_message, error_detail)

    # We dont need to create our CustomeException class again and again to show our CustomeException. That's y we are using static method
    @staticmethod 
    def get_detailed_error_message(error_message, error_detail:sys):

        _, _, exc_tb = traceback.sys.exc_info() # exc_info return 3 things, we don't need 1st two
        file_name = exc_tb.tb_frame.f_code.co_filename # Get the file name where the exception occurred
        line_number = exc_tb.tb_lineno # Get the line number where the exception occurred
        error_message = f"Error occurred in: {file_name} at line number: {line_number} with message: {error_message}"
        return error_message
    
    # This method gives textual representation of the exception when printed
    def __str__(self):
        return self.error_message