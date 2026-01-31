'''
Custom exception handling
structured error reporting
improves debugging

'''
import logging
import sys
import logger
def error_massage_detail(error,error_detail:sys):
    """
    Generate detailed error message with file name and line number.
    Args:
        error (Exception): The exception object.
        error_detail (sys): The sys module for accessing exception info.

    Returns:
        str: Formatted error massage
    """ 
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_msg = str(error)
    error_massage = "Error occurred in python script name [{0}] line number [{1}] error massage [{2}]".format(
        file_name,
        line_number,
        error_msg
    )
    return error_massage

class CustomException(Exception):
    def __init__(self,error_massage,error_detail:sys):
        super().__init__(error_massage)
        self.error_massage = error_massage_detail(error_massage,error_detail=error_detail)

    def __str__(self):
        return self.error_massage