import sys
from src.logger import logging


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()

    '''
    Basically exc_info gives three parameters firts 2 we are
    not interrested
    and 3rd exc_tb give information of all at which file or line exception occured

    '''
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = 'Error occured in python script name [{0}] line number [{1}] error message [{2}]'.format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)


    def __str__(self):
        return self.error_message
        






