import urllib3
import json
import six
import warnings

import ceic_api_client.version as Version

class OutputColor(object):
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class LoginSuccess(object):
    _MESSAGE = OutputColor.GREEN + "Connected successfully to the CEIC database" + OutputColor.END

    def show_info(self):
        print(self._MESSAGE)