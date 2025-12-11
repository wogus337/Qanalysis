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


class PackageUpdateWarning(object):
    _WARNING_MESSAGE_TEMPLATE = OutputColor.YELLOW + \
                                "\n" + OutputColor.BOLD + "WARNING: " + OutputColor.END + \
                                OutputColor.YELLOW + "A new version of the CEIC Python SDK is available - {0}\n" \
                                                     "Current version is: {1}\n" \
                                                     "To get the latest features and bug-fixes, " \
                                                     "please consider updating your package with the following command:\n" \
                                                     "pip install --extra-index-url https://{2}/python ceic_api_client --upgrade" + \
                                OutputColor.END

    def __init__(self, configuration):
        self._configuration = configuration

        config = configuration.api_client.configuration

        if config.proxy_url:
            self._http = urllib3.ProxyManager(
                proxy_url=config.proxy_url,
                proxy_headers=config.proxy_headers,
            )
        else:
            self._http = urllib3.PoolManager()

        self._latest_version = None

    def show_update_warning_if_needed(self):
        if not self._is_current_version_the_latest():
            self._show_update_warning()

    def _show_update_warning(self):
        warning_message = self._WARNING_MESSAGE_TEMPLATE.format(
            self._latest_version, Version.VERSION, self._configuration.python_package_url
        )

        warnings.warn(message=warning_message, category=UserWarning)

    def _is_current_version_the_latest(self):
        current_version = Version.VERSION.strip()
        self._latest_version = self._get_latest_package_version()

        if self._latest_version is not None:
            self._latest_version =  self._latest_version.strip()
            return current_version == self._latest_version
        else:
            return True

    def _get_latest_package_version(self):
        try:
            downloads_file = self._get_downloads_file()
            downloads_file = json.loads(downloads_file)

            return downloads_file["downloads"][2]["documentation"][0]["version"]
        except:
            print("Unable to get latest package version")

        return None

    def _get_downloads_file(self):
        if six.PY3 or six.PY34:
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", urllib3.exceptions.InsecureRequestWarning)

        url = self._configuration.downloads_file_url

        response = self._http.request('GET', url)
        downloads_file = response.data.decode('UTF-8')

        return downloads_file


class AbuseWarning(object):
    _WARNING_MESSAGE = OutputColor.YELLOW + \
                       "\n" + OutputColor.BOLD + "WARNING: " + OutputColor.END + \
                       OutputColor.YELLOW + "The PyCEIC package is designed as direct\n" \
                                            "interaction interface to CEIC macroeconomic data and\n" \
                                            "any data usage abuse attempt will be recorded." + \
                       OutputColor.END

    def show_warning(self):
        warnings.warn(message=self._WARNING_MESSAGE, category=UserWarning)


class DebugInfoWarning(object):
    _DEBUG_HEADERS = {
        "ISI-Px": "PROXY",
        "ISI-Env": "ENVIRONMENT"
    }

    def __init__(self, configuration):
        self._configuration = configuration

        config = configuration.api_client.configuration

        if config.proxy_url:
            self._http = urllib3.ProxyManager(
                proxy_url=config.proxy_url,
                proxy_headers=config.proxy_headers,
            )
        else:
            self._http = urllib3.PoolManager()

    def show_debug_warning(self, session_id):
        self._show_debug_headers(session_id)

    def _show_debug_headers(self, session_id):
        try:
            url = '{}/dictionary/statuses?token={}'.format(self._configuration.api_client.configuration.host, session_id)
            response_headers = self._http.request(url=url, method="GET").headers
            for name, value in self._DEBUG_HEADERS.items():
                if name in response_headers.keys():
                    print('{}: {}'.format(value, response_headers[name]))
        except:
            pass

class DeprecatedMethodWarning(object):
    _WARNING_MESSAGE = OutputColor.YELLOW + \
                       "\n" + OutputColor.BOLD + "WARNING: " + OutputColor.END + \
                       OutputColor.YELLOW + "Deprecated method 'Ceic.{0}()'{1}" + \
                       OutputColor.END

    _REPLACEMENT_METHOD_MESSAGE = "\nPlease use 'Ceic.{0}()' instead."

    def show_warning(self, deprecated_method, replacement_method_name=None):
        additional_message = self._REPLACEMENT_METHOD_MESSAGE.format(replacement_method_name) \
            if replacement_method_name is not None \
            else ""
        message = self._WARNING_MESSAGE.format(deprecated_method, additional_message)

        warnings.warn(message=message, category=UserWarning)


class DeprecatedParameterWarning(object):
    _WARNING_MESSAGE = OutputColor.YELLOW + \
                       "\n" + OutputColor.BOLD + "WARNING: " + OutputColor.END + \
                       OutputColor.YELLOW + "Deprecated parameter: '{0}'{1}" + \
                       OutputColor.END

    _REPLACEMENT_MESSAGE = "\nUse '{0}' instead."

    def show_warning(self, parameter_name, replacement_parameter_name=None):
        additional_message = self._REPLACEMENT_MESSAGE.format(replacement_parameter_name) \
            if replacement_parameter_name \
            else ""

        warning_message = self._WARNING_MESSAGE.format(parameter_name, additional_message)

        warnings.warn(message=warning_message, category=UserWarning)