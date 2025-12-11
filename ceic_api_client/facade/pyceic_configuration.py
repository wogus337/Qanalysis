from ceic_api_client import rest
from ceic_api_client.api_client import ApiClient
from ceic_api_client.facade.pyceic_parameter_validators import ParameterEnumValidator
from ceic_api_client.facade.pyceic_parameter_validators import ParameterDeprecatedValidator


class CeicConfiguration(object):

    _DEFAULT_RETURN_TYPE = "application/json"
    _GET_SERIES_SERIES_ID_LIMIT = 3000

    _DOWNLOADS_FILE_URL_PROD_US = "https://downloads.ceicdata.com/downloads.json"
    _DOWNLOADS_FILE_URL_PROD_CN = "https://downloads.ceicdata.com.cn/downloads.json"
    _DOWNLOADS_FILE_URL_STAGE = "https://downloads-stage.ceicdata.com/downloads.json"

    _PYTHON_PACKAGE_URL_PROD_US = "https://downloads.ceicdata.com/python"
    _PYTHON_PACKAGE_URL_PROD_CN = "https://downloads.ceicdata.com.cn/python"
    _PYTHON_PACKAGE_URL_STAGE = "https://downloads-stage.ceicdata.com/python"

    _SPECIFICATION_URL_PROD_US = "https://insights.ceicdata.com/api/v2/specification"
    _SPECIFICATION_URL_PROD_CN = "https://insights.ceicdata.com.cn/api/v2/specification"
    _SPECIFICATION_URL_STAGE = "https://stage.ceicdata.com/api/v2/specification"

    _API_URL_PROD_US = "https://api.ceicdata.com/"
    _API_URL_PROD_CN = "https://api.ceicdata.com.cn/"
    _API_URL_STAGE = "https://api-stage.ceicdata.com/stage"

    US_REGION = "US"
    CN_REGION = "CN"

    V2_ENV = "v2"
    SANDBOX_ENV = "sandbox"

    REGION = US_REGION
    ENVIRONMENT = V2_ENV

    def __init__(self, server=None):
        self._region = self.REGION
        self._environment = self.ENVIRONMENT
        self._api_client = ApiClient()
        self._set_default_return_type()

        if server is not None:
            self._api_client.configuration.host = server

        self._parameter_validators = [
            ParameterEnumValidator(self.specification_url),
            ParameterDeprecatedValidator(self.specification_url)
        ]

        self._default_query_params = {}

    @property
    def api_client(self):
        return self._api_client

    @property
    def downloads_file_url(self):
        return self._DOWNLOADS_FILE_URL_STAGE if "api-stage" in self._api_client.configuration.host else \
            self._DOWNLOADS_FILE_URL_PROD_US if self._region == self.US_REGION else \
            self._DOWNLOADS_FILE_URL_PROD_CN

    @property
    def python_package_url(self):
        return self._PYTHON_PACKAGE_URL_STAGE if "api-stage" in self._api_client.configuration.host else \
            self._PYTHON_PACKAGE_URL_PROD_US if self._region == self.US_REGION else \
            self._PYTHON_PACKAGE_URL_PROD_CN

    @property
    def specification_url(self):
        return self._SPECIFICATION_URL_STAGE if "api-stage" in self._api_client.configuration.host else \
            self._SPECIFICATION_URL_PROD_US if self._region == self.US_REGION else \
            self._SPECIFICATION_URL_PROD_CN

    @property
    def parameter_validators(self):
        return self._parameter_validators

    @property
    def get_series_series_id_limit(self):
        return self._GET_SERIES_SERIES_ID_LIMIT

    @property
    def server(self):
        return self._api_client.configuration.host

    @server.setter
    def server(self, value):
        self._api_client.configuration.host = value

    @property
    def region(self):
        return self._region

    @region.setter
    def region(self, value):
        value = value.upper()
        if value not in [self.US_REGION, self.CN_REGION]:
            raise ValueError("Possible values: " + self.US_REGION + ", " + self.CN_REGION)

        self._region = value
        self.server = self._API_URL_STAGE if "api-stage" in self._api_client.configuration.host else \
            self._API_URL_PROD_US if self._region == self.US_REGION else \
            self._API_URL_PROD_CN

        self.server += self.environment

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        value = value.lower()
        if value not in [self.V2_ENV, self.SANDBOX_ENV]:
            raise ValueError("Possible values: " + self.V2_ENV + ", " + self.SANDBOX_ENV)

        self._environment = value
        self.region = self._region

    @property
    def default_query_params(self):
        return self._default_query_params

    def set_token(self, access_token):
        self.default_query_params["token"] = access_token

    def unset_token(self):
        if "token" in self._default_query_params:
            self.default_query_params.pop("token")

    def set_proxy(self, proxy_url=None, proxy_username=None, proxy_password=None):
        self._api_client.configuration.set_proxy(proxy_url, proxy_username, proxy_password)
        self._api_client.rest_client = rest.RESTClientObject(self._api_client.configuration)

    def _set_default_return_type(self):
        self._api_client.set_default_header("Accept", self._DEFAULT_RETURN_TYPE)