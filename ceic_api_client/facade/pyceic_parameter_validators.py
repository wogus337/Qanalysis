
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import json

from ceic_api_client.facade.pyceic_exception import CeicInvalidInputParameterException
from ceic_api_client.facade.pyceic_warnings import DeprecatedParameterWarning


class ParameterValidator(object):

    _SOURCE_URL = None
    _SOURCE = None

    def __init__(self, source_url):
        if self._SOURCE is None or self._SOURCE_URL is None or self._SOURCE_URL != source_url:
            self._SOURCE_URL = source_url
            self._load_source()

        self._source = self._SOURCE

    def _load_source(self):
        response = urlopen(self._SOURCE_URL)
        data = response.read().decode('utf-8')
        self._SOURCE = json.loads(data)


class ParameterEnumValidator(ParameterValidator):

    def __init__(self, source_url):
        super(ParameterEnumValidator, self).__init__(source_url)

        self._paths_source = self._source["paths"]
        self._parameters_source = self._source["parameters"]

        self._paths = self._load_paths()

    def validate_parameters(self, operation_id, **kwargs):
        if operation_id not in self._paths:
            raise KeyError("Invalid operation ID: " + operation_id)

        parameters = self._paths[operation_id]
        for param_name, param_values in kwargs.items():
            self._validate_parameter(param_name, param_values, parameters)

    @staticmethod
    def _build_exception_message(param_name, param_value, param_values):
        message = "Invalid valid for '{}' - '{}'. You should use one of these values: '{}'".format(
            param_name,
            param_value,
            ",".join(param_values)
        )

        return message

    def _load_paths(self):
        paths = {}
        for path_source in self._paths_source.values():
            if "get" in path_source:
                operation_id = path_source["get"]["operationId"]
                parameters = self._load_parameters_from(path_source["get"]["parameters"])
                paths[operation_id] = parameters

            if "post" in path_source:
                operation_id = path_source["post"]["operationId"]
                parameters = self._load_parameters_from(path_source["post"]["parameters"])
                paths[operation_id] = parameters

        return paths

    def _load_parameters_from(self, parameters_source):
        parameters = {}
        for parameter_source in parameters_source:
            if self._parameter_is_enum(parameter_source):
                parameter = self._load_parameter(parameter_source)
                parameters[parameter["name"]] = parameter["enums"]

        return parameters

    def _parameter_is_enum(self, parameter_source):
        if "$ref" not in parameter_source:
            return False

        parameter_key = parameter_source["$ref"].split('/')
        parameter_key = parameter_key[len(parameter_key) - 1]

        parameter = self._parameters_source[parameter_key]
        try:
            return "enum" in parameter or "enum" in parameter["items"]
        except KeyError:
            return False

    def _load_parameter(self, parameter_source):
        parameter_key = parameter_source["$ref"].split('/')
        parameter_key = parameter_key[len(parameter_key) - 1]

        parameter = self._parameters_source[parameter_key]
        parameter = {
            "name": parameter["name"],
            "enums": parameter["enum"] if "enum" in parameter else parameter["items"]["enum"]
        }
        parameter["enums"] = [
            str(value).strip().lower() for value in parameter["enums"]
        ]

        return parameter

    def _validate_parameter(self, param_name, param_value, parameters):
        if param_name in parameters:
            param_values = parameters[param_name]
            param_value = self._normalize_param_value(param_value)
            for value in param_value:
                if value not in param_values:
                    message = self._build_exception_message(param_name, param_value, param_values)
                    raise CeicInvalidInputParameterException(message)

    @staticmethod
    def _normalize_param_value(param_value):
        if type(param_value) == int:
            param_value = str(param_value)

        if type(param_value) == str:
            if "," in param_value:
                param_value = param_value.split(",")
            else:
                param_value = [param_value]

        if type(param_value) != list:
            raise TypeError("Usupported format: " + str(type(param_value)))

        param_value = [str(value).strip().lower() for value in param_value]

        return param_value


class ParameterDeprecatedValidator(ParameterEnumValidator):

    def validate_parameters(self, operation_id, **kwargs):
        if operation_id not in self._paths:
            raise KeyError("Invalid operation ID: " + operation_id)

        parameters = self._paths[operation_id]
        for param_name, param_values in kwargs.items():
            self._validate_parameter(param_name, param_values, parameters)

    def _load_parameters_from(self, parameters_source):
        parameters = {}
        for parameter_source in parameters_source:
            parameter = self._load_parameter(parameter_source)
            parameters[parameter["name"]] = parameter

        return parameters

    def _load_parameter(self, parameter_source):
        parameter_key = parameter_source["$ref"].split('/')
        parameter_key = parameter_key[len(parameter_key) - 1]

        parameter = self._parameters_source[parameter_key]
        parameter = {
            "name": parameter["name"],
            "deprecated": parameter["deprecated"] if "deprecated" in parameter else False,
            "description": parameter["description"] if "description" in parameter else ""
        }

        return parameter

    def _validate_parameter(self, param_name, param_value, parameters):
        if param_name not in parameters \
                or not parameters[param_name]["deprecated"] \
                or "**DEPRECATED" not in parameters[param_name]["description"]:
            return

        warning = DeprecatedParameterWarning()
        description = parameters[param_name]["description"]
        replacement_param = description.split('**')[1].split('`')[1]

        warning.show_warning(param_name, replacement_parameter_name=replacement_param)
