from time import sleep
import six
import warnings

from ceic_api_client.rest import ApiException

import ceic_api_client.facade.pyceic_exception as ceic_exception_module


# TODO: Need unit tests
class CeicRequestsFacade(object):

    _MAX_REQUEST_COUNT = 4
    _PROGRESSIVE_DELAY_SECONDS_STEP = 2

    def __init__(self, configuration):
        self._configuration = configuration

    def make_request(self, api_call, *args, **kwargs):
        if six.PY3 or six.PY34:
            warnings.simplefilter("ignore", ResourceWarning)

        kwargs.update(self._configuration.default_query_params)

        return self._try_get_response(api_call, *args, **kwargs)

    def _try_get_response(self, api_call, *args, **kwargs):
        try:
            response = self._make_api_call_recursively(api_call, 0, *args, **kwargs)
        except TypeError:
            if "token" in kwargs:
                kwargs.pop("token")
                response = self._try_get_response(api_call, *args, **kwargs)
            else:
                raise

        return response

    def _make_api_call_recursively(self, api_call, recursive_calls_count=0, *args, **kwargs):
        recursive_calls_count += 1

        try:
            return api_call(*args, **kwargs)
        except ApiException as api_exception:
            ceic_exception = self._try_build_ceic_exception_from(api_exception)

            if not self._should_make_recursive_call(recursive_calls_count, ceic_exception):
                ceic_exception.__suppress_context__ = True
                raise ceic_exception

            delay_seconds = recursive_calls_count * self._PROGRESSIVE_DELAY_SECONDS_STEP
            sleep(delay_seconds)

            return self._make_api_call_recursively(api_call, recursive_calls_count, *args, **kwargs)

    @staticmethod
    def _try_build_ceic_exception_from(api_exception):
        try:
            return ceic_exception_module.CeicException.build_ceic_exception_from(api_exception)
        except TypeError:
            return ceic_exception_module.CeicException.build_default_exception_from(api_exception)

    def _should_make_recursive_call(self, count, ceic_exception):
        return count < self._MAX_REQUEST_COUNT and \
               ceic_exception.status != 401 and \
               ceic_exception.status != 400 and \
               type(ceic_exception) is not ceic_exception_module.CeicEmptyLoginException
