from ceic_api_client.facade.pyceic_exception import *


class CeicRequestDecorator(object):

    _DEFAULT_ID_KEY = "id"

    @staticmethod
    def _get_by_id(self, func, id_key=None, params_to_normalize=None, ids_validator=None, **kwargs):
        if params_to_normalize is None:
            params_to_normalize = {}

        if id_key is None:
            id_key = CeicRequestDecorator._DEFAULT_ID_KEY

        kwargs = CeicRequestDecorator._normalize_kwargs(params_to_normalize, **kwargs)

        def wrapper():
            ids = kwargs[id_key]
            ids = CeicRequestDecorator._normalize(ids)
            if ids_validator is not None:
                ids_validator(ids)

            kwargs[id_key] = ids

            return func(self, **kwargs)

        return wrapper()

    @staticmethod
    def _search(self, func, search_adaptor, params_to_normalize=None, **kwargs):
        if params_to_normalize is None:
            params_to_normalize = {}

        kwargs = CeicRequestDecorator._normalize_kwargs(params_to_normalize, **kwargs)

        def wrapper():
            def search_method(offset, limit):
                kwargs["offset"] = offset
                kwargs["limit"] = limit

                return search_adaptor.adapt_api_call(**kwargs)

            return func(self, search_method, **kwargs)

        return wrapper()

    @staticmethod
    def _normalize(param_value):
        param_value = CeicRequestDecorator._normalize_param_format(param_value)
        param_value = CeicRequestDecorator._remove_duplicates(param_value)

        return param_value

    @staticmethod
    def _normalize_param_format(param_value):
        if isinstance(param_value, int):
            param_value = str(param_value)

        if isinstance(param_value, str):
            param_value = CeicRequestDecorator._convert_param_values_to_list(param_value)

        if not isinstance(param_value, list):
            raise ValueError("Unsupported parameter value format: {0}".format(type(param_value)))

        param_value = CeicRequestDecorator._normalize_param_value_list_elements(param_value)

        for single_id in param_value:
            CeicRequestDecorator._check_param_value_str_limit(single_id)

        return param_value

    @staticmethod
    def _remove_duplicates(param_value):
        ids_set = set(param_value)
        param_value = list(ids_set)

        return param_value

    @staticmethod
    def _convert_param_values_to_list(param_value):
        CeicRequestDecorator._check_param_value_str_limit(param_value)

        return [param_value]

    @staticmethod
    def _normalize_param_value_list_elements(param_values):
        for index in range(0, len(param_values)):
            single_id = param_values[index]
            if not isinstance(single_id, str):
                param_values[index] = str(single_id)

        return param_values

    @staticmethod
    def _check_param_value_str_limit(param_value):
        if "," in param_value:
            raise ValueError("Only single values can be passed as strings.\n"
                             "Multiple values must be passed as list.")

    @staticmethod
    def _normalize_kwargs(params, **kwargs):
        for param_key in params.keys():
            if param_key in kwargs:
                kwargs[param_key] = CeicRequestDecorator._normalize(kwargs[param_key])
                if type(params[param_key]) == str:
                    kwargs[param_key] = ",".join(kwargs[param_key])

        return kwargs


class CeicSeriesRequestDecorators(CeicRequestDecorator):

    _PARAMETERS_TO_NORMALIZE = {
        "layout": [],
        "database": [],
        "frequency": [],
        "country": [],
        "source": [],
        "unit": [],
        "indicator": [],
        "region": [],
        "status": [],
        "topic": [],
        "section": [],
        "table": [],
        "order": [],
        "direction": [],
        "filter_id": [],
        "facet": [],
        "time_points_status": "",
        "lang": ""
    }

    @staticmethod
    def get_by_id(func):
        def wrapper(self, **kwargs):
            return CeicRequestDecorator._get_by_id(
                self,
                func,
                params_to_normalize=CeicSeriesRequestDecorators._PARAMETERS_TO_NORMALIZE,
                **kwargs
            )

        return wrapper

    @staticmethod
    def series_by_id(func):
        def wrapper(self, **kwargs):
            return CeicRequestDecorator._get_by_id(
                self,
                func,
                params_to_normalize=CeicSeriesRequestDecorators._PARAMETERS_TO_NORMALIZE,
                ids_validator=CeicSeriesRequestDecorators._series_ids_validator,
                **kwargs
            )

        return wrapper

    @staticmethod
    def search(func):
        def wrapper(self, **kwargs):
            search_adaptor = self._search_series_adaptor
            return CeicRequestDecorator._search(
                self, func, search_adaptor,
                params_to_normalize=CeicSeriesRequestDecorators._PARAMETERS_TO_NORMALIZE,
                **kwargs
            )

        return wrapper

    @staticmethod
    def _series_ids_validator(series_ids):
        for series_id in series_ids:
            if "-" in series_id:
                raise CeicInvalidSeriesIdException("Invalid series_id: " + series_id)


class CeicReleasesRequestDecorators(CeicSeriesRequestDecorators):
    _PARAMETERS_TO_NORMALIZE = {
        "release_status": []
    }

    @staticmethod
    def search(func):
        def wrapper(self, **kwargs):
            search_adaptor = self._search_releases_adaptor
            parameters_to_normalize = CeicReleasesRequestDecorators._get_parameters_to_normalize()
            return CeicRequestDecorator._search(
                self, func, search_adaptor,
                params_to_normalize=parameters_to_normalize,
                **kwargs
            )

        return wrapper

    @staticmethod
    def _get_parameters_to_normalize():
        params = CeicSeriesRequestDecorators._PARAMETERS_TO_NORMALIZE
        params.update(CeicReleasesRequestDecorators._PARAMETERS_TO_NORMALIZE)

        return params


class CeicInsightsRequestDecorators(CeicRequestDecorator):

    _INSIGHT_SERIES_KEY = "series_id"
    _PARAMETERS_TO_NORMALIZE = {
        "tags": [],
        "categories": [],
        "time_points_status": "",
        "lang": ""
    }

    @staticmethod
    def insight_search(func):
        def wrapper(self, **kwargs):
            search_adaptor = self._search_insight_adaptor
            return CeicRequestDecorator._search(
                self, func, search_adaptor, params_to_normalize=CeicInsightsRequestDecorators._PARAMETERS_TO_NORMALIZE,
                **kwargs
            )

        return wrapper

    @staticmethod
    def insight_by_id(func):
        def wrapper(self, **kwargs):
            return CeicRequestDecorator._get_by_id(
                self, func, params_to_normalize=CeicInsightsRequestDecorators._PARAMETERS_TO_NORMALIZE, **kwargs
            )

        return wrapper

    @staticmethod
    def insight_series_by_id(func):
        def wrapper(self, **kwargs):
            return CeicRequestDecorator._get_by_id(
                self,
                func,
                id_key=CeicInsightsRequestDecorators._INSIGHT_SERIES_KEY,
                params_to_normalize=CeicInsightsRequestDecorators._PARAMETERS_TO_NORMALIZE,
                ids_validator=CeicInsightsRequestDecorators._series_ids_validator,
                **kwargs
            )

        return wrapper

    @staticmethod
    def insight(func):
        def wrapper(self, **kwargs):
            kwargs = CeicRequestDecorator._normalize_kwargs(
                CeicInsightsRequestDecorators._PARAMETERS_TO_NORMALIZE, **kwargs
            )
            return func(self, **kwargs)

        return wrapper

    @staticmethod
    def _series_ids_validator(series_ids):
        for series_id in series_ids:
            if "-" not in series_id:
                raise CeicInvalidSeriesIdException("Invalid insight series_id: " + series_id)


class CeicDictionaryRequestDecorator(CeicRequestDecorator):

    _PARAMETERS_TO_NORMALIZE = {
        "lang": ""
    }

    @staticmethod
    def dictionary(func):
        def wrapper(self, **kwargs):
            kwargs = CeicRequestDecorator._normalize_kwargs(
                CeicDictionaryRequestDecorator._PARAMETERS_TO_NORMALIZE, **kwargs
            )
            return func(self, **kwargs)

        return wrapper


class CeicLayoutRequestDecorator(CeicRequestDecorator):

    _PARAMETERS_TO_NORMALIZE = {
        "lang": "",
        "filter_id": [],
        "country": [],
        "frequency": [],
        "indicator": [],
        "region": [],
        "source": [],
        "status": [],
        "unit": []
    }

    @staticmethod
    def layout(func):
        def wrapper(self, **kwargs):
            kwargs = CeicRequestDecorator._normalize_kwargs(
                CeicLayoutRequestDecorator._PARAMETERS_TO_NORMALIZE, **kwargs
            )
            return func(self, **kwargs)

        return wrapper

