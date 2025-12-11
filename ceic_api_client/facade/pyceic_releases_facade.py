from ceic_api_client.apis.releases_api import ReleasesApi
from ceic_api_client.facade.pyceic_decorators import CeicReleasesRequestDecorators as ReleasesDecorators
from ceic_api_client.facade.pyceic_adaptors import *
from ceic_api_client.facade.pyceic_facade_models import *


class CeicReleasesFacade(object):

    def __init__(self, ceic_configuration, ceic_requests_facade):
        self._ceic_configuration = ceic_configuration
        self._ceic_requests_facade = ceic_requests_facade

        releases_api = ReleasesApi(ceic_configuration.api_client)
        max_series_ids_per_request = ceic_configuration.get_series_series_id_limit

        self._search_releases_adaptor = SearchReleasesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, releases_api
        )

        self._get_series_releases_adaptor = GetSeriesReleasesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, releases_api
        )

        self._get_release_series_adaptor = GetReleaseSeriesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, releases_api
        )

    @ReleasesDecorators.search
    def releases(self, search_method, **kwargs):
        self._validate_parameters("searchSeriesReleases")

        result = CeicSearchReleasesResult(
            search_method,
            self._search_releases_adaptor.adapt_api_call(**kwargs),
            limit=kwargs["limit"] if "limit" in kwargs else None,
            offset=kwargs["offset"] if "offset" in kwargs else None
        )

        return result

    @ReleasesDecorators.series_by_id
    def get_series_releases(self, **kwargs):
        self._validate_parameters("getSeriesReleases", **kwargs)
        result = CeicGetSeriesReleasesResult(self._get_series_releases_adaptor.adapt_api_call, **kwargs)

        return result

    def get_release_series(self, **kwargs):
        self._validate_parameters("getReleaseSeries", **kwargs)
        result = CeicGetReleaseSeriesResult(self._get_release_series_adaptor.adapt_api_call, **kwargs)

        return result

    def _validate_parameters(self, operation_id, **kwargs):
        for parameter_validator in self._ceic_configuration.parameter_validators:
            parameter_validator.validate_parameters(operation_id, **kwargs)