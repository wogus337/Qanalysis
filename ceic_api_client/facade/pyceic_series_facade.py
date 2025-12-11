from ceic_api_client.apis.series_api import SeriesApi
from ceic_api_client.facade.pyceic_decorators import CeicSeriesRequestDecorators as SeriesDecorator
from ceic_api_client.facade.pyceic_adaptors import *
from ceic_api_client.facade.pyceic_facade_models import *


class CeicSeriesFacade(object):

    def __init__(self, ceic_configuration, ceic_requests_facade):
        self._ceic_configuration = ceic_configuration
        self._ceic_requests_facade = ceic_requests_facade

        series_api = SeriesApi(ceic_configuration.api_client)
        max_series_ids_per_request = ceic_configuration.get_series_series_id_limit

        self._get_series_adaptor = GetSeriesAdaptor(max_series_ids_per_request, ceic_requests_facade, series_api)
        self._get_series_metadata_adaptor = GetSeriesMetadataAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )
        self._get_series_earliest_adaptor = GetSeriesEarliestAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_data_adaptor = GetSeriesDataAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._search_series_adaptor = SearchSeriesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_statistics_adaptor = GetSeriesStatisticsAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_continuous_chains_adaptor = GetSeriesContinuousChainsAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_continuous_data_adaptor = GetSeriesContinuousDataAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_vintages_adaptor = GetSeriesVintagesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_vintages_dates_adaptor = GetSeriesVintagesDatesAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_vintages_continuous_adaptor = GetSeriesVintagesContinuousAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

        self._get_series_vintages_continuous_chain_adaptor = GetSeriesVintagesContinuousChainAdaptor(
            max_series_ids_per_request, ceic_requests_facade, series_api
        )

    @SeriesDecorator.series_by_id
    def get_series(self, **kwargs):
        self._validate_parameters("getSeries", **kwargs)

        result = CeicGetSeriesResult(
            self._get_series_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_vintages(self, **kwargs):
        self._validate_parameters("getSeriesVintages", **kwargs)

        result = CeicGetSeriesResult(
            self._get_series_vintages_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_vintages_dates(self, **kwargs):
        self._validate_parameters("getSeriesVintagesDates", **kwargs)

        result = CeicGetSeriesVintageDatesResult(
            self._get_series_vintages_dates_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_vintages_continuous(self, **kwargs):
        self._validate_parameters("getVintageContinuousSeries", **kwargs)

        result = CeicSeriesContinuousDataResult(
            self._get_series_vintages_continuous_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_vintages_continuous_chain(self, **kwargs):
        self._validate_parameters("getVintageContinuousSeriesTimePoints", **kwargs)

        result = CeicSeriesContinuousDataResult(
            self._get_series_vintages_continuous_chain_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_statistics(self, **kwargs):
        self._validate_parameters("getSeriesStatistics", **kwargs)

        result = CeicSeriesStatisticsResult(
            self._get_series_statistics_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @SeriesDecorator.series_by_id
    def get_series_metadata(self, **kwargs):
        self._validate_parameters("getSeriesMetadata", **kwargs)

        result = CeicGetSeriesResult(
            self._get_series_metadata_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_earliest(self, **kwargs):
        self._validate_parameters("getSeriesEarliest", **kwargs)

        result = CeicGetSeriesResult(
            self._get_series_earliest_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.series_by_id
    def get_series_data(self, **kwargs):
        self._validate_parameters("getSeriesTimePoints", **kwargs)

        result = CeicGetSeriesResult(
            self._get_series_data_adaptor.adapt_api_call,
            **kwargs
        )
        return result

    @SeriesDecorator.get_by_id
    def get_series_layouts(self, **kwargs):
        self._validate_parameters("getSeries", **kwargs)

        result = CeicSeriesLayoutsResult(
            self._get_series_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @SeriesDecorator.get_by_id
    def get_series_continuous_chains(self, **kwargs):
        self._validate_parameters("getContinuousSeries", **kwargs)

        result = CeicSeriesContinuousChainsResult(
            self._get_series_continuous_chains_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @SeriesDecorator.get_by_id
    def get_series_continuous_data(self, **kwargs):
        self._validate_parameters("getContinuousSeriesTimePoints", **kwargs)

        result = CeicSeriesContinuousDataResult(
            self._get_series_continuous_data_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @SeriesDecorator.search
    def search_series(self, search_method, **kwargs):
        self._validate_parameters("searchSeries", **kwargs)

        result = CeicSearchSeriesResult(
            search_method,
            self._search_series_adaptor.adapt_api_call(**kwargs),
            limit=kwargs["limit"] if "limit" in kwargs else None,
            offset=kwargs["offset"] if "offset" in kwargs else None
        )

        return result

    def _validate_parameters(self, operation_id, **kwargs):
        for parameter_validator in self._ceic_configuration.parameter_validators:
            parameter_validator.validate_parameters(operation_id, **kwargs)