from ceic_api_client.apis.insights_api import InsightsApi
from ceic_api_client.facade.pyceic_decorators import CeicInsightsRequestDecorators as InsightDecorators
from ceic_api_client.facade.pyceic_facade_models import *
from ceic_api_client.facade.pyceic_adaptors import *

from ceic_api_client.facade.pyceic_facade_models import CeicGetInsightsResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetInsightsCategoriesResult


class CeicInsightsFacade(object):
    
    def __init__(self, ceic_configuration, ceic_requests_facade):
        self._ceic_configuration = ceic_configuration
        self._ceic_requests_facade = ceic_requests_facade
        
        self._insights_api = InsightsApi(self._ceic_configuration.api_client)
        max_ids_per_request = ceic_configuration.get_series_series_id_limit

        self._get_insights_adaptor = GetInsightsAdaptor(self._ceic_requests_facade, self._insights_api)
        self._search_insight_adaptor = SearchInsightsAdaptor(self._ceic_requests_facade, self._insights_api)
        self._get_insights_categories_adaptor = GetInsightsCategoriesAdaptor(
            self._ceic_requests_facade, self._insights_api
        )
        self._get_gallery_insights_categories_adaptor = GetGalleryInsightsCategoriesAdaptor(
            self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_adaptor = GetInsightAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._download_insight_adaptor = DownloadInsightAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_adaptor = GetInsightSeriesAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_data_adaptor = GetInsightSeriesDataAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_metadata_adaptor = GetInsightSeriesMetadataAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_list_adaptor = GetInsightSeriesListAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_data_list_adaptor = GetInsightSeriesDataListAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )
        self._get_insight_series_metadata_list_adaptor = GetInsightSeriesMetadataListAdaptor(
            max_ids_per_request, self._ceic_requests_facade, self._insights_api
        )

    @InsightDecorators.insight
    def get_insights(self, **kwargs):
        self._validate_parameters("getInsights", **kwargs)

        result = CeicGetInsightsResult(
            self._get_insights_adaptor.adapt_api_call(**kwargs).data
        )

        return result

    @InsightDecorators.insight_search
    def search_insights(self, search_method, **kwargs):
        self._validate_parameters("searchInsights", **kwargs)

        result = CeicSearchInsightsResult(
            search_method,
            self._search_insight_adaptor.adapt_api_call(**kwargs),
            limit=kwargs["limit"] if "limit" in kwargs else None,
            offset=kwargs["offset"] if "offset" in kwargs else None
        )

        return result

    @InsightDecorators.insight
    def get_insights_categories(self, **kwargs):
        self._validate_parameters("getInsightsCategories", **kwargs)

        result = CeicGetInsightsCategoriesResult(
            self._get_insights_categories_adaptor.adapt_api_call(**kwargs).data
        )

        return result

    @InsightDecorators.insight
    def get_gallery_insights_categories(self, **kwargs):
        self._validate_parameters("getGalleryInsightsCategories", **kwargs)

        result = self._get_gallery_insights_categories_adaptor.adapt_api_call(**kwargs)

        return result

    @InsightDecorators.insight_by_id
    def get_insight(self, **kwargs):
        self._validate_parameters("getInsight", **kwargs)

        result = CeicGetInsightsResult(
            self._get_insight_adaptor.adapt_api_call(**kwargs).data
        )

        return result

    @InsightDecorators.insight_by_id
    def download_insight(self, **kwargs):
        self._validate_parameters("downloadInsight", **kwargs)

        result = self._download_insight_adaptor.adapt_api_call(**kwargs)

        return result

    @InsightDecorators.insight_by_id
    def get_insight_series(self, **kwargs):
        self._validate_parameters("getInsightSeries", **kwargs)

        result = CeicGetInsightSeriesResult(
            self._get_insight_series_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @InsightDecorators.insight_by_id
    def get_insight_series_data(self, **kwargs):
        self._validate_parameters("getInsightSeriesData", **kwargs)

        result = CeicGetInsightSeriesResult(
            self._get_insight_series_data_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @InsightDecorators.insight_by_id
    def get_insight_series_metadata(self, **kwargs):
        self._validate_parameters("getInsightSeriesMetadata", **kwargs)

        result = CeicGetInsightSeriesResult(
            self._get_insight_series_metadata_adaptor.adapt_api_call,
            **kwargs
        )

        return result
    
    @InsightDecorators.insight_series_by_id
    def get_insight_series_list(self, **kwargs):
        self._validate_parameters("getInsightSeriesList", **kwargs)

        result = CeicGetInsightSeriesListResult(
            self._get_insight_series_list_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @InsightDecorators.insight_series_by_id
    def get_insight_series_data_list(self, **kwargs):
        self._validate_parameters("getInsightSeriesListData", **kwargs)

        result = CeicGetInsightSeriesListResult(
            self._get_insight_series_data_list_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    @InsightDecorators.insight_series_by_id
    def get_insight_series_metadata_list(self, **kwargs):
        self._validate_parameters("getInsightSeriesListMetadata", **kwargs)

        result = CeicGetInsightSeriesListResult(
            self._get_insight_series_metadata_list_adaptor.adapt_api_call,
            **kwargs
        )

        return result

    def _validate_parameters(self, operation_id, **kwargs):
        for parameter_validator in self._ceic_configuration.parameter_validators:
            parameter_validator.validate_parameters(operation_id, **kwargs)
