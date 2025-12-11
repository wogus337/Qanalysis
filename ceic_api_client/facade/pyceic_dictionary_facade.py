from ceic_api_client.apis.dictionary_api import DictionaryApi
from ceic_api_client.facade.pyceic_decorators import CeicDictionaryRequestDecorator as DictionaryDecorator
from ceic_api_client.facade.pyceic_warnings import DeprecatedMethodWarning
from ceic_api_client.facade.pyceic_facade_models import CeicGetDictionariesResult, CeicGetIndicatorsResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetClassificationsResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetCountriesResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetGeoResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetSourcesResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetRegionsResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetUnitsResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetFrequenciesResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetStatusesResult


class CeicDictionaryFacade(object):

    def __init__(self, ceic_configuration, ceic_requests_facade):
        self._ceic_configuration = ceic_configuration
        self._ceic_requests_facade = ceic_requests_facade

        self._dictionary_api = DictionaryApi(self._ceic_configuration.api_client)

        self._method_warning = DeprecatedMethodWarning()

    @DictionaryDecorator.dictionary
    def get_dictionaries(self, **kwargs):
        self._validate_parameters("getDictionaries", **kwargs)

        result = CeicGetDictionariesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_dictionaries,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_indicators(self, **kwargs):
        self._validate_parameters("getIndicators", **kwargs)

        result = CeicGetIndicatorsResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_indicators,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_classifications(self, **kwargs):
        self._validate_parameters("getClassifications", **kwargs)

        result = CeicGetClassificationsResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_classifications,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_classification_indicators(self, **kwargs):
        self._validate_parameters("getClassificationIndicators", **kwargs)

        result = CeicGetIndicatorsResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_classification_indicators,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_countries(self, **kwargs):
        # TODO: Move this check to a validator similar to parameters validator
        self._method_warning.show_warning('countries', 'geo')

        self._validate_parameters("getCountries", **kwargs)

        result = CeicGetCountriesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_countries,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_geo(self, **kwargs):
        self._validate_parameters("getGeo", **kwargs)

        result = CeicGetGeoResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_geo,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_country_sources(self, **kwargs):
        self._validate_parameters("getCountrySources", **kwargs)

        result = CeicGetSourcesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_country_sources,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_regions(self, **kwargs):
        self._validate_parameters("getRegions", **kwargs)

        result = CeicGetRegionsResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_regions,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_sources(self, **kwargs):
        self._validate_parameters("getSources", **kwargs)

        result = CeicGetSourcesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_sources,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_units(self, **kwargs):
        self._validate_parameters("getUnits", **kwargs)

        result = CeicGetUnitsResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_units,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_frequencies(self, **kwargs):
        self._validate_parameters("getFrequencies", **kwargs)

        result = CeicGetFrequenciesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_frequencies,
                **kwargs
            ).data
        )

        return result

    @DictionaryDecorator.dictionary
    def get_statuses(self, **kwargs):
        self._validate_parameters("getStatuses", **kwargs)

        result = CeicGetStatusesResult(
            self._ceic_requests_facade.make_request(
                self._dictionary_api.get_statuses,
                **kwargs
            ).data
        )

        return result

    def _validate_parameters(self, operation_id, **kwargs):
        for parameter_validator in self._ceic_configuration.parameter_validators:
            parameter_validator.validate_parameters(operation_id, **kwargs)
