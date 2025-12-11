import datetime
import sys
from typing import Dict

from ceic_api_client.facade.pyceic_configuration import CeicConfiguration
from ceic_api_client.facade.pyceic_requests_facade import CeicRequestsFacade
from ceic_api_client.facade.pyceic_sessions_facade import CeicSessionsFacade
from ceic_api_client.facade.pyceic_series_facade import CeicSeriesFacade
from ceic_api_client.facade.pyceic_dictionary_facade import CeicDictionaryFacade
from ceic_api_client.facade.pyceic_layout_facade import CeicLayoutFacade
from ceic_api_client.facade.pyceic_insights_facade import CeicInsightsFacade
from ceic_api_client.facade.pyceic_releases_facade import CeicReleasesFacade
from ceic_api_client.facade.pyceic_exception import *


class Ceic(object):
    _INSTANCE = None
    _INSTANCE_INIT = False
    _DEFAULT_VINTAGES_COUNT = 12
    _DEFAULT_TIMEPOINTS_COUNT = 365
    _EARLIEST_SERIES_LIMIT = 100

    def __new__(cls, username=None, password=None, *args, **kwargs):
        if not cls._INSTANCE:
            cls._INSTANCE = super(Ceic, cls).__new__(cls, *args)

        return cls._INSTANCE

    def __init__(self, username=None, password=None, proxy_url=None, proxy_username=None, proxy_password=None, server=None):
        """
        Constructor for the CEIC SDK Facade.

        :param username: Login username
        :type username: str
        :param password: Login password
        :type password: str
        :param proxy_url: Proxy URL
        :type proxy_url: str
        :param str proxy_username: Proxy username
        :type proxy_username: str
        :param proxy_password: Proxy password
        :type proxy_password: str
        :param server: Api server
        :type server: str
        """

        if not self._INSTANCE_INIT:
            self._init_object(username, password, proxy_url, proxy_username, proxy_password, server)
            self._INSTANCE_INIT = True

        self._try_set_proxy(proxy_url, proxy_username, proxy_password)
        self._try_login(username, password)

    @staticmethod
    def set_region(region):
        """
        Changes the Environment Region. Can be either CN, or US

        :param region: Possible values - CN, US
        :type region: str

        :return: self
        :rtype: ceic_api_client.pyceic.Ceic
        """

        CeicConfiguration.REGION = region

        instance = Ceic._get_instance()
        instance._ceic_configuration.region = CeicConfiguration.REGION

        return instance

    @staticmethod
    def set_environment(environment):
        """
        Changes the environment. Cane be either V2, or SANDBOX

        :param environment: Possible values - V2, SANDBOX
        :type environment: str

        :return: self
        :rtype: ceic_api_client.pyceic.Ceic
        """

        CeicConfiguration.ENVIRONMENT = environment

        instance = Ceic._get_instance()
        instance._ceic_configuration.environment = CeicConfiguration.ENVIRONMENT

        return instance

    @staticmethod
    def set_server(server):
        """
        Changes the API server address.

        :param server: Server URL
        :type server: str

        :return: self
        :rtype: ceic_api_client.pyceic.Ceic
        """

        instance = Ceic._get_instance(server=server)

        instance._ceic_configuration.server = server

        return instance

    @staticmethod
    def get_server():
        """
        Gets the address of the currently set API server.

        :return: The currently set API server
        :rtype: str
        """

        instance = Ceic._get_instance()

        return instance._ceic_configuration.server

    @staticmethod
    def set_proxy(proxy_url=None, proxy_username=None, proxy_password=None):
        """
        Sets a proxy connection.

        :param proxy_url: Proxy URL
        :type proxy_url: str
        :param proxy_username: Proxy username
        :type proxy_username: str
        :param proxy_password: Proxy password
        :type proxy_password: str

        :return: self
        :rtype: ceic_api_client.pyceic.Ceic
        """

        instance = Ceic._get_instance(
            proxy_url=proxy_url,
            proxy_username=proxy_username,
            proxy_password=proxy_password
        )

        instance._ceic_configuration.set_proxy(proxy_url, proxy_username, proxy_password)

        return instance

    @staticmethod
    def login(username=None, password=None):
        """
        Attempts to login and create a login session.

        :param username: Login username
        :type username: str
        :param password: Login password
        :type password: str

        :return self
        :rtype ceic_api_client.pyceic.Ceic

        :raises ceic_api_client.facade.pyceic_exception.CeicInvalidLoginDetailsException : Invalid login details
        :raises ceic_api_client.facade.pyceic_exception.CeicActiveSessionException : User already has an active session
        """

        instance = Ceic._get_instance()

        if instance._should_try_login(username, password):
            instance._sessions_facade.login(username, password)
            instance._try_set_session()

        return instance

    @staticmethod
    def logout():
        """
        Attempts to logout and delete the saved session.

        :return: self - Same Ceic instance
        :rtype: ceic_api_client.pyceic.Ceic

        :raises ceic_api_client.facade.pyceic_exception.CeicSessionExpiredException : Saved session is already expired
        :raises ceic_api_client.facade.pyceic_exception.CeicNoActiveSessionsException : There is no saved session
        """

        instance = Ceic._get_instance()

        if instance._ceic_configuration.environment is instance._ceic_configuration.V2_ENV:
            instance._sessions_facade.logout()
            instance._try_unset_session()

        return instance

    @staticmethod
    def series(series_id, **kwargs):
        """
        Gets full series data. Result contains both metadata and time-points data.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesResult
        """

        instance = Ceic._get_instance()
        if instance._is_insight_series(series_id):
            return Ceic._insight_series_for(series_id, **kwargs)

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_vintages(series_id, **kwargs):
        """
        Gets full series data with vintage timepoints included. Result contains both metadata and time-points data.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword int count: Limits the number of timepoints. Default value is `12`.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesResult
        """
        if type(series_id) is list and len(series_id) > Ceic._EARLIEST_SERIES_LIMIT:
            raise Exception(f"You cannot get more than {Ceic._EARLIEST_SERIES_LIMIT} series per one request.")

        return Ceic._get_vintages(series_id, **kwargs)

    @staticmethod
    def series_vintages_as_dict(series_id, **kwargs) -> Dict[datetime.date, Dict[datetime.date, float]]:
        """
        Gets full series data with vintage timepoints included. Result contains both metadata and time-points data.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword date vintages_start_date: Limits the start date after which the revisions are going to be used.
        :keyword date vintages_end_date: Limits the end date before which the revisions are going to be used.
        :keyword int vintages_count: Limits the number of revisions to be used.
        :keyword int count: Limits the number of timepoints. Default value is `12`.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: A key-value dictionary, where keys are revision dates, values are dictionaries for a timepoint at that revision date,
        consisting of timepoint date as a key and timepoint value as a value.
        :rtype: Dict
        """

        return Ceic._get_vintages_dict(series_id, **kwargs)

    @staticmethod
    def series_metadata(series_id, **kwargs):
        """
        Gets series metadata only.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesResult
        """
        if type(series_id) is list and len(series_id) > Ceic._EARLIEST_SERIES_LIMIT:
            raise Exception(f"You cannot get more than {Ceic._EARLIEST_SERIES_LIMIT} series per one request.")

        instance = Ceic._get_instance()
        if instance._is_insight_series(series_id):
            return Ceic._insight_series_metadata_for(series_id, **kwargs)

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series_metadata
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_data(series_id, **kwargs):
        """
        Gets series time-points only.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword vintage_revision_date: Snapshot of the series at the time of the vintage date.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesResult
        """

        instance = Ceic._get_instance()
        if instance._is_insight_series(series_id):
            return Ceic._insight_series_data_for(series_id, **kwargs)

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series_data
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_earliest(series_id, **kwargs):
        """
        Gets earliest possible data about the series

        :param series_id: A single series id can be passed as a string, an integer, or a list (up to 100 ids).
                            Multiple series ids can be passed as a list only (up to 100 ids).
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword vintage_revision_date: Snapshot of the series at the time of the vintage date.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesResult
        """

        if type(series_id) is list and len(series_id) > Ceic._EARLIEST_SERIES_LIMIT:
            raise Exception(f"You cannot get more than {Ceic._EARLIEST_SERIES_LIMIT} series per one request.")

        instance = Ceic._get_instance()
        if instance._is_insight_series(series_id):
            return Ceic._insight_series_data_for(series_id, **kwargs)

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series_earliest
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_layouts(series_id, **kwargs):
        """
        Gets series layout information only.
        
        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSeriesLayoutsResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series_layouts
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_statistics(series_id, **kwargs):
        """
        Get series statistics
        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :return: Statistics list
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSeriesStatisticsResult
        """
        if type(series_id) is list and len(series_id) > Ceic._EARLIEST_SERIES_LIMIT:
            raise Exception(f"You cannot get more than {Ceic._EARLIEST_SERIES_LIMIT} series per one request.")

        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        get_series_method = instance._series_facade.get_series_statistics
        result = instance._make_request(get_series_method, **kwargs)

        return result

    @staticmethod
    def series_continuous_info(series_id, **kwargs):
        """
        Gets series extended history information, if such exists.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSeriesContinuousChainsResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        get_continuous_info_method = instance._series_facade.get_series_continuous_chains
        result = instance._make_request(get_continuous_info_method, **kwargs)

        return result

    @staticmethod
    def series_continuous_data(series_id, chain_id, **kwargs):
        """
        Gets series extended history data, for a specific chain.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
                            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :param chain_id: A single chain id can be passed as a string
        :type chain_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSeriesContinuousDataResult
        """
        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        kwargs["chain_id"] = chain_id
        get_continuous_data_method = instance._series_facade.get_series_continuous_data
        result = instance._make_request(get_continuous_data_method, **kwargs)

        return result

    @staticmethod
    def search(keyword=None, **kwargs):
        """
        Allows searching for series by a keyword and additional filtering criteria.
        Each filtering criteria accepts one or more, comma separated code values.
        See Dictionary functions for details on how to retrieve a specific filter code.
        The multi-dimensional filters include the economic classification and
        indicators (defined by CEIC database structure), region/country, frequency,
        unit, source, status and observation date.
        
        :param keyword: Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :type keyword: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword bool with_vintage_enabled_only: If it is `true` result will contain ONLY vintage enabled series.
        :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
        :keyword float offset: The offset from which the records will be returned.
            Used to get the next set of records when the limit is reached.
        :keyword list[str] database:  Database filter. One or more comma separated database code values.
            Use `/dictionary/databases` to get an up to date list of available databases. WORLD - *World Trend Plus*
            GLOBAL - *Global Database*  CEICGLBKS - *Global Key Series Database*
            PMI - *Markit Purchasing Managers' Index*  DAILY - *Daily Database*
            BRAZIL - *Brazil Premium Database*  RUSSIA - *Russia Premium Database*
            INDIA - *India Premium Database*  INDONESIA - *Indonesia Premium Database*
            CN - *China Premium Database*  OECD-MEI - *OECD - Main Economic Indicators*
            OECD-EO - *OECD - Economic Outlook*  OECD-PROD - *OECD - Productivity*
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] data_source: Data source for series search operation.
            When CEIC is selected, we search for CEIC series only.
            When USER-IMPORTED is selected, we search only for user imported series
            Accepted values: `CEIC`, `USER-IMPORTED`
            Default: `CEIC`
        :keyword list[str] topic: Topic filter. One or more comma separated topic code values.
        :keyword list[str] section: Section filter. One or more comma separated section code values.
        :keyword list[str] table: Table filter. One or more comma separated table code values.
        :keyword list[str] order: Sort order. Default is `relevance`.
        :keyword list[str] direction: Sort order direction. Default is `asc`.
            Accepted values: `asc` - ascending `desc` - descending
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s
        :keyword bool forecast_only: If it is true result will only contain series with forecast
        :keyword bool with_release_only: If it is true result will only contain series with released schedule
        :keyword bool with_replacements_only: If it is true result will only contain series with suggestions
        :keyword bool with_continuous_only: If it is true, then the result will only contain series with available historical extensions
        :keyword list[str] facet: List of facets to return

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSearchSeriesResult
        """

        instance = Ceic._get_instance()

        if keyword is not None and keyword.strip() != "":
            kwargs["keyword"] = keyword

        search_series_method = instance._series_facade.search_series
        result = instance._make_request(search_series_method, **kwargs)

        return result

    @staticmethod
    def releases(keyword=None, **kwargs):
        """
                Allows searching for series' release schedule by a keyword and additional filtering criteria.
                Each filtering criteria accepts one or more, comma separated code values.
                See Dictionary functions for details on how to retrieve a specific filter code.
                The multi-dimensional filters include the economic classification and
                indicators (defined by CEIC database structure), region/country, frequency,
                unit, source, status and observation date.

                :param keyword: Search term. One or more keywords.
                                May contain special words further controlling the search results. Keyword search tips:
                                    * Retail Sales - Show series with both keywords while
                                        the sequence of keywords is irrelevant.
                                        Equivalent to search Sales Retail
                                    * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                        regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                                    * Retail;Sales - Show series with either keyword and series with both keywords while
                                        the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                                    * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                        regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                        Sales Price, Motor Vehicle Sales
                                    * Retail NOT Sales - Narrow a search by excluding specific terms while
                                        the sequence of keywords is relevant. Show results: series with terms that
                                        include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                                    * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                        while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                        Retail Sales NOT (Hong Kong). Show results: series with terms that
                                        include Retail Sales, but NOT Hong Kong, such as
                                        Retail Sales YoY: China, Retail Sales YoY: United States
                :type keyword: str

                :keyword str lang: Preferred language code in which data will be returned.
                    Defaults to `English` if no translation in the language specified is available. Possible Values:
                                    * en - English
                                    * zh - Chinese
                                    * ru - Russian
                                    * id - Indonesian
                                    * jp - Japanese
                :keyword str format: Response data format. Default is `json`. Possible values:
                                    * json
                                    * xml
                                    * csv
                :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
                :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
                :keyword float offset: The offset from which the records will be returned.
                    Used to get the next set of records when the limit is reached.
                :keyword list[str] database:  Database filter. One or more comma separated database code values.
                    Use `/dictionary/databases` to get an up to date list of available databases. WORLD - *World Trend Plus*
                    GLOBAL - *Global Database*  CEICGLBKS - *Global Key Series Database*
                    PMI - *Markit Purchasing Managers' Index*  DAILY - *Daily Database*
                    BRAZIL - *Brazil Premium Database*  RUSSIA - *Russia Premium Database*
                    INDIA - *India Premium Database*  INDONESIA - *Indonesia Premium Database*
                    CN - *China Premium Database*  OECD-MEI - *OECD - Main Economic Indicators*
                    OECD-EO - *OECD - Economic Outlook*  OECD-PROD - *OECD - Productivity*
                :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
                    D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
                :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
                    See related Dictionary function to get the full list of accepted countries.
                :keyword list[str] source: Source filter. One or more comma separated source code values.
                    See related Dictionary function to get the full list of accepted sources.
                :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
                    See related Dictionary function to get the full list of accepted units.
                :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
                    See related Dictionary function to get full list of accepted indicators.
                :keyword list[str] region: Region filter. One or more comma separated region code values.
                    See related Dictionary function to get the full list of accepted regions.
                :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
                    By default show results for all the series found.
                :keyword bool key_only: Show only 'key' series when set to `true`.
                :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
                :keyword bool name_only: This filter related with the `keyword` filter.
                    If it's `true` keyword search will be searched only in series name instead of all series attributes.
                :keyword date start_date_before: Will return series with first observation before `start_date_before`
                :keyword date end_date_after: Will return series with last observation after `end_date_after`
                :keyword date created_after: Will return series created after `created_after` date
                :keyword date updated_after: Will return series last time updated after `updated_after` date
                :keyword date release_date_from:  Will return releases with first observation before release_date_from
                :keyword date release_date_to:  Will return releases with last observation after release_date_to
                :keyword list[str] status: Status filter. One or more comma separated status code values.
                    When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
                :keyword list[str] topic: Topic filter. One or more comma separated topic code values.
                :keyword list[str] section: Section filter. One or more comma separated section code values.
                :keyword list[str] table: Table filter. One or more comma separated table code values.
                :keyword list[str] order: Sort order. Default is `relevance`.
                :keyword list[str] direction: Sort order direction. Default is `asc`.
                    Accepted values: `asc` - ascending `desc` - descending
                :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
                    When combined with additional search criterion, the result will be an intesection of both.
                :keyword bool with_replacements_metadata: If it is `true` result will contain
                    replacements metadata not only list of id`s
                :keyword list[str] facet: List of facets to return
                :keyword list[str] release_status:  List of release statuses to return

                :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
                :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSearchReleasesResult
                """
        instance = Ceic._get_instance()

        if keyword is not None and keyword.strip() != "":
            kwargs["keyword"] = keyword

        search_releases_method = instance._releases_facade.releases
        result = instance._make_request(search_releases_method, **kwargs)

        return result

    @staticmethod
    def release_series(code, **kwargs):
        """
        Lists the series for a specific release identifier code.
        :param code: Release identifier code for a group of series with the same release schedule.
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
            * json
            * xml
            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
        :keyword float offset: The offset from which the records will be returned.
        """

        instance = Ceic._get_instance()

        kwargs["code"] = code
        releases_series_method = instance._releases_facade.get_release_series
        result = instance._make_request(releases_series_method, **kwargs)

        return result

    @staticmethod
    def series_releases(series_id, **kwargs):
        """
                Lists the release schedule for a single series id
                :param series_id: A single series id can be passed as a string, an integer, or a list.
                :keyword str lang: Preferred language code in which data will be returned.
                    Defaults to `English` if no translation in the language specified is available. Possible Values:
                                    * en - English
                                    * zh - Chinese
                                    * ru - Russian
                                    * id - Indonesian
                                    * jp - Japanese
                :keyword str format: Response data format. Default is `json`. Possible values:
                    * json
                    * xml
                    * csv
                :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
                :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
                :keyword float offset: The offset from which the records will be returned.
                :keyword date release_date_from:  Will return releases with first observation before release_date_from
                :keyword date release_date_to:  Will return releases with last observation after release_date_to
                :keyword list[str] release_status:  List of release statuses to return
                :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesReleasesResult
                """
        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        releases_series_method = instance._releases_facade.get_series_releases
        result = instance._make_request(releases_series_method, **kwargs)

        return result

    @staticmethod
    def dictionaries(**kwargs):
        """
        Full dictionary list . Returns all the available dictionaries.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Full dictionary list
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetDictionariesResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_dictionaries
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def indicators(**kwargs):
        """
        Returns full list of supported indicators, their codes and the related top level classifications.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing indicators list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetIndicatorsResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_indicators
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def classifications(**kwargs):
        """
        Returns full list of supported top level classifications and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing classifications list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetClassificationsResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_classifications
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def classification_indicators(classification_id, **kwargs):
        """
        Returns full list of indicators for specific classification.

        :param classification_id: The ID of the specific classification
        :type classification_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing indicators list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetIndicatorsResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = classification_id
        get_dictionaries_method = instance._dictionary_facade.get_classification_indicators
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def countries(**kwargs):
        """
        DEPRECATED. Please use `Ceic.geo()`. Returns full list of supported countries and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing countries list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetCountriesResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_countries
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def geo(**kwargs):
        """
        Returns full list of supported geo entries.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing geo list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetGeoResult
        """

        instance = Ceic._get_instance()

        get_geo_method = instance._dictionary_facade.get_geo
        result = instance._make_request(get_geo_method, **kwargs)

        return result

    @staticmethod
    def country_sources(country_id, **kwargs):
        """
        Returns full list of sources for a specific country.

        :param country_id: Country ISO code
        :type country_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing sources list for the specific country.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSourcesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = country_id
        get_dictionaries_method = instance._dictionary_facade.get_country_sources
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def regions(**kwargs):
        """
        Returns full list of supported regions and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing regions list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetRegionsResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_regions
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def sources(**kwargs):
        """
        Returns full list of supported sources and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing sources list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSourcesResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_sources
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def units(**kwargs):
        """
        Returns full list of supported units and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing units list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetUnitsResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_units
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def frequencies(**kwargs):
        """
        Returns full list of supported frequencies and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing units list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetFrequenciesResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_frequencies
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def statuses(**kwargs):
        """
        Returns full list of supported statuses and their codes.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing statuses list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetStatusesResult
        """

        instance = Ceic._get_instance()

        get_dictionaries_method = instance._dictionary_facade.get_statuses
        result = instance._make_request(get_dictionaries_method, **kwargs)

        return result

    @staticmethod
    def layout_databases(**kwargs):
        """
        Returns list of layout databases. This is the top level from the layout hierarchy.

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.

        :return: Object containing layout databases list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetLayoutResult
        """

        instance = Ceic._get_instance()

        get_layouts_method = instance._layouts_facade.get_layout_databases
        result = instance._make_request(get_layouts_method, **kwargs)

        return result

    @staticmethod
    def footnotes(node_code, download_path, **kwargs):
        """
        Downloads the footnote and the footnote resources in a specified download path directory

        :param node_code: The node_code for which the footnote will be downloaded
        :param download_path: The desired download directory
        :param kwargs:
        :return: None
        """
        instance = Ceic._get_instance()

        kwargs["node_code"] = node_code
        kwargs["download_path"] = download_path

        download_footnotes_method = instance._layouts_facade.download_footnotes
        instance._make_request(download_footnotes_method, **kwargs)

    @staticmethod
    def layout_database_topics(database_id, **kwargs):
        """
        Returns list of topics for a specific database.

        :param database_id: The database ID
        :type database_id: str

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.

        :return: Object containing layout databases list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetLayoutResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = database_id

        get_layouts_method = instance._layouts_facade.get_layout_database_topics
        result = instance._make_request(get_layouts_method, **kwargs)

        return result

    @staticmethod
    def layout_topic_sections(topic_id, **kwargs):
        """
        Returns list of sections for a specific topic.

        :param topic_id: The topic ID
        :type topic_id: str

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.

        :return: Object containing layout databases list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetLayoutResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = topic_id

        get_layouts_method = instance._layouts_facade.get_layout_topic_sections
        result = instance._make_request(get_layouts_method, **kwargs)

        return result

    @staticmethod
    def layout_section_tables(section_id, **kwargs):
        """
        Returns list of tables for a specific section.

        :param section_id: The section ID
        :type section_id: str

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.

        :return: Object containing layout databases list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetLayoutResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = section_id

        get_layouts_method = instance._layouts_facade.get_layout_section_tables
        result = instance._make_request(get_layouts_method, **kwargs)

        return result

    @staticmethod
    def layout_table_series(table_id, **kwargs):
        """
        Returns list of series inside of a specific table

        :param table_id: The section ID
        :type table_id: str

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword list[str] frequency: Frequency filter. One or more comma separated frequency code values.
            D - Daily W - Weekly M - Monthly Q - Quarterly S - Semi-annual Y - Annual
        :keyword list[str] country: **DEPRECATED. Please use geo parameter.** Country filter. One or more comma separated country code values.
            See related Dictionary function to get the full list of accepted countries.
        :keyword list[str] source: Source filter. One or more comma separated source code values.
            See related Dictionary function to get the full list of accepted sources.
        :keyword list[str] unit: Unit filter. One or more comma separated unit code values.
            See related Dictionary function to get the full list of accepted units.
        :keyword list[str] indicator: Indicator filter. One or more comma separated indicator code values.
            See related Dictionary function to get full list of accepted indicators.
        :keyword list[str] region: Region filter. One or more comma separated region code values.
            See related Dictionary function to get the full list of accepted regions.
        :keyword bool subscribed_only: Show only results for subscribed series when set to `true`.
            By default show results for all the series found.
        :keyword bool key_only: Show only 'key' series when set to `true`.
        :keyword bool new_only: Show only series created less than 1 month ago when set to `true`.
        :keyword bool name_only: This filter related with the `keyword` filter.
            If it's `true` keyword search will be searched only in series name instead of all series attributes.
        :keyword date start_date_before: Will return series with first observation before `start_date_before`
        :keyword date end_date_after: Will return series with last observation after `end_date_after`
        :keyword date created_after: Will return series created after `created_after` date
        :keyword date updated_after: Will return series last time updated after `updated_after` date
        :keyword list[str] status: Status filter. One or more comma separated status code values.
            When not explicitly set, defaults to T.  T - Active C - Discontinued B - Rebased
        :keyword list[str] filter_id: Filter ID used to define a subset of data over which the search will be executed.
            When combined with additional search criterion, the result will be an intesection of both.

        :return: Object containing layout databases list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetLayoutSeriesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = table_id

        get_layouts_method = instance._layouts_facade.get_layout_table_series
        result = instance._make_request(get_layouts_method, **kwargs)

        return result

    @staticmethod
    def insights(**kwargs):
        """
        Returns full list of CDMNext user created insights.

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing insights list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightsResult
        """

        instance = Ceic._get_instance()

        get_insights_method = instance._insights_facade.get_insights
        result = instance._make_request(get_insights_method, **kwargs)

        return result

    @staticmethod
    def search_insights(**kwargs):
        """
        Search for insights. Those could be user created, shared, or CEIC created ones.

        :keyword str keyword Search term. One or more keywords.
                        May contain special words further controlling the search results. Keyword search tips:
                            * Retail Sales - Show series with both keywords while
                                the sequence of keywords is irrelevant.
                                Equivalent to search Sales Retail
                            * Retail AND Sales - Show results: series with terms of Retail AND Sales,
                                regardless of the sequence. E. g. Retail Sales, Automobile Sales Retail.
                            * Retail;Sales - Show series with either keyword and series with both keywords while
                                the sequence of keywords is irrelevant, equivalent to search: Sales;Retail
                            * Retail OR Sales - Show results: series with terms of Retail OR Sales,
                                regardless of the sequence. E. g. Retail Sales, Retail Trade,
                                Sales Price, Motor Vehicle Sales
                            * Retail NOT Sales - Narrow a search by excluding specific terms while
                                the sequence of keywords is relevant. Show results: series with terms that
                                include Retail, but NOT Sales. E. g. Retail Trade, Retail Price, Retail Bank
                            * Retail Sales NOT (Hong Kong) - Narrow a search by excluding a set of words in parentheses
                                while the sequence of keywords in parentheses is irrelevant, equivalent to search:
                                Retail Sales NOT (Hong Kong). Show results: series with terms that
                                include Retail Sales, but NOT Hong Kong, such as
                                Retail Sales YoY: China, Retail Sales YoY: United States
        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword str group: Insights group. Default is `my`. Possible values:
                            * favorite
                            * my
                            * analytics
                            * shared
                            * recent
                            * all
                            * gallery
                            * data_talk
                            * wpic_platinum
        :keyword float limit: Number of records to return
        :keyword float offset: The offset from which the records will be returned
        :keyword str order: Sort order. Possible values:
                            * name
                            * edit_date
                            * open_date
        :keyword str direction: Sort order direction. Possible values:
                            * asc
                            * desc
        :keyword list[str] tags: List of insight tags to search by tag
        :keyword list[str] categories: List of insights categories to search by category

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSearchInsightsResult
        """

        instance = Ceic._get_instance()

        search_insights_method = instance._insights_facade.search_insights
        result = instance._make_request(search_insights_method, **kwargs)

        return result

    @staticmethod
    def insights_categories(**kwargs):
        """
        Returns list of insight categories.
        To be used wtih group filters \"favorite\", \"my\", \"shared\", \"recent\", all\".

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing insight categories list.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightsCategoriesResult
        """

        instance = Ceic._get_instance()

        get_insight_categories_method = instance._insights_facade.get_insights_categories
        result = instance._make_request(get_insight_categories_method, **kwargs)

        return result

    @staticmethod
    def gallery_insights_categories(**kwargs):
        """
        Returns list of gallery categories. To be used with group filters \"analytics\" and \"gallery\".

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing gallery insight categories list.
        :rtype: ceic_api_client.models.insights_categories_result.InsightsCategoriesResult
        """

        instance = Ceic._get_instance()

        get_gallery_insights_categories_method = instance._insights_facade.get_gallery_insights_categories
        result = instance._make_request(get_gallery_insights_categories_method, **kwargs)

        return result

    @staticmethod
    def insight(insight_id, **kwargs):
        """
        Returns information about a specified insight.

        :param insight_id: The insight ID
        :type insight_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing insight result data.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightsResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = insight_id
        get_insight_method = instance._insights_facade.get_insight
        result = instance._make_request(get_insight_method, **kwargs)

        return result

    @staticmethod
    def download_insight(insight_id, file_format, **kwargs):
        """
        Returns one or more links to the insight report.
        When the report generation takes too much time to complete in a timely manner, returns HTTP 408.
        In this case the request have to be repeated after a minute.
        Once the report is generated, consecutive requests are returned immediately.
        Each successful response returns one or more download links that expires in 5 minutes.
        The client application consuming the API shall download the file within this period or
        send additional request to the API.

        :param insight_id: The insight ID
        :type insight_id: str
        :param file_format: Insight report file format. Possible values:
                            * xlsx
                            * pdf
        :type file_format: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: Object containing links to the insight report.
        :rtype: ceic_api_client.models.insight_download_result.InsightDownloadResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = insight_id
        kwargs["file_format"] = file_format
        download_insight_method = instance._insights_facade.download_insight
        result = instance._make_request(download_insight_method, **kwargs)

        return result

    @staticmethod
    def insight_series(insight_id, **kwargs):
        """
        Returns all series from the specified insight(s), including all time-points and metadata,
        as well as their layout in the insight context in terms of grouping and separators.

        :param insight_id: The insight ID
        :type insight_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
        :keyword float offset: The offset from which the records will be returned.

        :return: An iterable object containing insight series result data.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = insight_id
        get_insight_series_method = instance._insights_facade.get_insight_series
        result = instance._make_request(get_insight_series_method, **kwargs)

        return result

    @staticmethod
    def insight_series_data(insight_id, **kwargs):
        """
        Returns all series time-points from the specified insight series.

        :param insight_id: The insight ID
        :type insight_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned.
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted
        :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
        :keyword float offset: The offset from which the records will be returned.

        :return: An iterable object containing insight series time-points result data.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = insight_id
        get_insight_series_data_method = instance._insights_facade.get_insight_series_data
        result = instance._make_request(get_insight_series_data_method, **kwargs)

        return result

    @staticmethod
    def insight_series_metadata(insight_id, **kwargs):
        """
        Returns all series metadata from the specified insight(s),
        as well as their layout in the insight context in terms of grouping and separators.

        :param insight_id: The insight ID
        :type insight_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.
        :keyword float limit: Number of records to return in the range 1 - 100. Default is 100.
        :keyword float offset: The offset from which the records will be returned.

        :return: An iterable object containing insight series metadata result.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = insight_id
        get_insight_series_metadata_method = instance._insights_facade.get_insight_series_metadata
        result = instance._make_request(get_insight_series_metadata_method, **kwargs)

        return result

    @staticmethod
    def series_vintages_dates(series_id, **kwargs):
        """
        Gets vintage dates for particular series.

        :param series_id: A single series id can be passed as a string, an integer, or a list.
            Multiple series ids can be passed as a list only.
        :type series_id: str, int, list

        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword int count: Limits the number of timepoints to use for vintages dates extraction.

        :return: An iterable object which contains result data. Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetSeriesVintageDatesResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        kwargs["count"] = kwargs.get("count", Ceic._DEFAULT_VINTAGES_COUNT)

        if kwargs.get("start_date", False) or kwargs.get("end_date", False):
            kwargs["count"] = sys.maxsize

        get_series_with_vintages_dates_method = instance._series_facade.get_series_vintages_dates
        result = instance._make_request(get_series_with_vintages_dates_method, **kwargs)

        return result

    @staticmethod
    def series_vintages_continuous(series_id, **kwargs):
        """
        Returns vintage continuous series timepoints information

        :param series_id: The series ID.
        :type series_id: int

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: An iterable object containing insight series metadata result for specific insight.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.ContinuousSeriesWithAppliedFunctionsResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = series_id

        get_vintages_continuous_method = instance._series_facade.get_series_vintages_continuous
        result = instance._make_request(get_vintages_continuous_method, **kwargs)

        return result

    @staticmethod
    def series_vintages_continuous_chain(series_id, chain_id, **kwargs):
        """
        Returns vintage continuous series timepoints information

        :param series_id: The series ID.
        :param chain_id: The series' chain ID.
        :type series_id: int
        :type chain_id: int

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.

        :return: An iterable object containing insight series metadata result for specific insight.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicSeriesContinuousDataResult
        """

        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        kwargs["chain_id"] = chain_id

        get_vintages_continuous_method = instance._series_facade.get_series_vintages_continuous_chain
        result = instance._make_request(get_vintages_continuous_method, **kwargs)

        return result

    @staticmethod
    def _insight_series_for(insight_series_id, **kwargs):
        """
        Returns full series data, based on their insight ID.
        It can include any formulas or transformations applied to the data,
        or changes to the metadata (ex. title) as part of the insight context.

        :param insight_series_id: The insight series ID.
        :type insight_series_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted

        :return: An iterable object containing insight series result data for specific insight.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesListResult
        """
        instance = Ceic._get_instance()

        kwargs["series_id"] = insight_series_id
        get_insight_series_list_method = instance._insights_facade.get_insight_series_list
        result = instance._make_request(get_insight_series_list_method, **kwargs)

        return result

    @staticmethod
    def _insight_series_data_for(insight_series_id, **kwargs):
        """
        Returns series time-points data, based on their insight ID.
        It can include any formulas or
        transformations applied to the data as part of the insight context.

        :param insight_series_id: The insight series ID.
        :type insight_series_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword int count: Limit the amount of latest time-points returned, by the number specified.
        :keyword date start_date: Limits the start date after which the time-points will be returned.
        :keyword date end_date: Limits the end date before which the time-points will be returned.
        :keyword datetime updated_after: Returns only the updated time-points after the date specified.
        :keyword bool blank_observations: If it's set to true, empty time-points will be returned.
        :keyword str time_points_status: Time points filter. One or more comma separated status code values.
            When not explicitly set, defaults to `active`. Possible values:
                            * active
                            * deleted

        :return: An iterable object containing insight series time-points result data for specific insight.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesListResult
        """
        instance = Ceic._get_instance()

        kwargs["series_id"] = insight_series_id
        get_insight_series_data_list_method = instance._insights_facade.get_insight_series_data_list
        result = instance._make_request(get_insight_series_data_list_method, **kwargs)

        return result

    @staticmethod
    def _insight_series_metadata_for(insight_series_id, **kwargs):
        """
        Returns series metadata, based on their insight ID.
        It can include changes to the metadata (ex. title) as part of the insight context.

        :param insight_series_id: The insight series ID.
        :type insight_series_id: str

        :keyword str lang: Preferred language code in which data will be returned.
            Defaults to `English` if no translation in the language specified is available. Possible Values:
                            * en - English
                            * zh - Chinese
                            * ru - Russian
                            * id - Indonesian
                            * jp - Japanese
        :keyword str format: Response data format. Default is `json`. Possible values:
                            * json
                            * xml
                            * csv
        :keyword bool with_model_information: If set to `true` returns the model names as part of the response.
        :keyword bool with_replacements_metadata: If it is `true` result will contain
            replacements metadata not only list of id`s.

        :return: An iterable object containing insight series metadata result for specific insight.
            Each object can contain up to 20 result objects.
        :rtype: ceic_api_client.facade.pyceic_facade_models.CeicGetInsightSeriesListResult
        """

        instance = Ceic._get_instance()

        kwargs["series_id"] = insight_series_id
        get_insight_series_metadata_list_method = instance._insights_facade.get_insight_series_metadata_list
        result = instance._make_request(get_insight_series_metadata_list_method, **kwargs)

        return result

    @staticmethod
    def set_token(token):
        """
              Alternative method for authenticating and use CEIC API.

              :param token: User CEIC authentication token
              :type token: str
        """

        instance = Ceic._get_instance()

        instance._ceic_configuration.set_token(token)

        return instance

    @staticmethod

    def _get_instance(proxy_username=None, proxy_password=None, proxy_url=None, server=None):
        if not Ceic._INSTANCE:
            Ceic._INSTANCE = Ceic(
                proxy_username=proxy_username,
                proxy_password=proxy_password,
                proxy_url=proxy_url,
                server=server
            )

        return Ceic._INSTANCE

    @staticmethod
    def _is_insight_series(series_id):
        if type(series_id) is list and len(series_id) > 0:
            series_id = series_id[0]

        return "-" in str(series_id)

    def _init_object(self, username=None, password=None, proxy_url=None, proxy_username=None, proxy_password=None, server=None):
        self._ceic_configuration = CeicConfiguration(server)

        if proxy_url is not None:
            self._ceic_configuration.set_proxy(proxy_url, proxy_username, proxy_password)
        
        self._ceic_requests_facade = CeicRequestsFacade(self._ceic_configuration)

        self._sessions_facade = CeicSessionsFacade(self._ceic_configuration, self._ceic_requests_facade)
        self._series_facade = CeicSeriesFacade(
            self._ceic_configuration,
            self._ceic_requests_facade
        )
        self._dictionary_facade = CeicDictionaryFacade(self._ceic_configuration, self._ceic_requests_facade)
        self._layouts_facade = CeicLayoutFacade(self._ceic_configuration, self._ceic_requests_facade)
        self._insights_facade = CeicInsightsFacade(self._ceic_configuration, self._ceic_requests_facade)
        self._releases_facade = CeicReleasesFacade(self._ceic_configuration, self._ceic_requests_facade)

        self._try_set_proxy(proxy_url, proxy_username, proxy_password)
        self._try_login(username, password)

        self._try_set_session()

    def _try_set_proxy(self, proxy_url=None, proxy_username=None, proxy_password=None):
        if proxy_url is not None or proxy_username is not None or proxy_password is not None:
            self.set_proxy(proxy_url, proxy_username, proxy_password)

    def _try_login(self, username=None, password=None):
        if self._should_try_login(username, password):
            self.login(username, password)

    def _should_try_login(self, username=None, password=None):
        return (username is not None or password is not None) and \
            self._ceic_configuration.environment is self._ceic_configuration.V2_ENV

    def _make_request(self, method, **kwargs):
        try:
            result = method(**kwargs)
        except (CeicNotLoggedInException, CeicSessionExpiredException, CeicSessionTerminatedException):
            self.login()
            result = self._make_request(method, **kwargs)

        return result

    def _try_set_session(self):
        if self._sessions_facade.session_id is not None:
            self._ceic_configuration.set_token(self._sessions_facade.session_id)

    def _try_unset_session(self):
        if self._sessions_facade.session_id is None:
            self._ceic_configuration.unset_token()

    @staticmethod
    def _get_vintages(series_id, **kwargs):
        instance = Ceic._get_instance()

        kwargs["id"] = series_id
        kwargs["count"] = kwargs.get("count", Ceic._DEFAULT_TIMEPOINTS_COUNT)
        get_series_with_vintages_method = instance._series_facade.get_series_vintages
        result = instance._make_request(get_series_with_vintages_method, **kwargs)

        return result

    @staticmethod
    def _get_vintages_dict(series_id, **kwargs):
        vintages_count = kwargs.pop("vintages_count", Ceic._DEFAULT_VINTAGES_COUNT)
        vintages_start_date = kwargs.pop("vintages_start_date", None)
        vintages_end_date = kwargs.pop("vintages_end_date", None)
        with_historical_extension = kwargs.get("with_historical_extension", False)
        last_date = datetime.date(1, 1, 1)

        vintages_kwargs = {
            "count": vintages_count
        }

        if vintages_start_date is not None:
            vintages_kwargs["start_date"] = vintages_start_date

        if vintages_end_date is not None:
            vintages_kwargs["end_date"] = vintages_end_date

        def fetch_vintage_data(s_id, rev_date, shared_data):
            result = Ceic.series_data(s_id, vintage_revision_date=rev_date, **kwargs)
            for item in result.data:
                for tp in item.time_points:
                    time_point_date = tp.date
                    value = tp.value
                    if rev_date not in shared_data:
                        shared_data[rev_date] = {}
                    shared_data[rev_date][time_point_date] = value

        def fetch_vintage_data_whe(s_id, rev_date, shared_data, previous_s_id):
            curr_date = None
            result = Ceic.series_data(s_id, vintage_revision_date=rev_date, **kwargs)
            for item in result.data:
                for tp in item.time_points:
                    time_point_date = tp.date
                    value = tp.value
                    if rev_date not in shared_data:
                        shared_data[rev_date] = {}
                    shared_data[rev_date][time_point_date] = value
                    curr_date = time_point_date

                    if previous_s_id and previous_s_id != s_id and last_date > curr_date:
                        for shared_item in shared_data:
                            shared_data[shared_item][time_point_date] = value

            return curr_date

        if with_historical_extension:
            data = {}
            previous_series_id = None
            chain = Ceic.series_vintages_continuous(series_id)
            series_ids = chain.data.items[0].series
            for sid in reversed(series_ids):
                vintages_dates_result = Ceic.series_vintages_dates(sid, **vintages_kwargs)
                vintages_dates = [obj.date for obj in vintages_dates_result.data]
                vintages_dates.sort(reverse=True)

                for revision_date in vintages_dates:
                    last_date = fetch_vintage_data_whe(sid, revision_date, data, previous_series_id)

                previous_series_id = sid
            return data

        vintages_dates_result = Ceic.series_vintages_dates(series_id, **vintages_kwargs)
        vintages_dates = [obj.date for obj in vintages_dates_result.data]
        data = {}

        for revision_date in vintages_dates:
            fetch_vintage_data(series_id, revision_date, data)

        return data