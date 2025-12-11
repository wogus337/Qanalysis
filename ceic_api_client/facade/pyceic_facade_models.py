import pprint
import re  # noqa: F401

import six

from ceic_api_client.models.search_series_result import SearchSeriesResult
from ceic_api_client.models.release_series_result import ReleaseSeriesResult
from ceic_api_client.models.series_result import SeriesResult
from ceic_api_client.models.vintage_dates_result import VintageDatesResult
from ceic_api_client.models.insights_search_result import InsightsSearchResult
from ceic_api_client.models.insight_series_result import InsightSeriesResult
from ceic_api_client.models.series_statistics import SeriesStatistics
from ceic_api_client.models.series_release_schedule_search_result import SeriesReleaseScheduleSearchResult
from ceic_api_client.models.continuous_series_result import ContinuousSeriesResult
from ceic_api_client.models.continuous_series_with_applied_functions_result import \
    ContinuousSeriesWithAppliedFunctionsResult
from ceic_api_client.models.classifications_result import ClassificationsResult
from ceic_api_client.models.countries_result import CountriesResult
from ceic_api_client.models.dictionary_result import DictionaryResult
from ceic_api_client.models.frequencies_result import FrequenciesResult
from ceic_api_client.models.geo_result import GeoResult
from ceic_api_client.models.indicators_result import IndicatorsResult
from ceic_api_client.models.insights_categories_result import InsightsCategoriesResult
from ceic_api_client.models.insights_result import InsightsResult
from ceic_api_client.models.regions_result import RegionsResult
from ceic_api_client.models.series_with_vintages import SeriesWithVintages
from ceic_api_client.models.sources_result import SourcesResult
from ceic_api_client.models.statuses_result import StatusesResult
from ceic_api_client.models.units_result import UnitsResult
from ceic_api_client.models.layout_items_result import LayoutItemsResult
from ceic_api_client.models.insight_series import InsightSeries

import pandas as pd
from pandas import DataFrame, Series



class CeicSeriesLayout(object):

    def __init__(self, id, layout):
        self._id = str(id)
        self._layout = layout

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, value):
        self._id = value

    @property
    def layout(self):
        return self._layout

    @layout.setter
    def layout(self, value):
        self._layout = value

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {
            "id": str(self.id),
            "layout": [layout.to_dict() for layout in self.layout]
        }

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, SeriesResult):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other


class CeicSearchResult(object):
    _DEFAULT_LIMIT = 100
    _DEFAULT_OFFSET = 0

    def __init__(self, search_method, search_series_result, limit, offset):
        self._search_method = search_method
        self._as_pandas = False
        self._limit = limit if limit is not None else self._DEFAULT_LIMIT
        self._offset = offset if offset is not None else self._DEFAULT_OFFSET

        self._initial_offset = self._offset
        self._initial_result = search_series_result
        self._initial_iteration = True

    def __iter__(self):
        self._reset_iteration()
        return self

    def __next__(self):
        if self._initial_iteration:
            self._initial_iteration = False
            if self._as_pandas:
                return self.convert_to_pandas(self._initial_result)
            return self._initial_result

        self._offset += self._limit
        result = self._search_method(self._offset, self._limit)
        if len(result.data.items) == 0:
            self._reset_iteration()
            raise StopIteration()

        if self._as_pandas:
            return self.convert_to_pandas(result)

        return result

    def next(self):
        return self.__next__()

    def _reset_iteration(self):
        self._offset = self._initial_offset
        self._initial_iteration = True

    def as_pandas(self):
        self._as_pandas = True
        return self.convert_to_pandas(self._initial_result)


    def convert_to_pandas(self, series_result):
        if len(series_result.data.items) == 1:
            series = series_result.data.items[0]

            return pd.Series(
                data=(
                    getattr(series.metadata, "id", None),
                    getattr(series.metadata, "name", None),
                    ', '.join(
                        indicator["name"] for indicators in getattr(series.metadata, "indicators", []) for indicator in
                        indicators),
                    getattr(series.metadata.indicator, "name", None),
                    getattr(series.metadata.classification, "name", None),
                    getattr(series.metadata.unit, "name", None),
                    getattr(series.metadata.country, "name", None),
                    getattr(series.metadata.frequency, "name", None),
                    getattr(series.metadata.status, "name", None),
                    getattr(series.metadata.source, "name", None),
                    ', '.join(geo_info.name for geo_info in getattr(series.metadata, "geo_info", []) or []),
                    getattr(series.metadata, "remarks", None),
                    getattr(series.metadata, "mnemonic", None),
                    getattr(series.metadata, "is_forecast", None),
                    getattr(series.metadata, "has_vintage", None),
                    getattr(series.metadata, "headline_series", None),
                    getattr(series.metadata, "has_schedule", None),
                    getattr(series.metadata, "series_tag", None),
                    pd.to_datetime(getattr(series.metadata, "start_date", None)),
                    pd.to_datetime(getattr(series.metadata, "end_date", None)),
                    getattr(series.metadata, "multiplier_code", None),
                    pd.to_datetime(getattr(series.metadata, "last_update_time", None)),
                    pd.to_datetime(getattr(series.metadata, "timepoints_last_update_time", None)),
                    getattr(series.metadata, "key_series", None),
                    getattr(series.metadata, "new_series", None),
                    getattr(series.metadata, "period_end", None),
                    getattr(getattr(series.metadata, "last_change", None), "value", None),
                    getattr(series.metadata, "last_change", None),
                    getattr(series.metadata, "number_of_observations", None),
                    getattr(series.metadata, "has_continuous_series", None),
                    getattr(series.metadata, "trade_code", None),
                    series.replacements,
                    ", ".join(layout.series_code for layout in series.layout),
                    ", ".join(layout.database.name for layout in series.layout),
                    ", ".join(layout.topic.name for layout in series.layout),
                    ", ".join(layout.section.name for layout in series.layout),
                    ", ".join(layout.table.name for layout in series.layout)
                ),
                index=[
                    'id',
                    'name',
                    'indicators',
                    'indicator',
                    'classification',
                    'unit',
                    'country',
                    'frequency',
                    'status',
                    'source',
                    'geo_info',
                    'remarks',
                    'mnemonic',
                    'is_forecast',
                    'has_vintage',
                    'headline_series',
                    'has_schedule',
                    'series_tag',
                    'start_date',
                    'end_date',
                    'multiplier_code',
                    'last_update_date',
                    'time_points_last_update_time',
                    'key_series',
                    'new_series',
                    'period_end',
                    'last_value',
                    'last_change',
                    'number_of_observations',
                    'has_continuous_series',
                    'trade_code',
                    'replacement',
                    'series_code',
                    'database',
                    'topic',
                    'section',
                    'table'
                ]
            )

        df = pd.DataFrame(
            [(getattr(series.metadata, "id", None),
              getattr(series.metadata, "name", None),
              ', '.join(indicator["name"] for indicators in getattr(series.metadata, "indicators", []) for indicator in
                        indicators),
              getattr(series.metadata.indicator, "name", None),
              getattr(series.metadata.classification, "name", None),
              getattr(series.metadata.unit, "name", None),
              getattr(series.metadata.country, "name", None),
              getattr(series.metadata.frequency, "name", None),
              getattr(series.metadata.status, "name", None),
              getattr(series.metadata.source, "name", None),
              ', '.join(geo_info.name for geo_info in getattr(series.metadata, "geo_info", []) or []),
              getattr(series.metadata, "remarks", None),
              getattr(series.metadata, "mnemonic", None),
              getattr(series.metadata, "is_forecast", None),
              getattr(series.metadata, "has_vintage", None),
              getattr(series.metadata, "headline_series", None),
              getattr(series.metadata, "has_schedule", None),
              getattr(series.metadata, "series_tag", None),
              pd.to_datetime(getattr(series.metadata, "start_date", None)),
              pd.to_datetime(getattr(series.metadata, "end_date", None)),
              getattr(series.metadata, "multiplier_code", None),
              pd.to_datetime(getattr(series.metadata, "last_update_time", None)),
              pd.to_datetime(getattr(series.metadata, "timepoints_last_update_time", None)),
              getattr(series.metadata, "key_series", None),
              getattr(series.metadata, "new_series", None),
              getattr(series.metadata, "period_end", None),
              getattr(series.metadata, "last_value", None),
              getattr(getattr(series.metadata, "last_change", None), "value", None),
              getattr(series.metadata, "number_of_observations", None),
              getattr(series.metadata, "has_continuous_series", None),
              getattr(series.metadata, "trade_code", None),
              series.replacements,
              ", ".join(layout.series_code for layout in series.layout),
              ", ".join(layout.database.name for layout in series.layout),
              ", ".join(layout.topic.name for layout in series.layout),
              ", ".join(layout.section.name for layout in series.layout),
              ", ".join(layout.table.name for layout in series.layout))
             for series in series_result.data.items
             if series.metadata is not None
             ],
            columns=[
                'id',
                'name',
                'indicators',
                'indicator',
                'classification',
                'unit',
                'country',
                'frequency',
                'status',
                'source',
                'geo_info',
                'remarks',
                'mnemonic',
                'is_forecast',
                'has_vintage',
                'headline_series',
                'has_schedule',
                'series_tag',
                'start_date',
                'end_date',
                'multiplier_code',
                'last_update_date',
                'time_points_last_update_time',
                'key_series',
                'new_series',
                'period_end',
                'last_value',
                'last_change',
                'number_of_observations',
                'has_continuous_series',
                'trade_code',
                'replacement',
                'series_code',
                'database',
                'topic',
                'section',
                'table'
            ]
        )
        df.attrs["batch_size"] = len(series_result.data.items)
        df.attrs["total"] = series_result.data.total

        return df


class CeicSearchSeriesResult(CeicSearchResult, SearchSeriesResult):

    def __init__(self, search_method, search_series_result, limit, offset):
        CeicSearchResult.__init__(self, search_method, search_series_result, limit, offset)
        SearchSeriesResult.__init__(self, search_series_result.data)


class CeicSearchReleasesResult(CeicSearchResult, SeriesReleaseScheduleSearchResult):

    def __init__(self, search_method, search_releases_result, limit, offset):
        CeicSearchResult.__init__(self, search_method, search_releases_result, limit, offset)
        SeriesReleaseScheduleSearchResult.__init__(self, search_releases_result.data)

    def as_pandas(self):
        self._as_pandas = True
        return self.convert_to_pandas(self._initial_result)


    def convert_to_pandas(self, series_result):
        if len(series_result.data.items) == 1:
            series = series_result.data.items[0]

            return pd.Series(
                data=(series.code,
                    series.timepoint_date,
                    series.release_date,
                    series.release_date,
                    series.release_date,
                    series.release_type,
                    series.release_status,
                ),
                index=[
                    'code',
                    'time_point_date',
                    'release_date',
                    'period_start',
                    'period_end',
                    'release_type',
                    'release_status'
                ]
            )

        df = pd.DataFrame(
            [(series.code,
              series.timepoint_date,
              series.release_date,
              series._from,
              series.to,
              series.release_type,
              series.release_status,
              )
             for series in series_result.data.items
             ],
            columns=[
                'code',
                'time_point_date',
                'release_date',
                'period_start',
                'period_end',
                'release_type',
                'release_status'
            ]
        )

        df.attrs["total"] = len(series_result.data.items)

        return df

class CeicSearchInsightsResult(CeicSearchResult, InsightsSearchResult):

    def __init__(self, search_method, search_insights_result, limit, offset):
        CeicSearchResult.__init__(self, search_method, search_insights_result, limit, offset)
        InsightsSearchResult.__init__(self, search_insights_result.data)


class CeicSeriesResult(object):
    _SERIES_PER_REQUEST = 100

    def __init__(self, series_method, ids_key, **kwargs):
        self._series_method = series_method
        self._ids_key = ids_key
        self._kwargs = kwargs

        self._id_chunks = self._split_array_into_chunks(self._kwargs[ids_key])

        self._reset_iteration()
        self._initial_result = self._get_result_for(0)
        self._as_pandas = False

    def __iter__(self):
        self._reset_iteration()
        return self

    def __next__(self):
        self._current_chunk_index += 1
        if self._current_chunk_index >= len(self._id_chunks):
            self._reset_iteration()
            raise StopIteration()

        if self._current_chunk_index == 0:
            return self._initial_result

        self._kwargs[self._ids_key] = self._id_chunks[self._current_chunk_index]
        result = self._series_method(**self._kwargs)

        if self._as_pandas:
            series_list = self._as_series_list(result)
            return self.convert_to_pandas(series_list)
        return result

    def next(self):
        return self.__next__()

    def _split_array_into_chunks(self, ids):
        return [
            ids[index: index + self._SERIES_PER_REQUEST] for
            index in
            range(0, len(ids), self._SERIES_PER_REQUEST)
        ]

    def _reset_iteration(self):
        self._current_chunk_index = -1

    def _get_result_for(self, chunk_index):
        original_id = self._kwargs[self._ids_key]

        self._kwargs[self._ids_key] = self._id_chunks[chunk_index]
        result = self._series_method(**self._kwargs)

        self._kwargs[self._ids_key] = original_id

        return result

    def as_series_list(self) -> list[Series]:
        result = list()
        if type(self._initial_result) is ReleaseSeriesResult:
            for series in self._initial_result.data.items:
                result.append(
                    pd.Series(
                        data=(
                            getattr(series, "series_id", None),
                            getattr(series, "name", None),
                            getattr(series, "frequency", None),
                            getattr(series, "source_name", None),
                            pd.to_datetime(getattr(series, "start_date", None)),
                            pd.to_datetime(getattr(series, "end_date", None)),
                            pd.to_datetime(getattr(series, "last_updated", None)),
                            getattr(series, "unit", None)
                        ),
                        index=[
                            'series_id',
                            'name',
                            'frequency',
                            'source_name',
                            'start_date',
                            'end_date',
                            'last_updated',
                            'unit'
                        ]
                    )
                )
            return result

        for series in self._initial_result:
            result.append(
                pd.Series(
                    data=[d.value for d in series.time_points],
                    index=[d.date for d in series.time_points],
                    name=getattr(series.metadata, "name", None)
                )
            )

        return result

    def _as_series_list(self, series_list) -> list[Series]:
        result = list()
        for series in series_list:
            result.append(
                pd.Series(
                    data=[d.value for d in series.time_points],
                    index=[d.date for d in series.time_points],
                    name=getattr(series.metadata, "name", None)
                )
            )

        return result

    def as_pandas(self):
        self._as_pandas = True

        series_list = self.as_series_list()
        return self.convert_to_pandas(series_list)


    def convert_to_pandas(self, series_list):
        if len(series_list) == 1:
            return series_list[0]

        df = pd.DataFrame(series_list)
        df.attrs["batch_size"] = self._SERIES_PER_REQUEST
        df.attrs["total"] = len(series_list)

        return df


class CeicGetSeriesResult(CeicSeriesResult, SeriesResult):
    _IDS_KEY = "id"

    def __init__(self, get_series_method, **kwargs):
        CeicSeriesResult.__init__(self, get_series_method, self._IDS_KEY, **kwargs)
        SeriesResult.__init__(self, errors=self._initial_result.errors, data=self._initial_result.data)

    def _as_series_list(self) -> list[Series]:
        result = list()
        for series in self.data:
            if type(series) is SeriesWithVintages:
                if series.metadata is not None:
                    ps = pd.Series(
                        data=(
                            getattr(series.metadata, "id", None),
                            getattr(series.metadata, "name", None),
                            getattr(series.metadata.country, "name", None),
                            getattr(series.metadata.classification, "name", None),
                        ),
                        index=['series_id', 'vintage_id', 'vintage_date', 'description']
                    )
                    result.append(ps)
                else:
                    ps = pd.Series(
                        data=[
                            {"vintage_value": vintage.value, "revision_date":  pd.to_datetime(vintage.revision_date)}
                            for tp in getattr(series, "time_points", []) for vintage in tp.vintages
                        ],
                        index=[
                            pd.to_datetime(tp.date)
                            for tp in getattr(series, "time_points", []) for vintage in tp.vintages
                        ],
                        name=series.entity_id
                    )

                    result.append(ps)

            elif series.time_points is None:
                ps = pd.Series(
                    data=(
                        getattr(series.metadata, "id", None),
                        getattr(series.metadata, "name", None),
                        ', '.join(indicator["name"] for indicators in getattr(series.metadata, "indicators", []) for indicator in indicators),
                        getattr(series.metadata.indicator, "name", None),
                        getattr(series.metadata.classification, "name", None),
                        getattr(series.metadata.unit, "name", None),
                        getattr(series.metadata.country, "name", None),
                        getattr(series.metadata.frequency, "name", None),
                        getattr(series.metadata.status, "name", None),
                        getattr(series.metadata.source, "name", None),
                        ', '.join(geo_info.name for geo_info in getattr(series.metadata, "geo_info", []) or []),
                        getattr(series.metadata, "remarks", None),
                        getattr(series.metadata, "mnemonic", None),
                        getattr(series.metadata, "is_forecast", None),
                        getattr(series.metadata, "has_vintage", None),
                        getattr(series.metadata, "headline_series", None),
                        getattr(series.metadata, "has_schedule", None),
                        getattr(series.metadata, "series_tag", None),
                        pd.to_datetime(getattr(series.metadata, "start_date", None)),
                        pd.to_datetime(getattr(series.metadata, "end_date", None)),
                        getattr(series.metadata, "multiplier_code", None),
                        pd.to_datetime(getattr(series.metadata, "last_update_time", None)),
                        pd.to_datetime(getattr(series.metadata, "timepoints_last_update_time", None)),
                        getattr(series.metadata, "key_series", None),
                        getattr(series.metadata, "new_series", None),
                        getattr(series.metadata, "period_end", None),
                        getattr(series.metadata, "last_value", None),
                        getattr(series.metadata, "last_change", None),
                        getattr(series.metadata, "number_of_observations", None),
                        getattr(series.metadata, "has_continuous_series", None),
                        getattr(series.metadata, "trade_code", None),
                        series.replacements,
                    ),
                    index=[
                        'id',
                        'name',
                        'indicators',
                        'indicator',
                        'classification',
                        'unit',
                        'country',
                        'frequency',
                        'status',
                        'source',
                        'geo_info',
                        'remarks',
                        'mnemonic',
                        'is_forecast',
                        'has_vintage',
                        'headline_series',
                        'has_schedule',
                        'series_tag',
                        'start_date',
                        'end_date',
                        'multiplier_code',
                        'last_update_date',
                        'time_points_last_update_time',
                        'key_series',
                        'new_series',
                        'period_end',
                        'last_value',
                        'last_change',
                        'number_of_observations',
                        'has_continuous_series',
                        'trade_code',
                        'replacement'
                    ]
                )
                result.append(ps)
            else:
                ps = pd.Series(
                    data=[
                        {"value": tp.value, "last_update_date": pd.to_datetime(tp.last_update_time)}
                        for tp in getattr(series, "time_points", [])
                    ],
                    index=[
                        pd.to_datetime(tp.date)
                        for tp in getattr(series, "time_points", [])
                    ],
                    name=series.entity_id
                )

                if series.metadata is not None:
                    ps.attrs["series_id"] = getattr(series.metadata, "id", None),
                    ps.attrs["name"] = getattr(series.metadata, "name", None),
                    ps.attrs["unit"] = getattr(series.metadata.unit, "name", None),
                    ps.attrs["frequency"] = getattr(series.metadata.frequency, "name", None),
                    ps.attrs["period_end"] = getattr(series.metadata, "period_end", None)

                result.append(ps)

        return result

    def as_pandas(self) -> DataFrame:
        series_list = self._as_series_list()
        if type(self.data[0]) is SeriesWithVintages and self.data[0].metadata is None:
            frames = []
            for s in series_list:
                dataframe = s.apply(pd.Series)
                dataframe["id"] = s.name
                dataframe["date"] = dataframe.index
                frames.append(dataframe)

            df = pd.concat(frames, ignore_index=True)

            df = df[['revision_date', 'vintage_value', 'date', 'id']] # set columns to specific order
            df = (df
                  .sort_values(by=['date', 'revision_date','id'], ascending=[False, False,False])
                  .reset_index(drop=True))

            return df
        elif self.data[0].time_points is None:
            df = pd.DataFrame(series_list)
        else:
            frames = []
            for s in series_list:
                dataframe = s.apply(pd.Series)
                dataframe["id"] = s.name
                dataframe["date"] = dataframe.index
                frames.append(dataframe)

            df = pd.concat(frames, ignore_index=True)

            df = df[['last_update_date', 'value', 'date', 'id']]  # set columns to specific order
            df = (df
                  .sort_values(by=['date', 'last_update_date', 'id'], ascending=[False, False, False])
                  .reset_index(drop=True))

            if len(series_list) == 1:
                return df

        if len(series_list) == 1 or len(series_list[0].attrs) == 0:
            return df

        for series in series_list:
            series_id = str(series.attrs["series_id"])
            df.attrs[series_id] = {}
            df.attrs[series_id]["name"] = series.attrs["name"]
            df.attrs[series_id]["unit"] = series.attrs["unit"]
            df.attrs[series_id]["frequency"] = series.attrs["frequency"]
            df.attrs[series_id]["period_end"] = series.attrs["period_end"]

        return df

class CeicGetSeriesVintageDatesResult(CeicSeriesResult, VintageDatesResult):
    _IDS_KEY = "id"

    def __init__(self, get_series_vintage_dates_method, **kwargs):
        CeicSeriesResult.__init__(self, get_series_vintage_dates_method, self._IDS_KEY, **kwargs)
        VintageDatesResult.__init__(self, data=self._initial_result.data)


class CeicGetSeriesReleasesResult(CeicSeriesResult, SeriesReleaseScheduleSearchResult):
    _IDS_KEY = "id"

    def __init__(self, get_series_releases_method, **kwargs):
        CeicSeriesResult.__init__(self, get_series_releases_method, self._IDS_KEY, **kwargs)
        SeriesReleaseScheduleSearchResult.__init__(self, data=self._initial_result.data)

    def as_pandas(self):
        self._as_pandas = True
        return self.convert_to_pandas(self._initial_result)


    def convert_to_pandas(self, series_result):
        if len(series_result.data.items) == 1:
            series = series_result.data.items[0]

            return pd.Series(
                data=(series.code,
                    series.timepoint_date,
                    series.release_date,
                    series.release_date,
                    series.release_date,
                    series.release_type,
                    series.release_status,
                ),
                index=[
                    'code',
                    'time_point_date',
                    'release_date',
                    'period_start',
                    'period_end',
                    'release_type',
                    'release_status'
                ]
            )

        df = pd.DataFrame(
            [(series.code,
              series.timepoint_date,
              series.release_date,
              series._from,
              series.to,
              series.release_type,
              series.release_status,
              )
             for series in series_result.data.items
             ],
            columns=[
                'code',
                'time_point_date',
                'release_date',
                'period_start',
                'period_end',
                'release_type',
                'release_status'
            ]
        )

        df.attrs["total"] = len(series_result.data.items)

        return df

class CeicGetReleaseSeriesResult(CeicSeriesResult, ReleaseSeriesResult):
    _IDS_KEY = "code"

    def __init__(self, get_release_series, **kwargs):
        CeicSeriesResult.__init__(self, get_release_series, self._IDS_KEY, **kwargs)
        ReleaseSeriesResult.__init__(self, data=self._initial_result.data)


class CeicSeriesContinuousChainsResult(CeicSeriesResult, ContinuousSeriesResult):
    _IDS_KEY = "id"

    def __init__(self, get_continuous_series, **kwargs):
        CeicSeriesResult.__init__(self, get_continuous_series, self._IDS_KEY, **kwargs)
        ContinuousSeriesResult.__init__(self, data=self._initial_result.data)

    def as_pandas(self):
        return self.convert_to_pandas(self._initial_result)


    def convert_to_pandas(self, series_result):
        if len(series_result.data.items) == 1:
            series = series_result.data.items[0]

            return pd.Series(
                data=(
                    series.chain_id,
                    series.function_description,
                    ', '.join(s for s in series.series)
                ),
                index=['chain_id', 'function_description', 'series_id']
            )

        df = pd.DataFrame(
            [(
                series.chain_id,
                series.function_description,
                ', '.join(s for s in series.series)
            )
                for series in series_result.data.items
            ],
            columns=['chain_id', 'function_description', 'series_id']
        )
        df.attrs["total"] = len(series_result.data.items)

        return df

class CeicSeriesContinuousDataResult(CeicSeriesResult, ContinuousSeriesWithAppliedFunctionsResult):
    _IDS_KEY = "id"

    def __init__(self, get_continuous_data, **kwargs):
        CeicSeriesResult.__init__(self, get_continuous_data, self._IDS_KEY, **kwargs)
        ContinuousSeriesWithAppliedFunctionsResult.__init__(self, data=self._initial_result.data)

    def as_pandas(self):
        return self.convert_to_pandas(self._initial_result)


    def convert_to_pandas(self, series_result):
        if type(series_result) is ContinuousSeriesWithAppliedFunctionsResult:
            return pd.Series(
                data=[float(d.value) for d in series_result.data.time_points],
                index=[d.date for d in series_result.data.time_points],
                name=""
                )
        if len(series_result.data.items) == 1:
            series = series_result.data.items[0]

            return pd.Series(
                data=(
                    series.chain_id,
                    series.function_description,
                    ', '.join(s for s in series.series)
                ),
                index=['chain_id', 'function_description', 'series_ids']
            )

        df = pd.DataFrame(
            [(
                series.chain_id,
                series.function_description,
                ', '.join(s for s in series.series)
            )
             for series in series_result.data.items
             ],
            columns=['chain_id', 'function_description', 'series_id']
        )
        df.attrs["total"] = len(series_result.data.items)

        return df

class CeicSeriesLayoutsResult(CeicGetSeriesResult):

    def __next__(self):
        result = super(CeicSeriesLayoutsResult, self).__next__()
        self.data = result.data

        return self

    @property
    def data(self):
        """Gets the data of this CeicSeriesLayoutsResult.  # noqa: E501

        An array of series  # noqa: E501

        :return: The data of this CeicSeriesLayoutsResult.  # noqa: E501
        :rtype: list[CeicSeriesLayout]
        """
        return self._get_layouts_for(self._data)

    @data.setter
    def data(self, data):
        self._data = data

    @staticmethod
    def _get_layouts_for(series_list):
        series_layouts = [
            CeicSeriesLayout(series.metadata.id, series.layout) for series in series_list
        ]

        return series_layouts

    def as_pandas(self):
        return self.convert_to_pandas(self.data)


    def convert_to_pandas(self, series_result):
        df = pd.DataFrame(
            [(
                series.id,
                layout.series_code,
                getattr(layout.database, "name", None),
                getattr(layout.topic, "name", None),
                getattr(layout.section, "name", None),
                getattr(layout.table, "name", None),
            )
            for series in series_result for layout in series.layout
            ],
            columns=['series_id', 'sr_code', 'database', 'topic', 'section', 'table']
        )
        df.attrs["total"] = sum(len(series.layout) for series in series_result)

        return df

    def as_series_list(self) -> list[Series]:
        result = list()
        for series in self._initial_result:
            result.append(
                pd.Series(
                    data=[d.value for d in series.time_points],
                    index=[d.date for d in series.time_points],
                    name=getattr(series.metadata, "name", None)
                )
            )

        return result

class CeicGetInsightSeriesResult(CeicSeriesResult, InsightSeriesResult):
    _IDS_KEY = "id"

    def __init__(self, get_series_method, **kwargs):
        CeicSeriesResult.__init__(self, get_series_method, self._IDS_KEY, **kwargs)
        InsightSeriesResult.__init__(self, data=self._initial_result.data)

    def _as_series_list(self) -> list[Series]:
        result = list()
        if getattr(self.data, "items", None) is None:
            for series in self.data:
                if not hasattr(series.series_data, "time_points") or series.series_data.time_points is None:
                    ps = pd.Series(
                        data=(
                            series.insight_series.id,
                            series.insight_series.insight_id,
                            series.insight_series.group,
                            series.insight_series.type,
                            series.insight_series.applied_functions,
                            getattr(series.series_data.metadata, "id", None),
                            getattr(series.series_data.metadata, "name", None),
                            ', '.join(indicator["name"] for indicators in getattr(series.series_data.metadata, "indicators", []) for
                                      indicator in indicators),
                            getattr(series.series_data.metadata.indicator, "name", None),
                            getattr(series.series_data.metadata.classification, "name", None),
                            getattr(series.series_data.metadata.unit, "name", None),
                            getattr(series.series_data.metadata.country, "name", None),
                            getattr(series.series_data.metadata.frequency, "name", None),
                            getattr(series.series_data.metadata.status, "name", None),
                            getattr(series.series_data.metadata.source, "name", None),
                            ', '.join(geo_info.name for geo_info in getattr(series.series_data.metadata, "geo_info", [])),
                            getattr(series.series_data.metadata, "remarks", None),
                            getattr(series.series_data.metadata, "mnemonic", None),
                            getattr(series.series_data.metadata, "is_forecast", None),
                            getattr(series.series_data.metadata, "has_vintage", None),
                            getattr(series.series_data.metadata, "headline_series", None),
                            getattr(series.series_data.metadata, "has_schedule", None),
                            getattr(series.series_data.metadata, "series_tag", None),
                            pd.to_datetime(getattr(series.series_data.metadata, "start_date", None)),
                            pd.to_datetime(getattr(series.series_data.metadata, "end_date", None)),
                            getattr(series.series_data.metadata, "multiplier_code", None),
                            pd.to_datetime(getattr(series.series_data.metadata, "last_update_time", None)),
                            pd.to_datetime(getattr(series.series_data.metadata, "timepoints_last_update_time", None)),
                            getattr(series.series_data.metadata, "key_series", None),
                            getattr(series.series_data.metadata, "new_series", None),
                            getattr(series.series_data.metadata, "period_end", None),
                            getattr(series.series_data.metadata, "last_value", None),
                            getattr(getattr(series.series_data.metadata, "last_change", None), "value", None),
                            getattr(series.series_data.metadata, "number_of_observations", None),
                            getattr(series.series_data.metadata, "has_continuous_series", None),
                            getattr(series.series_data.metadata, "trade_code", None),
                            series.series_data.replacements,
                        ),
                        index=[
                            'insight_series_id',
                            'insight_id',
                            'insight_group',
                            'insight_type',
                            'applied_function',
                            'id',
                            'name',
                            'indicators',
                            'indicator',
                            'classification',
                            'unit',
                            'country',
                            'frequency',
                            'status',
                            'source',
                            'geo_info',
                            'remarks',
                            'mnemonic',
                            'is_forecast',
                            'has_vintage',
                            'headline_series',
                            'has_schedule',
                            'series_tag',
                            'start_date',
                            'end_date',
                            'multiplier_code',
                            'last_update_date',
                            'time_points_last_update_time',
                            'key_series',
                            'new_series',
                            'period_end',
                            'last_value',
                            'last_change',
                            'number_of_observations',
                            'has_continuous_series',
                            'trade_code',
                            'replacement'
                        ]
                    )

                    ps.attrs["insight_series_id"] = series.insight_series.id
                    ps.attrs["insight_id"] = series.insight_series.insight_id
                    result.append(ps)
                else:
                    ps = pd.Series(
                        data=[
                            {"value": tp.value, "last_update_date": pd.to_datetime(tp.last_update_time)}
                            for tp in getattr(series.series_data, "time_points", [])
                        ],
                        index=[
                            pd.to_datetime(tp.date)
                            for tp in getattr(series.series_data, "time_points", [])
                        ],
                        name=series.entity_id
                    )

                    ps.attrs["insight_series_id"] = series.insight_series.id

                    result.append(ps)

            return result

        for series in self.data:
            if not hasattr(series.series_data, "time_points") or series.series_data.time_points is None:
                ps = pd.Series(
                    data=(
                        getattr(series.series_data.metadata, "id", None),
                        getattr(series.series_data.metadata, "name", None),
                        getattr(series.series_data.metadata.country, "name", None),
                        getattr(series.series_data.metadata.classification, "name", None),
                        getattr(series.series_data.metadata.frequency, "name", None),
                        getattr(series.series_data.metadata, "start_date", None),
                        getattr(series.series_data.metadata, "end_date", None),
                        getattr(series.series_data.metadata, "last_update_time", None),
                        getattr(series.series_data.metadata.country, "last_value", None),
                        getattr(series.series_data.metadata.status, "name", None),
                        getattr(series.series_data.metadata.unit, "name", None),
                        ', '.join(geo_info.name for geo_info in getattr(series.series_data.metadata, "geo_info", [])),
                        series.series_data.replacements,
                    ),
                    index=[
                        'id',
                        'name',
                        'country',
                        'classification',
                        'frequency',
                        'start_date',
                        'end_date',
                        'last_update_date',
                        'last_value',
                        'status',
                        'unit',
                        'geo_info',
                        'replacement'
                    ]
                )
                result.append(ps)
            elif hasattr(series.series_data, "metadata") and series.series_data.metadata is not None:
                ps = pd.Series(
                    data=[d.value for d in series.series_data.time_points],
                    index=[d.date for d in series.series_data.time_points],
                    name=series.series_data.metadata.name
                )
                ps.attrs["insight_series_id"] = series.insight_series.id
                ps.attrs["insight_id"] = series.insight_series.insight_id
                ps.attrs["series_id"] = getattr(series.series_data.metadata, "id", None),
                ps.attrs["name"] = getattr(series.series_data.metadata, "name", None),
                ps.attrs["unit"] = getattr(series.series_data.metadata.unit, "name", None),
                ps.attrs["frequency"] = getattr(series.series_data.metadata.frequency, "name", None),
                ps.attrs["country"] = getattr(series.series_data.metadata.country, "name", None),
                result.append(ps)
            else:
                ps = pd.Series(
                    data=[d.value for d in series.series_data.time_points],
                    index=[d.date for d in series.series_data.time_points],
                    name=series.insight_series.id
                )
                ps.attrs["insight_series_id"] = series.insight_series.id
                ps.attrs["insight_id"] = series.insight_series.insight_id
                ps.attrs["series_code"] = getattr(series.series_data.layout[0], "series_code", None)
                result.append(ps)
        return result

    def as_pandas(self) -> DataFrame | Series:
        series_list = self._as_series_list()

        if self.data and type(self.data[0]) is InsightSeries:
            if (hasattr(self.data[0].series_data, "time_points") and self.data[0].series_data.time_points is not None):
                frames = []
                for s in series_list:
                    dataframe = s.apply(pd.Series)
                    dataframe["id"] = s.name
                    dataframe["insight_series_id"] = s.attrs["insight_series_id"]
                    dataframe["date"] = dataframe.index
                    frames.append(dataframe)

                df = pd.concat(frames, ignore_index=True)

                df = df[['last_update_date', 'value', 'date', 'id', 'insight_series_id']]  # set columns to specific order
                df["id"] = df["id"].astype(str)
                df["insight_series_id"] = df["insight_series_id"].astype(str)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df["last_update_date"] = pd.to_datetime(df["last_update_date"], errors="coerce")

                df = (df
                    .sort_values(by=['date', 'last_update_date', 'id', 'insight_series_id'], ascending=[False, False, False, False])
                    .reset_index(drop=True))

                return df
            else:
                df = pd.DataFrame(series_list)
                return df

        if len(series_list) == 1:
            return series_list[0]

        df = pd.DataFrame(series_list)

        if series_list and len(series_list[0].attrs) == 0:
            return df

        for series in series_list:
            insight_series_id = str(series.attrs["insight_series_id"])
            df.attrs[insight_series_id] = {}
            df.attrs[insight_series_id]["insight_id"] = series.attrs["insight_id"]
            if series.attrs.get('series_id', None) is None:
                df.attrs[insight_series_id]["series_code"] = series.attrs["series_code"]
            else:
                df.attrs[insight_series_id]["series_id"] = series.attrs["series_id"]
                df.attrs[insight_series_id]["name"] = series.attrs["name"]
                df.attrs[insight_series_id]["unit"] = series.attrs["unit"]
                df.attrs[insight_series_id]["frequency"] = series.attrs["frequency"]
                df.attrs[insight_series_id]["country"] = series.attrs["country"]
        return df

class CeicSeriesStatisticsResult(CeicSeriesResult, SeriesStatistics):
    _IDS_KEY = "id"

    def __init__(self, get_series_method, **kwargs):
        CeicSeriesResult.__init__(self, get_series_method, self._IDS_KEY, **kwargs)
        SeriesStatistics.__init__(self,
                                  min= self._initial_result.data[0].statistics.min,
                                  max= self._initial_result.data[0].statistics.max,
                                  median= self._initial_result.data[0].statistics.median,
                                  mean= self._initial_result.data[0].statistics.mean,
                                  standard_deviation= self._initial_result.data[0].statistics.standard_deviation,
                                  coefficient_variation= self._initial_result.data[0].statistics.coefficient_variation,
                                  variance= self._initial_result.data[0].statistics.variance,
                                  skewness= self._initial_result.data[0].statistics.skewness,
                                  kurtosis= self._initial_result.data[0].statistics.kurtosis,
                                  start_date= self._initial_result.data[0].statistics.start_date,
                                  end_date= self._initial_result.data[0].statistics.end_date,
                                  num_points= self._initial_result.data[0].statistics.num_points)


    def as_pandas(self):
        return self.convert_to_pandas(self._initial_result.data)

    def convert_to_pandas(self, series_result):
        if len(series_result) == 1:
            series = series_result[0]

            return pd.Series(
                data=(
                    series.entity_id,
                    series.statistics.coefficient_variation,
                    series.statistics.start_date,
                    series.statistics.end_date,
                    series.statistics.num_points,
                    series.statistics.mean,
                    series.statistics.variance,
                    series.statistics.standard_deviation,
                    series.statistics.skewness,
                    series.statistics.kurtosis,
                    series.statistics.min['date'],
                    series.statistics.min['value'],
                    series.statistics.median['date'],
                    series.statistics.median['value'],
                    series.statistics.max['date'],
                    series.statistics.max['value']
                ),
                index=[
                    'series_id',
                    'coefficient_variation',
                    'start_date',
                    'end_date',
                    'num_points',
                    'mean',
                    'variance',
                    'standard_deviation',
                    'skewness',
                    'kurtosis',
                    'min_date',
                    'min_value',
                    'median_date',
                    'median_value',
                    'max_date',
                    'max_value'
                ]
            )


        df = pd.DataFrame(
            [(
                series.entity_id,
                series.statistics.coefficient_variation,
                series.statistics.start_date,
                series.statistics.end_date,
                series.statistics.num_points,
                series.statistics.mean,
                series.statistics.variance,
                series.statistics.standard_deviation,
                series.statistics.skewness,
                series.statistics.kurtosis,
                series.statistics.min['date'],
                series.statistics.min['value'],
                series.statistics.median['date'],
                series.statistics.median['value'],
                series.statistics.max['date'],
                series.statistics.max['value']
            )
            for series in series_result
            ],
            columns=[
                'series_id',
                'coefficient_variation',
                'start_date',
                'end_date',
                'num_points',
                'mean',
                'variance',
                'standard_deviation',
                'skewness',
                'kurtosis',
                'min_date',
                'min_value',
                'median_date',
                'median_value',
                'max_date',
                'max_value'
            ]
        )
        df.attrs["total"] = len(series_result)

        return df

class CeicGetInsightSeriesListResult(CeicGetInsightSeriesResult):
    _IDS_KEY = "series_id"

#region Dictionaries
class CeicGetDictionariesResult(DictionaryResult):
    def __init__(self, dictionaries, **kwargs):
        DictionaryResult.__init__(self, dictionaries, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, dictionary_result):
        data_list = [
            {"name": attr_name, "value": len(value)}
            for attr_name, value in vars(dictionary_result[0]).items()
            if isinstance(value, list)
        ]

        df = pd.DataFrame(
            [(
                dic['name'].strip("_"),
                dic['value']
            )
                for dic in data_list
            ],
            columns=['name', 'total']
        )
        df.attrs["total"] = len(data_list)

        return df

class CeicGetIndicatorsResult(IndicatorsResult):
    def __init__(self, indicators, **kwargs):
        IndicatorsResult.__init__(self, indicators, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, indicators_result):
        df = pd.DataFrame(
            [(
                indicator.id,
                indicator.classification_id,
                indicator.name
            )
                for indicator in indicators_result
            ],
            columns=['id',
                     'classification_id',
                     'name']
        )
        df.attrs["total"] = len(indicators_result)

        return df

class CeicGetClassificationsResult(ClassificationsResult):
    def __init__(self, classifications, **kwargs):
        ClassificationsResult.__init__(self, classifications, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, classifications_result):
        df = pd.DataFrame(
            [(
                classification.id,
                classification.name
            )
                for classification in classifications_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(classifications_result)

        return df

class CeicGetClassificationIndicatorsResult(IndicatorsResult):
    def __init__(self, indicators, **kwargs):
        IndicatorsResult.__init__(self, indicators, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, indicators_result):
        df = pd.DataFrame(
            [(
                indicator.id,
                indicator.classification_id,
                indicator.name
            )
                for indicator in indicators_result
            ],
            columns=['id',
                     'classification_id',
                     'name']
        )
        df.attrs["total"] = len(indicators_result)

        return df

class CeicGetCountriesResult(CountriesResult):
    def __init__(self, countries, **kwargs):
        CountriesResult.__init__(self, countries, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, countries_result):
        df = pd.DataFrame(
            [(
                country.id,
                country.name
            )
                for country in countries_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(countries_result)

        return df

class CeicGetGeoResult(GeoResult):
    def __init__(self, geos, **kwargs):
        GeoResult.__init__(self, geos, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, geos_result):
        df = pd.DataFrame(
            [(
                geo.id,
                geo.iso_code,
                geo.title,
                geo.type
            )
                for geo in geos_result
            ],
            columns=['id',
                     'iso_code',
                     'title',
                     'type']
        )
        df.attrs["total"] = len(geos_result)

        return df

class CeicGetSourcesResult(SourcesResult):
    def __init__(self, sources, **kwargs):
        SourcesResult.__init__(self, sources, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, sources_result):
        df = pd.DataFrame(
            [(
                src.id,
                src.name
            )
                for src in sources_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(sources_result)

        return df

class CeicGetRegionsResult(RegionsResult):
    def __init__(self, regions, **kwargs):
        RegionsResult.__init__(self, regions, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, regions_result):
        df = pd.DataFrame(
            [(
                region.id,
                region.name
            )
                for region in regions_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(regions_result)

        return df

class CeicGetUnitsResult(UnitsResult):
    def __init__(self, units, **kwargs):
        UnitsResult.__init__(self, units, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, units_result):
        df = pd.DataFrame(
            [(
                region.id,
                region.name
            )
                for region in units_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(units_result)

        return df

class CeicGetFrequenciesResult(FrequenciesResult):
    def __init__(self, frequencies, **kwargs):
        FrequenciesResult.__init__(self, frequencies, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, frequencies_result):
        df = pd.DataFrame(
            [(
                region.id,
                region.name
            )
                for region in frequencies_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(frequencies_result)

        return df

class CeicGetStatusesResult(StatusesResult):
    def __init__(self, statuses, **kwargs):
        StatusesResult.__init__(self, statuses, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, statuses_result):
        df = pd.DataFrame(
            [(
                region.id,
                region.name
            )
                for region in statuses_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(statuses_result)

        return df

#endregion

#region Insights

class CeicGetInsightsCategoriesResult(InsightsCategoriesResult):
    def __init__(self, insights, **kwargs):
        InsightsCategoriesResult.__init__(self, insights, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, insights_result):
        df = pd.DataFrame(
            [(
                insight.id,
                insight.name,
            )
                for insight in insights_result
            ],
            columns=['id',
                     'name']
        )
        df.attrs["total"] = len(insights_result)

        return df

class CeicGetInsightsResult(InsightsResult):
    def __init__(self, insights, **kwargs):
        InsightsResult.__init__(self, insights, **kwargs)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, insights_result):
        df = pd.DataFrame(
            [(
                insight.category,
                insight.creation_time,
                insight.description,
                insight.id,
                insight.name,
                insight.subscribed,
                insight.creator.id,
                insight.creator.email,
                insight.creator.first_name,
                insight.creator.last_name
            )
                for insight in insights_result
            ],
            columns=[
                'category',
                'creation_time',
                'description',
                'id',
                'name',
                'subscribed',
                'creator_id',
                'creator_email',
                'creator_first_name',
                'creator_last_name'
            ]
        )
        df.attrs["total"] = len(insights_result)

        return df

#endregion

#region Layout
class CeicGetLayoutResult(LayoutItemsResult):
    def __init__(self, layout):
        LayoutItemsResult.__init__(self, layout)

    def as_pandas(self):
        return self.convert_to_pandas(self._data)

    def convert_to_pandas(self, layout_result):
        df = pd.DataFrame(
            [(
                layout.metadata.id,
                layout.metadata.name,
                int(layout.series_count),
                int(layout.ui.display_order)
            )
                for layout in layout_result.data
            ],
            columns=[
                'database_id',
                'database_name',
                'series_count',
                'display_order'
            ]
        )
        df.attrs["total"] = len(layout_result.data)

        return df

class CeicGetLayoutSeriesResult(LayoutItemsResult):
    def __init__(self, layout):
        LayoutItemsResult.__init__(self, layout)

    def as_pandas(self):
        series_list = self._as_series_list()
        if len(series_list) == 1:
            return series_list[0]

        df = pd.DataFrame(series_list)
        df.attrs["total"] = len(series_list)

        return pd.DataFrame(series_list)


    def _as_series_list(self) -> list[Series]:
        result = list()
        for series in self.data.data:
            ps = pd.Series(
                data=(
                    getattr(series.metadata, "id", None),
                    getattr(series.metadata, "name", None),
                    getattr(series.metadata.country, "name", None),
                    getattr(series.metadata.classification, "name", None),
                    getattr(series.metadata.frequency, "name", None),
                    getattr(series.metadata, "start_date", None),
                    getattr(series.metadata, "end_date", None),
                    getattr(series.metadata, "last_update_time", None),
                    getattr(series.metadata, "last_value", None),
                    getattr(series.metadata.status, "name", None),
                    getattr(series.metadata.unit, "name", None),
                    ', '.join(geo_info.name for geo_info in getattr(series.metadata, "geo_info", [])),
                    series.replacements,
                ),
                index=[
                    'id',
                    'name',
                    'country',
                    'classification',
                    'frequency',
                    'start_date',
                    'end_date',
                    'last_update_date',
                    'last_value',
                    'status',
                    'unit',
                    'geo_info',
                    'replacement'
                ]
            )
            result.append(ps)

        return result
#endregion
