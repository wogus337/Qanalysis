import os

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


from ceic_api_client.apis.layout_api import LayoutApi
from ceic_api_client.facade.pyceic_decorators import CeicLayoutRequestDecorator as LayoutDecorator
from ceic_api_client.facade.pyceic_facade_models import CeicGetLayoutResult
from ceic_api_client.facade.pyceic_facade_models import CeicGetLayoutSeriesResult

class CeicLayoutFacade(object):

    def __init__(self, ceic_configuration, ceic_requets_facade):
        self._ceic_configuration = ceic_configuration
        self._ceic_requests_facade = ceic_requets_facade

        self._layouts_api = LayoutApi(api_client=self._ceic_configuration.api_client)

    @LayoutDecorator.layout
    def get_layout_databases(self, **kwargs):
        self._validate_parameters("getLayoutDatabases", **kwargs)

        result = CeicGetLayoutResult(
            self._ceic_requests_facade.make_request(
                self._layouts_api.get_layout_databases,
                **kwargs
            )
        )

        return result

    def download_footnotes(self, **kwargs):

        node_code = kwargs["node_code"]
        download_path = kwargs["download_path"]

        footnote_metadata = self._ceic_requests_facade.make_request(
            self._layouts_api.get_layout_footnote_metadata,
            **{"node_code": node_code}
        )

        footnote = footnote_metadata.data.footnote
        resources = footnote_metadata.data.resources
        download_root = os.path.join(download_path, footnote.file_name.split('.')[0])

        self._download_footnote(footnote, download_root)
        self._download_footnote_resources(resources, download_root)

    def _download_footnote(self, footnote, download_root):
        if not os.path.exists(download_root):
            os.mkdir(download_root)

        footnote_file = os.path.join(download_root, footnote.file_name)
        self._download_file(footnote.download_link, footnote_file)

    def _download_footnote_resources(self, resources, download_root):
        for resource in resources:
            resource_file = os.path.join(download_root, resource.file_name)
            self._download_file(resource.download_link, resource_file)

    def _download_file(self, download_url, file_path):
        response = urlopen(download_url)
        data = response.read()
        try:
            data = data.decode('utf-8')
            self._write_text_file(data, file_path)
        except UnicodeDecodeError:
            self._write_bytes_file(data, file_path)

    @staticmethod
    def _write_text_file(data, file_path):
        with open(file_path, 'w+b') as f:
            try:
                f.write(data.encode('utf-8'))
            except TypeError:
                f.write(data)

    @staticmethod
    def _write_bytes_file(data, file_path):
        with open(file_path, 'w+b') as f:
            f.write(data)

    @LayoutDecorator.layout
    def get_layout_database_topics(self, **kwargs):
        self._validate_parameters("getLayoutTopics", **kwargs)

        result = CeicGetLayoutResult(
            self._ceic_requests_facade.make_request(
                self._layouts_api.get_layout_topics,
                **kwargs
            )
        )

        return result

    @LayoutDecorator.layout
    def get_layout_topic_sections(self, **kwargs):
        self._validate_parameters("getLayoutSections", **kwargs)

        result = CeicGetLayoutResult(
            self._ceic_requests_facade.make_request(
                self._layouts_api.get_layout_sections,
                **kwargs
            )
        )

        return result

    @LayoutDecorator.layout
    def get_layout_section_tables(self, **kwargs):
        self._validate_parameters("getLayoutTables", **kwargs)

        result = CeicGetLayoutResult(
            self._ceic_requests_facade.make_request(
                self._layouts_api.get_layout_tables,
                **kwargs
            )
        )

        return result

    @LayoutDecorator.layout
    def get_layout_table_series(self, **kwargs):
        self._validate_parameters("getLayoutSeries", **kwargs)

        result = CeicGetLayoutSeriesResult(
            self._ceic_requests_facade.make_request(
                self._layouts_api.get_layout_series,
                **kwargs
            )
        )

        return result

    def _validate_parameters(self, operation_id, **kwargs):
        for parameter_validator in self._ceic_configuration.parameter_validators:
            parameter_validator.validate_parameters(operation_id, **kwargs)
