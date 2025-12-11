import os
import json
import six
import getpass
import sys

from ceic_api_client.apis.sessions_api import SessionsApi
import ceic_api_client.version as Version

import ceic_api_client.facade.pyceic_exception as ceic_exception
from ceic_api_client.facade.pyceic_requests_facade import CeicRequestsFacade

from ceic_api_client.facade.pyceic_warnings import PackageUpdateWarning
from ceic_api_client.facade.pyceic_warnings import AbuseWarning
from ceic_api_client.facade.pyceic_warnings import DebugInfoWarning
from ceic_api_client.facade.pyceic_infos import LoginSuccess


class CeicSessionsFacade(object):

    _SESSIONS_DIR_NAME = ".ceic_python_sdk"
    _SESSIONS_FILE_NAME = "session.json"
    _APPLICATION_ID = "CEIC_Python"
    _APPLICATION_VERSION = Version.VERSION

    _GET_USERNAME_PROMPT = "CEIC Username: "
    _GET_PASSWORD_PROMPT = "CEIC Password: "

    # TODO: Remove optional ceic_requests parameter - make it required
    def __init__(self, ceic_configuration, ceic_requests):
        self._sessions_api = SessionsApi(ceic_configuration.api_client)
        self._ceic_requests = ceic_requests

        self._persist_sessions_file()

        self._sessions_file = self._load_sessions_file()

        self._update_warning = PackageUpdateWarning(ceic_configuration)
        self._debug_warning = DebugInfoWarning(ceic_configuration)
        self._abuse_warning = AbuseWarning()
        self._login_success = LoginSuccess()

        self._username = None
        self._password = None

        self._last_successful_session = None

    @staticmethod
    def get_sessions_file_path():
        sessions_file_path = os.path.join(
            CeicSessionsFacade._get_user_home_dir(),
            CeicSessionsFacade._SESSIONS_DIR_NAME,
            CeicSessionsFacade._SESSIONS_FILE_NAME
        )

        return sessions_file_path

    @property
    def session_id(self):
        return self._sessions_file["session"] if "session" in self._sessions_file else None

    def login(self, username=None, password=None):
        username = username if username is not None else self._username
        password = password if password is not None else self._password
        if username is None or password is None:
            self._update_warning.show_update_warning_if_needed()
            self._get_user_login_from_console(username, password)
            return self.login()

        try:
            self._logout()
        except ceic_exception.CeicException:
            pass

        session_id = self._try_call_api_login(username, password)
        self._save_session(session_id)

        self._username = username
        self._password = password
        self._last_successful_session = session_id

        self._login_success.show_info()
        self._debug_warning.show_debug_warning(session_id)

        return self

    def logout(self):
        self._username = None
        self._password = None

        self._logout()

        return self

    def _get_user_login_from_console(self, username, password):
        if username is None:
            username = self._get_username_cmd()

        if password is None:
            password = self._get_password_cmd()

        self._username = username
        self._password = password

    def _logout(self):
        if self.session_id is not None:
            try:
                self._call_api_logout(self._sessions_file["session"])
                self._remove_session()
            except ceic_exception.CeicNoActiveSessionsException:
                self._remove_session()
                exc = ceic_exception.CeicSessionExpiredException()
                exc.__suppress_context__ = True
                raise exc
        else:
            exc = ceic_exception.CeicNoActiveSessionsException()
            exc.__suppress_context__ = True
            raise exc

    @staticmethod
    def _persist_sessions_file():
        CeicSessionsFacade._persist_sessions_dir()

        file_path = CeicSessionsFacade.get_sessions_file_path()
        if not os.path.exists(file_path):
            with open(file_path, 'w+') as f:
                f.write("{}")

    @staticmethod
    def _persist_sessions_dir():
        file_path = CeicSessionsFacade.get_sessions_file_path()
        dir_path = os.path.dirname(file_path)

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    @staticmethod
    def _get_user_home_dir():
        if CeicSessionsFacade._os_is_windows():
            user_home_path = os.environ["LOCALAPPDATA"]
        else:
            user_home_path = os.environ["HOME"]

        return user_home_path

    @staticmethod
    def _os_is_windows():
        return os.name == 'nt'

    @staticmethod
    def _str_is_null_or_empty(str_obj):
        return str_obj is None or \
               str_obj.strip() == ""

    def _load_sessions_file(self):
        with open(self.get_sessions_file_path(), 'r') as opened_file:
            json_file = json.load(opened_file)

        return json_file

    def _save_sessions_file(self):
        with open(self.get_sessions_file_path(), 'w+') as opened_file:
            json.dump(self._sessions_file, opened_file)

    def _try_call_api_login(self, username, password):
        try:
            session_id = self._call_api_login(username, password)
        except ceic_exception.CeicException as le:
            self._username = None
            self._password = None
            raise le

        return session_id

    def _call_api_login(self, username, password):
        response = self._ceic_requests.make_request(
            self._sessions_api.login,
            login=username,
            password=password,
            application=self._APPLICATION_ID,
            application_version=self._APPLICATION_VERSION
        )
        session_id = response.data.session

        return session_id

    def _call_api_logout(self, session_id):
        response = self._ceic_requests.make_request(
            self._sessions_api.logout,
            session=session_id
        )

        return response

    def _save_session(self, session_id):
        self._sessions_file["session"] = session_id
        self._save_sessions_file()

    def _remove_session(self):
        self._sessions_file.pop("session")
        self._save_sessions_file()

    def _get_username_cmd(self):
        if six.PY2:
            username = raw_input(self._GET_USERNAME_PROMPT)
        else:
            username = input(self._GET_USERNAME_PROMPT)

        return username

    def _get_password_cmd(self):
        password = getpass.getpass(prompt=self._GET_PASSWORD_PROMPT)

        return password
