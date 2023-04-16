"""This module contains the SpotifyAPIManager class which is used to authenticate and manage the
Spotify API."""
from webbrowser import open as open_browser
from datetime import datetime, timedelta
from urllib.parse import urlencode
from json import load, dump
from os import path, getcwd
from base64 import b64encode
from requests import post, get, delete


class SpotifyAPIManager:
    """This class is used to authenticate and manage the Spotify API"""

    # pylint: disable=too-many-instance-attributes
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, scope: str) -> None:
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope
        self.token = None
        self.code = None
        self.token_expires = datetime.now()
        self.token_url = "https://accounts.spotify.com/api/token"
        self.api_url = "https://api.spotify.com/v1/"
        self.cache_path = path.join(getcwd(), ".cache")
        if path.exists(self.cache_path):
            with open(self.cache_path, encoding='utf-8') as file:
                cache = load(file)
                if cache["expires_at"] > datetime.now().timestamp():
                    self.token = cache["access_token"]
                    self.token_expires = datetime.fromtimestamp(cache["expires_at"])
                    self.code = cache["authorization_code"]
                else:
                    self.code = self.get_authorization_code()
                    self.token = self.get_access_token()
                    self.save_to_cache()
        else:
            self.code = self.get_authorization_code()
            self.token = self.get_access_token()
            self.save_to_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_token_to_cache()

    def save_to_cache(self) -> None:
        """Saves the token and code to a cache file"""
        cache = {
            "access_token": self.token,
            "expires_at": self.token_expires.timestamp(),
            "authorization_code": self.code,
        }
        with open(self.cache_path, "w", encoding='utf-8') as file:
            dump(cache, file)

    def save_token_to_cache(self) -> None:
        """Saves the token to a cache file"""
        cache = {"access_token": self.token, "expires_at": self.token_expires.timestamp(),
                 "authorization_code": self.code}
        with open(self.cache_path, "w", encoding='utf-8') as file:
            dump(cache, file)

    def get_authorize_url(self) -> str:
        """Returns the url to authorize the app"""
        params = {
            'client_id': self.client_id,
            'response_type': 'code',
            'redirect_uri': self.redirect_uri,
            'scope': self.scope
        }
        url = 'https://accounts.spotify.com/authorize?' + urlencode(params)
        return url

    def get_authorization_code(self) -> str:
        """Returns the authorization code from the url"""
        url = 'https://accounts.spotify.com/authorize'
        url += '?response_type=code'
        url += f'&client_id={self.client_id}'
        url += f'&redirect_uri={self.redirect_uri}'
        url += f'&scope={self.scope}'
        open_browser(url)
        code = input("Enter the full url: ")
        code = code.split('=')[1]
        return code

    def get_access_token(self) -> str:
        """Returns the access token"""
        if self.token and datetime.now() < self.token_expires:
            return self.token
        headers = {
            'Authorization': f'Basic {self.get_auth_header()}'
        }
        data = {
            'grant_type': 'authorization_code',
            'code': self.code,
            'redirect_uri': self.redirect_uri
        }
        response = post(self.token_url, headers=headers, data=data, timeout=5)
        if response.status_code == 200:
            self.token = response.json()['access_token']
            expires_in = response.json()['expires_in']
            self.token_expires = datetime.now() + timedelta(seconds=expires_in)
            return self.token
        raise ValueError("Authentication failed")

    def refresh_access_token(self, refresh_token) -> str:
        """Refreshes the access token"""
        headers = {
            'Authorization': f'Basic {self.get_auth_header()}'
        }
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token
        }
        response = post(self.token_url, headers=headers, data=data, timeout=5)
        if response.status_code == 200:
            self.token = response.json()['access_token']
            expires_in = response.json()['expires_in']
            self.token_expires = datetime.now() + timedelta(seconds=expires_in)
            return self.token
        raise ValueError("Token refresh failed")

    def get_auth_header(self) -> str:
        """Returns the authorization header"""
        auth_header = b64encode(f"{self.client_id}:{self.client_secret}".encode())
        return auth_header.decode()

    def get_access_token_header(self) -> dict:
        """Returns the access token header"""
        access_token = self.get_access_token()
        access_token_header = {
            'Authorization': f'Bearer {access_token}'
        }
        return access_token_header

    def get_resource(self, endpoint: str, params: list = None, fields: list = None) -> dict:
        """Returns the resource from the endpoint"""
        url = self.api_url + endpoint
        headers = self.get_access_token_header()
        if fields is not None:
            headers['Accept'] = 'application/json'
        response = get(url, headers=headers, params=params, data=None, timeout=5)
        if response.status_code == 200:
            if fields is not None:
                return response.json()[fields]
            return response.json()
        raise ValueError("Request failed")


class SpotifyAPI:
    """This class is used to interact with the Spotify API"""

    def __init__(self, api_manager: SpotifyAPIManager) -> None:
        self.api_manager = api_manager

    def current_user(self) -> dict:
        """Returns the current user"""
        endpoint = "me"
        return self.api_manager.get_resource(endpoint)

    def current_user_playlists(self) -> dict:
        """Returns the current user's playlists"""
        endpoint = "me/playlists"
        return self.api_manager.get_resource(endpoint)

    def playlist(self, playlist_id: str, fields: list = None) -> dict:
        """Returns the playlist with the given id"""
        endpoint = f"playlists/{playlist_id}"
        params = {}
        if fields is not None:
            params['fields'] = fields
        return self.api_manager.get_resource(endpoint, params=params)

    def playlist_items(self, playlist_id: str, offset: int = None, fields: list = None) -> dict:
        """Returns the items of the playlist with the given id"""
        endpoint = f"playlists/{playlist_id}/tracks"
        params = {}
        if offset is not None:
            params['offset'] = offset
        if fields is not None:
            params['fields'] = fields
        return self.api_manager.get_resource(endpoint, params=params)

    def playlist_remove_all_occurrences_of_items(self, playlist_id: str, items: list) -> bool:
        """Removes all occurrences of the given items from the playlist"""
        endpoint = f"{self.api_manager.api_url}playlists/{playlist_id}/tracks"
        headers = {"Authorization": f"Bearer {self.api_manager.token}"}
        data = {"uris": [f"spotify:track:{tid}" for tid in items]}
        response = delete(endpoint, headers=headers, json=data, timeout=5)
        if response.status_code == 200:
            return True
        print(f"Error {response.status_code} occurred: {response.text}")
        raise ValueError("Request failed")

    def playlist_add_items(self, playlist_id:  str, track_ids: list, position: int = None):
        """Adds the given items to the playlist in order"""
        endpoint = f"{self.api_manager.api_url}playlists/{playlist_id}/tracks"
        headers = {"Authorization": f"Bearer {self.api_manager.token}"}
        data = {"uris": [f"spotify:track:{tid}" for tid in track_ids]}
        if position is not None:
            data["position"] = position
        response = post(endpoint, headers=headers, json=data, timeout=5)
        if response.status_code != 201:
            print(f"Error {response.status_code} occurred: {response.text}")
            raise ValueError("Request failed")
        return response.json()
