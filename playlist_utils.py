"""Utilities for working with Spotify playlists"""
from io import BytesIO
from PIL import Image
from customtkinter import CTkImage
from spotify_api import SpotifyAPI, SpotifyAPIManager, public_get as client_get

SCOPE = (
    "user-library-modify playlist-modify-public playlist-modify-private user-library-read"
)

sp = SpotifyAPI(
    api_manager=SpotifyAPIManager(
        client_id="",
        client_secret="",
        redirect_uri="https://example.com",
        scope=SCOPE,
    )
)

userID = sp.current_user()["id"]
playlists = sp.current_user_playlists()


def get_playlist_id(playlist_name: str) -> str or None:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    for ids in playlists["items"]:
        if ids["name"] == playlist_name:
            return ids["id"]
    return None


def get_playlist_items(playlist_id: str) -> tuple[str, int]:
    """Returns a list of the items in a playlist"""
    playlist = sp.playlist(playlist_id, fields="name,tracks.total")
    playlist_length = playlist["tracks"]["total"]
    playlist_items = sp.playlist_items(
        playlist_id,
        fields="items(track(id,track_number,album(images)))",
        playlist_length=playlist_length,
    )
    return tuple(playlist_items)


def get_playlist_art(playlist_id: str) -> CTkImage:
    """Returns the playlist preview image"""
    playlist = sp.playlist(playlist_id, fields="images")
    playlist_art_url = playlist["images"][0]["url"]
    playlist_art_image = Image.open(BytesIO(client_get(playlist_art_url, timeout=5).content))
    playlist_art_image = CTkImage(playlist_art_image, size=(250, 250))
    return playlist_art_image


def reorder_playlist(playlist_id: str, sorted_track_ids: list[str]) -> None:
    """Reorders a playlist to match the order of the sorted track IDs"""
    sp.playlist_remove_all_occurrences_of_items(playlist_id, sorted_track_ids)
    sp.playlist_add_items(playlist_id, sorted_track_ids)


def remove_duplicates(items: list) -> tuple:
    """Removes duplicate items from a return API call"""
    seen = set()
    final_items = []
    for item in items:
        track_id = item["track"]["id"]
        if track_id not in seen:
            seen.add(track_id)
            final_items.append(item)
    return tuple(final_items)
