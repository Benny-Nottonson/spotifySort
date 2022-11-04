import base64
from io import BytesIO
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import requests
from PIL import Image

# IDS NEED TO BE INPUT BY THE USER, THIS SHOULD BE DONE THROUGH CREATING A SPOTIFY DEVELOPER APPLICATION
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="",
                                               client_secret="",
                                               redirect_uri="",
                                               scope=""))
user_id = sp.current_user()['id']
playlists = sp.current_user_playlists()


def append_row(df:pd.DataFrame, row:list) -> None:
    df.loc[len(df.index)] = row


def normalize_color(rgb:tuple) -> tuple:
    return tuple([x / 255 for x in rgb])


def rgb_to_hsY(rgb:tuple) -> tuple:
    r, g, b = rgb
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = (maxc + minc)
    rangec = (maxc - minc)
    l = sumc / 2.0
    if minc == maxc:
        return 0.0, l, 0.0
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0 - sumc)
    rc = (maxc - r) / rangec
    gc = (maxc - g) / rangec
    bc = (maxc - b) / rangec
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = int((h / 6.0) % 1.0 * 360.0)
    Y = 0.2126 * r ** 2.2 + 0.7152 * g ** 2.2 + 0.0722 * b ** 2.2

    return h, s, Y


def is_vivid(s:float, Y:float) -> bool:
    return s > 0.15 and 0.18 < Y < 0.95


def get_rainbow_band(hue:int, band_deg:int) -> int:
    rb_hue = (hue + 30) % 360
    return rb_hue // band_deg


def get_image_rainbow_bands_and_perceived_brightness(image:Image, band_deg:int) -> tuple:
    pixels = image.convert('RGB').getdata()
    band_cnt = 360 // band_deg
    all_bands = dict.fromkeys(range(band_cnt), 0)
    vivid_bands = dict.fromkeys(range(band_cnt), 0)
    perceived_luminance = 0.0
    vivid_pixels = 0
    for pixel in pixels:
        rgb = normalize_color(pixel)
        h, s, Y = rgb_to_hsY(rgb)
        perceived_luminance += Y
        ab = get_rainbow_band(h, band_deg)
        all_bands[ab] += 1
        if is_vivid(s, Y):
            vb = get_rainbow_band(h, band_deg)
            vivid_bands[vb] += 1
            vivid_pixels += 1
    if sum(vivid_bands.values()) > 0:
        bands = vivid_bands
        band_pixels = vivid_pixels
    else:
        bands = all_bands
        band_pixels = len(pixels)
    bands = {k: v / band_pixels for (k, v) in bands.items()}
    perceived_luminance = perceived_luminance / len(pixels)
    vividness = vivid_pixels / len(pixels)
    return bands, perceived_luminance, vividness


def get_primary_band(bands:dict) -> int:
    return max(bands, key=bands.get)


def getPlaylistID(playlistName:str) -> str:
    for lists in playlists['items']:
        if lists['name'] == playlistName:
            return lists['id']
    return None


def sortPlaylist(playlistName:str) -> None:
    source_playlist_id = f'spotify:user:spotifycharts:playlist:{getPlaylistID(playlistName)}'
    pl_results = sp.playlist(source_playlist_id, fields='name,tracks.total')
    playlist_name = pl_results['name']
    playlist_length = pl_results['tracks']['total']
    offset = 0
    playlist_items = []
    while offset < playlist_length:
        batch = sp.playlist_items(source_playlist_id, offset=offset,
                                  fields='items(track(id,track_number,album(images)))')
        batch_items = batch['items']
        offset += len(batch_items)
        playlist_items.extend(batch_items)
    df = pd.DataFrame(columns=['track_id', 'band', 'Y', 'vividness', 'track_number', 'img_url'])
    for item in playlist_items:
        track = item['track']
        track_id = track['id']
        track_number = track['track_number']
        cover_image_url = track['album']['images'][-1]['url']
        track_image = Image.open(requests.get(cover_image_url, stream=True).raw).resize((10, 10))
        bands, Y, vividness = get_image_rainbow_bands_and_perceived_brightness(track_image, band_deg=30)
        primary_band = get_primary_band(bands)
        append_row(df, [track_id, primary_band, Y, vividness, track_number, cover_image_url])
    df = df.sort_values(by=['band', 'Y', 'track_number'])
    sorted_track_ids = df['track_id'].tolist()
    newPlaylist = sp.user_playlist_create(user=user_id, public=True,
                                          name=playlist_name)

    playlist_image_url = sp.playlist(source_playlist_id, fields='images')['images'][0]['url']
    playlist_image = Image.open(requests.get(playlist_image_url, stream=True).raw)
    buffered = BytesIO()
    playlist_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    sp.playlist_upload_cover_image(newPlaylist['id'], img_str)

    offset = 0
    while offset < playlist_length:
        sp.playlist_add_items(playlist_id=newPlaylist['id'], items=sorted_track_ids[offset:offset + 100])
        offset += 100


for lists in playlists['items']:
    print(lists['name'])
choice = input("Enter playlist name: ")

for playlist in playlists['items']:
    if playlist['name'] == choice:
        print("Sorting...")
        sortPlaylist(choice)
        print("Done!")
        break
