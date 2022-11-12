import pickle
import spotipy
import requests
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import Image
from tkinter import ttk
from minisom import MiniSom
from sklearn.decomposition import PCA
from spotipy.oauth2 import SpotifyOAuth

# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and create an app to get the Client ID and Client Secret
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="37c0cd0d045d4728995a345cd3de949a",
                                               client_secret="039f36800bc44b198f56509a78e6b7fc",
                                               redirect_uri="http://example.com",
                                               scope="user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private user-library-read"))
user_id = sp.current_user()['id']
playlists = sp.current_user_playlists()


def getImage(url: str) -> Image:
    """Returns an Image object from a given url"""
    img = Image.open(requests.get(url, stream=True).raw).resize((32, 32))
    return img


def imgToArr(img: Image) -> np.array:
    """Returns a numpy array from a given Image object"""
    return np.array(img)


def PCAShift(dfOriginal: pd.DataFrame) -> pd.DataFrame:
    """Returns a dataframe with the PCA shifted values, (PCA is Principal Component Analysis, applied with sklearn.decomposition.PCA)"""
    df = dfOriginal.copy()
    pca = PCA(n_components=3)
    pca.fit(df[['band', 'Y', 'vividness']])
    df[['band', 'Y', 'vividness']] = pca.transform(df[['band', 'Y', 'vividness']])
    return df


def miniSOMSort(df: pd.DataFrame) -> list:
    """Implementation of the MiniSOM algorithm, returns a list of the sorted song IDs from a preprocessed dataframe"""
    df = df.sample(frac=1).reset_index(drop=True)
    df[['band', 'Y', 'vividness']] = (df[['band', 'Y', 'vividness']] - df[['band', 'Y', 'vividness']].mean()) / df[
        ['band', 'Y', 'vividness']].std()
    grid_size = int(5 * np.sqrt(len(df)))
    df = PCAShift(df)
    try:
        som = pickle.load(open('som.p', 'rb'))
    except FileNotFoundError:
        som = MiniSom(grid_size, grid_size, 3, sigma=2, learning_rate=.0225, neighborhood_function='triangle',
                      activation_distance='manhattan')
        som.train_random(df[['band', 'Y', 'vividness']].values, 500000, verbose=True)
        pickle.dump(som, open('som.p', 'wb'))
    qnt = som.quantization(df[['band', 'Y', 'vividness']].values)
    final = []
    for i in range(0, len(qnt)):
        final.append([df['track_id'][i], qnt[i][0], qnt[i][1]])
    final.sort(key=lambda x: (x[1], x[2]))
    return [node[0] for node in final]


def append_row(df: pd.DataFrame, row: list) -> None:
    """Appends a row to a dataframe"""
    df.loc[len(df.index)] = row


def normalize_color(rgb: tuple) -> tuple:
    """Normalizes a color tuple to a range of 0-1"""
    return tuple([x / 255 for x in rgb])


def rgb_to_hsY(rgb: tuple) -> tuple:
    """Converts an RGB color tuple to a Hue, Saturation, and Perceived Luminance tuple"""
    r, g, b = rgb
    maxc = max(r, g, b)
    minc = min(r, g, b)
    sumc = (maxc + minc)
    rangec = (maxc - minc)
    mid = sumc / 2.0
    if minc == maxc:
        return 0.0, mid, 0.0
    if mid <= 0.5:
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


def is_vivid(s: int, Y: int) -> bool:
    """Returns True if the color is vivid, False if not"""
    return s > 0.15 and 0.18 < Y < 0.95


def get_rainbow_band(hue: int, band_deg: int) -> int:
    """Returns the band of the rainbow that the color is in"""
    rb_hue = (hue + 30) % 360
    return rb_hue // band_deg


def get_image_rainbow_bands_and_perceived_brightness(image: Image, band_deg: int) -> tuple:
    """Returns a tuple of the rainbow bands, the perceived brightness, and the proportion of vivid colors in the image"""
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


def get_primary_band(bands: dict) -> int:
    """Returns the primary band of the image"""
    return max(bands, key=bands.get)


def getPlaylistID(playlistName: str) -> int or None:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    for playLists in playlists['items']:
        if playLists['name'] == playlistName:
            return playLists['id']
    return None


def sortPlaylist(playlistName: str) -> None:
    # Update progress bar
    progress['value'] = 5  # Update progress bar
    progress['maximum'] = 100
    progress.update()
    """Sorts a playlist by the MiniSOM algorithm"""
    # Processing the playlist
    playlistID = getPlaylistID(playlistName)
    source_playlist_id = f'spotify:user:spotifycharts:playlist:{playlistID}'
    pl_results = sp.playlist(source_playlist_id, fields='name,tracks.total')
    playlist_length = pl_results['tracks']['total']
    offset = 0
    playlist_items = []
    while offset < playlist_length:
        batch = sp.playlist_items(source_playlist_id, offset=offset,
                                  fields='items(track(id,track_number,album(images)))')
        batch_items = batch['items']
        offset += len(batch_items)
        playlist_items.extend(batch_items)
    # Update progress bar
    progress['value'] = 10  # Update progress bar
    progress.update()

    # Creating the dataframe
    df = pd.DataFrame(columns=['track_id', 'band', 'Y', 'vividness', 'track_number', 'img_url'])
    progress['value'] = 15  # Update progress bar
    progress.update()

    # Processing the images
    for item in playlist_items:
        track = item['track']
        track_id = track['id']
        track_number = track['track_number']
        cover_image_url = track['album']['images'][-1]['url']
        track_image = Image.open(requests.get(cover_image_url, stream=True).raw).resize((32, 32))
        bands, Y, vividness = get_image_rainbow_bands_and_perceived_brightness(track_image, band_deg=30)
        primary_band = get_primary_band(bands)
        append_row(df, [track_id, primary_band, Y, vividness, track_number, cover_image_url])
    progress['value'] = 20  # Update progress bar
    progress.update()

    # Applying a PCA Shift then sorting using the MiniSOM algorithm
    df = PCAShift(df)
    df = df.sort_values(by=['band', 'Y', 'vividness', 'track_number'])
    progress['value'] = 35  # Update progress bar
    progress.update()
    sorted_track_ids = miniSOMSort(df)
    progress['value'] = 65  # Update progress bar
    progress.update()

    # Reorders the playlist
    offset = 0
    if len(sorted_track_ids) > 100:
        while True:
            sp.playlist_remove_all_occurrences_of_items(playlistID, sorted_track_ids[offset:offset + 100])
            offset += 100
            if offset >= len(sorted_track_ids):
                break
    else:
        sp.playlist_remove_all_occurrences_of_items(playlist_id=playlistID, items=sorted_track_ids)
    for track_id in sorted_track_ids:
        sp.playlist_add_items(playlistID, [track_id])
    progress['value'] = 0
    progress.update()


if __name__ == '__main__':
    """Main function for GUI"""
    window = tk.Tk()
    window.title("Spotify Playlist Sorter")
    window.geometry('500x200')
    playlist_names = []
    for playLists in playlists['items']:
        playlist_names.append(playLists['name'])
    clicked = tk.StringVar()
    clicked.set(playlist_names[0])
    drop = tk.OptionMenu(window, clicked, *playlist_names)
    drop.pack()
    sortButton = tk.Button(window, text="Sort", command=lambda: sortPlaylist(clicked.get()))
    sortButton.pack()
    progress = ttk.Progressbar(window, orient=tk.HORIZONTAL, length=200, mode='determinate')
    progress.pack()
    window.mainloop()
