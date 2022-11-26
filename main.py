from PIL import Image
from numpy import sqrt
from tkinter import ttk
from requests import get
from spotipy import Spotify
from pandas import DataFrame
from pickle import load, dump
from minisom import MiniSom
from sklearn.decomposition import PCA
from spotipy.oauth2 import SpotifyOAuth
from tkinter import Tk, StringVar, OptionMenu, Button, HORIZONTAL

global progress

# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and create an app to get the Client ID and Client Secret
sp = Spotify(auth_manager=SpotifyOAuth(client_id="37c0cd0d045d4728995a345cd3de949a",
                                       client_secret="02b21f43b4ef4774986cf0f29b4888c5",
                                       redirect_uri="http://example.com",
                                       scope="user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private user-library-read"))
userID = sp.current_user()['id']
playlists = sp.current_user_playlists()


def updateProgressBar(newVal: int) -> None:
    """Updates the slider value"""
    progress['value'] = newVal
    progress.update()


def shuffleAndNormalize(df: DataFrame) -> DataFrame:
    """Shuffles the dataframe and normalizes the values"""
    df = df.sample(frac=1).reset_index(drop=True)
    df[['band', 'Y', 'vividness']] = (df[['band', 'Y', 'vividness']] - df[['band', 'Y', 'vividness']].mean()) / df[
        ['band', 'Y', 'vividness']].std()
    return df


def appendRow(df: DataFrame, row: list) -> None:
    """Appends a row to a dataframe"""
    df.loc[len(df.index)] = row


def normalizeColor(rgb: tuple) -> tuple:
    """Normalizes a color tuple to a range of 0-1"""
    return tuple([x / 255 for x in rgb])


def isVivid(s: float, Y: float) -> bool:
    """Returns True if the color is vivid, False if not"""
    return s > 0.15 and 0.18 < Y < 0.95


def getPlaylistID(playlistName: str) -> int or None:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    for ids in playlists['items']:
        if ids['name'] == playlistName:
            return ids['id']
    return None


def getPlaylistItems(playlistID: str) -> list:
    """Returns a list of the items in a playlist"""
    sourcePlaylistID = f'spotify:user:spotifycharts:playlist:{playlistID}'
    playlistResults = sp.playlist(sourcePlaylistID, fields='name,tracks.total')
    playlistLength = playlistResults['tracks']['total']
    offset = 0
    playlistItems = []
    while offset < playlistLength:
        batch = sp.playlist_items(sourcePlaylistID, offset=offset,
                                  fields='items(track(id,track_number,album(images)))')
        batchItems = batch['items']
        offset += len(batchItems)
        playlistItems.extend(batchItems)
    return playlistItems


def reorderPlaylist(playlistID: str, sortedTrackIDs: list) -> None:
    """Reorders a playlist to match the order of the sorted track IDs"""
    offset = 0
    if len(sortedTrackIDs) > 100:
        while True:
            sp.playlist_remove_all_occurrences_of_items(playlistID, sortedTrackIDs[offset:offset + 100])
            offset += 100
            if offset >= len(sortedTrackIDs):
                break
    else:
        sp.playlist_remove_all_occurrences_of_items(playlist_id=playlistID, items=sortedTrackIDs)
    offset = 0
    if len(sortedTrackIDs) > 100:
        while True:
            sp.playlist_add_items(playlistID, sortedTrackIDs[offset:offset + 100], offset)
            offset += 100
            if offset >= len(sortedTrackIDs):
                break
    else:
        sp.playlist_add_items(playlistID, sortedTrackIDs, offset)


def makeDataframe(playlistItems: list) -> DataFrame:
    """Returns a dataframe in containing the track IDs, the rainbow bands, the perceived brightness, and the proportion of vivid colors in the image"""
    df = DataFrame(columns=['trackID', 'band', 'Y', 'vividness'])
    for item in playlistItems:
        track = item['track']
        trackID = track['id']
        trackImage = Image.open(get(track['album']['images'][-1]['url'], stream=True).raw).resize((32, 32))
        bands, Y, vividness = getBandsAndBrightness(trackImage)
        primaryBand = max(bands, key=bands.get)
        appendRow(df, [trackID, primaryBand, Y, vividness])
    return df


def createUI() -> None:
    window = Tk()
    window.title("Spotify Playlist Sorter")
    window.geometry('210x100')
    playlistNames = []
    for playLists in playlists['items']:
        playlistNames.append(playLists['name'])
    clicked = StringVar()
    clicked.set(playlistNames[0])
    drop = OptionMenu(window, clicked, *playlistNames)
    drop.pack()
    sortButton = Button(window, text="Sort", command=lambda: sortPlaylist(getPlaylistID(clicked.get())))
    sortButton.pack()
    global progress
    progress = ttk.Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
    progress.pack()
    window.mainloop()


def PCAShift(dfOriginal: DataFrame) -> DataFrame:
    """Returns a dataframe with the PCA shifted values, (PCA is Principal Component Analysis, applied with sklearn.decomposition.PCA)"""
    df = dfOriginal.copy()
    pca = PCA(n_components=3)
    pca.fit(df[['band', 'Y', 'vividness']])
    df[['band', 'Y', 'vividness']] = pca.transform(df[['band', 'Y', 'vividness']])
    return df


def miniSOMSort(df: DataFrame) -> list:
    """Implementation of the MiniSOM algorithm, returns a list of the sorted song IDs from a preprocessed dataframe"""
    try:
        som = load(open('som.p', 'rb'))
    except FileNotFoundError:
        gridSize = int(5 * sqrt(len(df)))
        som = MiniSom(gridSize, gridSize, 3, learning_rate=0.001, sigma=1.0, neighborhood_function='cosine', activation_distance='bubble')
        som.train_random(df[['band', 'Y', 'vividness']].values, 5000000, verbose=False)
        dump(som, open('som.p', 'wb'))
    qnt = som.quantization(df[['band', 'Y', 'vividness']].values)
    final = []
    for i in range(0, len(qnt)):
        final.append([df['trackID'][i], qnt[i][0], qnt[i][1]])
    final.sort(key=lambda x: (x[1], x[2]))
    return [node[0] for node in final]


def getBandsAndBrightness(image: Image) -> tuple:
    """Returns a tuple of the rainbow bands, the perceived brightness, and the proportion of vivid colors in the image"""
    pixels = image.convert('RGB').getdata()
    bandCenter = 360 // 30
    allBands = dict.fromkeys(range(bandCenter), 0)
    vividBands = dict.fromkeys(range(bandCenter), 0)
    perceivedLuminance = 0.0
    vividPixels = 0
    for pixel in pixels:
        rgb = normalizeColor(pixel)
        h, s, Y = RGBtoHSY(rgb)
        perceivedLuminance += Y
        ab = ((h + 30) % 360) // 30
        allBands[ab] += 1
        if isVivid(s, Y):
            vb = ((h + 30) % 360) // 30
            vividBands[vb] += 1
            vividPixels += 1
    if sum(vividBands.values()) > 0:
        bands = vividBands
        bandPixels = vividPixels
    else:
        bands = allBands
        bandPixels = len(pixels)
    bands = {k: v / bandPixels for (k, v) in bands.items()}
    perceivedLuminance = perceivedLuminance / len(pixels)
    vividness = vividPixels / len(pixels)
    return bands, perceivedLuminance, vividness


def RGBtoHSY(rgb: tuple) -> tuple:
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


def sortPlaylist(playlistID: str) -> None:
    """Sorts a playlist by the MiniSOM algorithm"""
    updateProgressBar(5)
    playlistItems = getPlaylistItems(playlistID)
    df = makeDataframe(playlistItems)
    updateProgressBar(20)
    df = PCAShift(df)
    updateProgressBar(35)
    sortedTrackIDs = miniSOMSort(shuffleAndNormalize(df))
    updateProgressBar(65)
    reorderPlaylist(playlistID, sortedTrackIDs)
    updateProgressBar(0)


if __name__ == '__main__':
    createUI()
