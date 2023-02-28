from PIL import Image
from queue import Queue
from requests import get
from spotipy import Spotify
from threading import Thread
from functools import cache
from spotipy.oauth2 import SpotifyOAuth
from numpy import sum, ndarray, array, zeros, matmul, where
from cv2 import cvtColor, COLOR_RGB2BGR, connectedComponents
from tkinter import Tk, StringVar, OptionMenu, Button, HORIZONTAL, ttk

# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and create an app to get
# the Client ID and Client Secret
sp = Spotify(auth_manager=SpotifyOAuth(client_id="",
                                       client_secret="",
                                       redirect_uri="https://example.com",
                                       scope="user-library-modify playlist-modify-public ugc-image-upload "
                                             "playlist-modify-private user-library-read"))
userID = sp.current_user()['id']
playlists = sp.current_user_playlists()
seen_images = {}


def get_playlist_id(playlistName: str) -> str:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    return next((ids['id'] for ids in playlists['items'] if ids['name'] == playlistName), None)


def get_playlist_items(playlistID: str) -> tuple:
    """Returns a list of the items in a playlist"""
    playlistLength = sp.playlist(playlistID, fields='name,tracks.total')['tracks']['total']
    offset = 0
    playlistItems = []
    while offset < playlistLength:
        batch = sp.playlist_items(playlistID, offset=offset,
                                  fields='items(track(id,track_number,album(images)))')
        batchItems = batch['items']
        offset += len(batchItems)
        playlistItems.extend(batchItems)
    return tuple(playlistItems)


def reorder_playlist(playlistID: str, sortedTrackIDs: list) -> None:
    """Reorders a playlist to match the order of the sorted track IDs"""
    offset = 0
    length = len(sortedTrackIDs)
    if length > 100:
        while True:
            sp.playlist_remove_all_occurrences_of_items(playlistID, sortedTrackIDs[offset:offset + 100])
            offset += 100
            if offset >= length:
                break
    else:
        sp.playlist_remove_all_occurrences_of_items(playlist_id=playlistID, items=sortedTrackIDs)
    offset = 0
    if length > 100:
        while True:
            sp.playlist_add_items(playlistID, sortedTrackIDs[offset:offset + 100], offset)
            offset += 100
            if offset >= length:
                break
    else:
        sp.playlist_add_items(playlistID, sortedTrackIDs, offset)


def ccv(img_url: str) -> tuple:
    """Calculates the Color Coherence Vector of an image"""
    if img_url in seen_images:
        return seen_images[img_url]
    img = pil_to_cv2(Image.open(get(img_url, stream=True).raw).resize((64, 64)))
    threshold = round(0.01 * img.shape[0] * img.shape[1])
    mac = rgb_to_mac(img)
    n_blobs, blob = blob_extract(array(mac))
    table = [[0, 0] for _ in range(0, n_blobs)]
    table = [[mac[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0] for i in range(blob.shape[0]) for
             j in range(blob.shape[1])]
    CCV = [[0, 0] for _ in range(0, 24)]
    for entry in table:
        color_index = entry[0]
        size = entry[1]
        if size >= threshold:
            CCV[color_index][0] += size
        else:
            CCV[color_index][1] += size
    CCV = tuple(tuple(entry) for entry in CCV)
    seen_images[img_url] = CCV
    return CCV


def blob_extract(mac: ndarray) -> tuple:
    """Extracts blobs from a MAC image"""
    blob = zeros([mac.shape[0], mac.shape[1]]).astype('uint32').tolist()
    n_blobs = 0
    for index in range(0, 24):
        count, labels = connectedComponents(where(mac == index, 1, 0).astype('uint8'))
        labels[labels > 0] += n_blobs
        blob += labels
        if count > 1:
            n_blobs += (count - 1)
    return n_blobs, blob


def rgb_to_mac(img: ndarray) -> list:
    """Converts an RGB image to a MAC image"""
    return [[int(find_minimum(tuple(img[i][j]), lab_distance_3d)) for j in range(img.shape[1])]
            for i in range(img.shape[0])]


@cache
def find_minimum(p_entry: tuple, func: callable, q_entries=None) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    if q_entries is None:
        q_entries = (('dark skin', (115, 82, 68)), ('light skin', (194, 150, 130)),
                     ('blue sky', (98, 122, 157)),
                     ('foliage', (87, 108, 67)), ('blue flower', (133, 128, 177)),
                     ('bluish green', (103, 189, 170)),
                     ('orange', (214, 126, 44)), ('purplish blue', (80, 91, 166)),
                     ('moderate red', (193, 90, 99)),
                     ('purple', (94, 60, 108)), ('yellow green', (157, 188, 64)),
                     ('orange yellow', (224, 163, 46)),
                     ('blue', (56, 61, 150)), ('green', (70, 148, 73)), ('red', (175, 54, 60)),
                     ('yellow', (231, 199, 31)),
                     ('magenta', (187, 86, 149)), ('cyan', (8, 133, 161)),
                     ('white 9.5', (243, 243, 242)),
                     ('neutral 8', (200, 200, 200)), ('neutral 6.5', (160, 160, 160)),
                     ('neutral 5', (122, 122, 121)),
                     ('neutral 3.5', (85, 85, 85)), ('black 2', (52, 52, 52)))
        p = p_entry
    else:
        p = p_entry[1]
    val = -1
    minIndex = -1
    for i in range(len(q_entries)):
        q = q_entries[i][1]
        f = func(p, tuple(q))
        if val == -1 or f < val:
            minIndex, val = i, f
    return minIndex


@cache
def bgr_to_lab(v: tuple) -> tuple:
    """Converts a BGR color to a LAB color"""
    v = v[::-1]
    RGB = [
        100 * (((element / 255.0 + 0.055) / 1.055) ** 2.4 if element / 255.0 > 0.04045 else (element / 255.0) / 12.92)
        for element in v]
    RGB2XYZ_MAT = array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    XYZ = [round(element, 4) for element in matmul(RGB2XYZ_MAT, RGB)]
    XYZ = matmul(array([[1 / 95.047, 0, 0], [0, 1 / 100.0, 0], [0, 0, 1 / 108.883]]), XYZ)
    XYZ = [element ** 0.33333 if element > 0.008856 else 7.787 * element + 16.0 / 116 for element in XYZ]
    XYZ2LAB_MAT = array([[0, 116, 0], [500, -500, 0], [0, 200, -200]])
    return tuple([round(element, 4) for element in matmul(XYZ2LAB_MAT, XYZ) + array([-16, 0, 0])])


@cache
def ccv_distance(V1: tuple, V2: tuple) -> ndarray:
    """Calculates the distance between two CCV vectors"""
    return sum([3 * abs(V1[i][0] - V2[i][0]) + abs(V1[i][1] - V2[i][1]) for i in range(0, len(V1))])


def lab_distance_3d(A: tuple, B: tuple) -> float:
    """Calculates the distance between two LAB colors"""
    A, B = bgr_to_lab(tuple(A)), bgr_to_lab(tuple(B))
    return ((A[0] - B[0]) ** 2.0) + ((A[1] - B[1]) ** 2.0) + ((A[2] - B[2]) ** 2.0) ** 0.5


def pil_to_cv2(img: Image) -> ndarray:
    """Converts a PIL image to a CV2 image"""
    return cvtColor(array(img), COLOR_RGB2BGR)


class App:
    def __init__(self):
        self.window = Tk()
        self.window.title("Spotify Playlist Sorter")
        self.window.geometry('210x100')
        self.playlistNames = [playList['name'] for playList in playlists['items']]
        self.clicked = StringVar()
        self.clicked.set(self.playlistNames[0])
        drop = OptionMenu(self.window, self.clicked, *self.playlistNames)
        drop.pack()
        sortButton = Button(self.window, text="Sort",
                            command=lambda: self.sort_playlist(get_playlist_id(self.clicked.get())))
        sortButton.pack()
        self.progress = ttk.Progressbar(self.window, orient=HORIZONTAL, length=200, mode='determinate')
        self.progress.pack()
        self.window.mainloop()

    def updateProgressBar(self, value):
        """Updates the progress bar to the given value"""""
        self.progress['value'] = value
        self.window.update_idletasks()

    def loop_sort(self, entries: tuple, func: callable) -> list:
        """Sorts a list of entries by the function func"""
        entries = list(entries)
        loop = []
        length = len(entries)
        find_minimum.cache_clear()
        for i in range(length):
            if i == 0:
                loop.append(entries.pop(0))
            else:
                a1 = loop[i - 1]
                b1 = tuple(entries)
                loop.append(entries.pop(find_minimum(a1, func, b1)))
            self.updateProgressBar(60 + (i / length) * 20)
        return loop

    def ccv_sort(self, playlistID: str) -> list:
        """Sorts a playlist using CCVs"""
        entries = self.make_ccv_collection(get_playlist_items(playlistID), ccv)
        self.updateProgressBar(60)
        loop = self.loop_sort(entries, ccv_distance)
        self.updateProgressBar(80)
        return [loop[i][0] for i in range(0, len(loop))]

    def make_ccv_collection(self, playlistItems: tuple, data: callable) -> list:
        """Returns a list of tuples containing the track IDs and the """
        total = len(playlistItems)
        tupleCollection = []
        resultQueue = Queue()

        def process_item(toProcess: dict) -> None:
            """Processes an item in the playlistItems list"""
            track = toProcess['track']
            trackID = track['id']
            url = track['album']['images'][-1]['url']
            resultQueue.put((trackID, data(url)))

        threads = []
        for ix, item in enumerate(playlistItems):
            self.updateProgressBar(20 + (ix / total) * 40)
            t = Thread(target=process_item, args=(item,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        while not resultQueue.empty():
            tupleCollection.append(resultQueue.get())

        return tupleCollection

    def sort_playlist(self, playlistID: str) -> None:
        """Sorts a playlist by the given algorithm"""
        self.updateProgressBar(20)
        sortedTrackIDs = self.ccv_sort(playlistID)
        self.updateProgressBar(90)
        reorder_playlist(playlistID, sortedTrackIDs)
        self.updateProgressBar(0)


if __name__ == '__main__':
    app = App()
