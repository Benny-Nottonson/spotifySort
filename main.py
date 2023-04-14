from PIL import Image
from io import BytesIO
from queue import Queue
from requests import get
from spotipy import Spotify
from functools import cache
from collections import deque
from skimage.measure import label
from spotipy.oauth2 import SpotifyOAuth
from cv2 import cvtColor, COLOR_RGB2BGR, resize
from concurrent.futures import ThreadPoolExecutor
from tkinter import Tk, StringVar, OptionMenu, Button, HORIZONTAL, ttk
from numpy import sum, ndarray, array, zeros, matmul, max, bincount, count_nonzero


# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and create an app to get
# the Client ID and Client Secret
scope = "user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private user-library-read"
sp = Spotify(auth_manager=SpotifyOAuth(client_id="",
                                       client_secret="",
                                       redirect_uri="https://example.com",
                                       scope=scope))
userID = sp.current_user()['id']
playlists = sp.current_user_playlists()


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
    img = get_image_from_url(img_url)
    threshold = round(0.01 * img.shape[0] * img.shape[1])
    mac = rgb_to_mac(img)
    n_blobs, blob = blob_extract(array(mac))
    table = [[mac[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0] for i in range(blob.shape[0]) for
             j in range(blob.shape[1]) for table in [[[0, 0] for _ in range(0, n_blobs)]]]
    CCV = [[0, 0] for _ in range(24)]
    for color_index, size in ((entry[0], entry[1]) for entry in table):
        CCV[color_index][size >= threshold] += size
    CCV = tuple(map(tuple, CCV))
    return CCV


def blob_extract(mac: ndarray) -> tuple:
    """Extracts blobs from a MAC image"""
    blob = label(mac, connectivity=1) + 1
    n_blobs = max(blob)
    if n_blobs > 1:
        count = bincount(blob.ravel())[2:]
        n_blobs -= 1
        n_blobs += count_nonzero(count > 1)
    return n_blobs, blob


def rgb_to_mac(img: ndarray) -> list:
    """Converts an RGB image to a MAC image"""
    return [[int(find_minimum_macbeth(tuple(img[i][j]), lab_distance_3d)) for j in range(img.shape[1])]
            for i in range(img.shape[0])]


@cache
def find_minimum_macbeth(p_entry: tuple, func: callable) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
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
    return (min(enumerate([func(p_entry, tuple(q[1])) for q in q_entries]), key=lambda x: x[1]))[0]


def find_minimum(p_entry: tuple, func: callable, q_entries: tuple) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    return (min(enumerate(q_entries), key=lambda x: func(p_entry[1], x[1][1])))[0]


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
    final = tuple([round(element, 4) for element in matmul(XYZ2LAB_MAT, XYZ) + array([-16, 0, 0])])
    return final


@cache
def ccv_distance(V1: tuple, V2: tuple) -> ndarray:
    """Calculates the distance between two CCV vectors"""
    return sum([3 * abs(V1[i][0] - V2[i][0]) + abs(V1[i][1] - V2[i][1]) for i in range(0, len(V1))])


def lab_distance_3d(A: tuple, B: tuple) -> float:
    """Calculates the distance between two LAB colors"""
    A, B = bgr_to_lab(tuple(A)), bgr_to_lab(tuple(B))
    return ((A[0] - B[0]) ** 2.0) + ((A[1] - B[1]) ** 2.0) + ((A[2] - B[2]) ** 2.0) ** 0.5


@cache
def get_image_from_url(url: str) -> ndarray:
    """Gets an image from a URL and converts it to BGR"""
    response = get(url)
    img = Image.open(BytesIO(response.content))
    img = array(img)
    img = cvtColor(img, COLOR_RGB2BGR)
    img = resize(img, (16, 16))
    return img


def reSort(loop, func, total):
    n_loop = [(i,) + tpl[1:] for i, tpl in enumerate(loop)]
    loop_length = len(n_loop)
    distance_matrix = zeros((loop_length, loop_length))
    for i in range(loop_length):
        for j in range(i):
            dist = func(n_loop[i][1], n_loop[j][1])
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    max_pass_count = 150
    pass_count = 0
    while pass_count < max_pass_count:
        moving_loop_entry = n_loop.pop(-1)
        moving_index = moving_loop_entry[0]
        minIndex = -1
        val = -1
        for i in range(loop_length - 1):
            if i == 0 or i == loop_length - 2:
                behind_index = n_loop[loop_length - 3][0]
                ahead_index = n_loop[0][0]
            else:
                behind_index = n_loop[i - 1][0]
                ahead_index = n_loop[i + 1][0]
            distance_behind = distance_matrix[behind_index, moving_index]
            distance_ahead = distance_matrix[ahead_index, moving_index]
            avg_of_distances_i = (distance_behind + distance_ahead) / 2
            if total:
                total_distance_i = sum([distance_matrix[n_loop[k - 1][0], n_loop[k][0]] for k in range(loop_length - 1)]
                                       )
                total_distance_i += distance_matrix[n_loop[0][0], n_loop[-1][0]]
                if minIndex == -1 or total_distance_i < val:
                    val = total_distance_i
                    minIndex = i
            else:
                if minIndex == -1 or avg_of_distances_i < val:
                    val = avg_of_distances_i
                    minIndex = i
        if minIndex == loop_length - 2:
            n_loop.insert(0, moving_loop_entry)
        else:
            n_loop.insert(minIndex + 1, moving_loop_entry)
        pass_count += 1
    new_loop = [(loop[tpl[0]][0],) + tpl[1:] for tpl in n_loop]
    return new_loop


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
        loop = deque()
        loop.append(entries[0])
        entries = deque(entries[1:])
        length = len(entries)
        updateProgressBar = self.updateProgressBar
        for i in range(1, length + 1):
            a1 = loop[-1]
            b1 = entries
            j = find_minimum(a1, func, b1)
            loop.append(b1[j])
            b1.rotate(-j)
            b1.popleft()
            updateProgressBar(60 + (i / length) * 20)
        return list(loop)

    def ccv_sort(self, playlistID: str) -> list:
        """Sorts a playlist using CCVs"""
        entries = self.make_ccv_collection(get_playlist_items(playlistID), ccv)
        self.updateProgressBar(60)
        loop = self.loop_sort(entries, ccv_distance)
        self.updateProgressBar(80)
        loop = reSort(loop, ccv_distance, True)
        return [loop[i][0] for i in range(0, len(loop))]

    def make_ccv_collection(self, playlistItems: tuple, data: callable) -> list:
        total = len(playlistItems)
        tupleCollection = []
        resultQueue = Queue()

        def process_item(toProcess: dict) -> None:
            track = toProcess['track']
            trackID = track['id']
            url = track['album']['images'][-1]['url']
            resultQueue.put((trackID, data(url)))

        with ThreadPoolExecutor(max_workers=8) as executor:
            for ix, item in enumerate(playlistItems):
                self.updateProgressBar(20 + (ix / total) * 40)
                executor.submit(process_item, item)

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
        ccv_distance.cache_clear()


if __name__ == '__main__':
    app = App()
