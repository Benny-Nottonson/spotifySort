"""This program is used to sort a Spotify playlist by the color of the album art of the songs"""
from io import BytesIO
from queue import Queue
from functools import cache
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from tkinter import Tk, StringVar, OptionMenu, Button, HORIZONTAL, ttk
from numpy import sum as numpy_sum, ndarray, array, zeros, matmul, max as numpy_max, bincount, \
    count_nonzero
from PIL import Image
from requests import get
from skimage.measure import label
from spotipy import Spotify, SpotifyOAuth
from cv2 import cv2

# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and
# create an app to get the Client ID and Client Secret
SCOPE = "user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private " \
        "user-library-read"
sp = Spotify(auth_manager=SpotifyOAuth(client_id="",
                                       client_secret="",
                                       redirect_uri="https://example.com",
                                       scope=SCOPE))
userID = sp.current_user()['id']
playlists = sp.current_user_playlists()


def get_playlist_id(playlist_name: str) -> str:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    return next((ids['id'] for ids in playlists['items'] if ids['name'] == playlist_name), None)


def get_playlist_items(playlist_id: str) -> tuple:
    """Returns a list of the items in a playlist"""
    playlist_length = sp.playlist(playlist_id, fields='name,tracks.total')['tracks']['total']
    offset = 0
    playlist_items = []
    while offset < playlist_length:
        batch = sp.playlist_items(playlist_id, offset=offset,
                                  fields='items(track(id,track_number,album(images)))')
        batch_items = batch['items']
        offset += len(batch_items)
        playlist_items.extend(batch_items)
    return tuple(playlist_items)


def reorder_playlist(playlist_id: str, sorted_track_ids: list) -> None:
    """Reorders a playlist to match the order of the sorted track IDs"""
    offset = 0
    length = len(sorted_track_ids)
    if length > 100:
        while True:
            sp.playlist_remove_all_occurrences_of_items(playlist_id,
                                                        sorted_track_ids[offset:offset + 100])
            offset += 100
            if offset >= length:
                break
    else:
        sp.playlist_remove_all_occurrences_of_items(playlist_id=playlist_id, items=sorted_track_ids)
    offset = 0
    if length > 100:
        while True:
            sp.playlist_add_items(playlist_id, sorted_track_ids[offset:offset + 100], offset)
            offset += 100
            if offset >= length:
                break
    else:
        sp.playlist_add_items(playlist_id, sorted_track_ids, offset)


def ccv(img_url: str) -> tuple:
    """Calculates the Color Coherence Vector of an image"""
    img = get_image_from_url(img_url)
    threshold = round(0.01 * img.shape[0] * img.shape[1])
    mac = rgb_to_mac(img)
    n_blobs, blob = blob_extract(array(mac))
    table = [[mac[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0]
             for i in range(blob.shape[0]) for j in range(blob.shape[1])
             for table in [[[0, 0] for _ in range(0, n_blobs)]]]
    color_coherence_vector = [[0, 0] for _ in range(24)]
    for color_index, size in ((entry[0], entry[1]) for entry in table):
        color_coherence_vector[color_index][size >= threshold] += size
    color_coherence_vector = tuple(map(tuple, color_coherence_vector))
    return color_coherence_vector


def blob_extract(mac: ndarray) -> tuple:
    """Extracts blobs from a MAC image"""
    blob = label(mac, connectivity=1) + 1
    n_blobs = numpy_max(blob)
    if n_blobs > 1:
        count = bincount(blob.ravel())[2:]
        n_blobs -= 1
        n_blobs += count_nonzero(count > 1)
    return n_blobs, blob


def rgb_to_mac(img: ndarray) -> list:
    """Converts an RGB image to a MAC image"""
    return [
        [int(find_minimum_macbeth(tuple(img[i][j]), lab_distance_3d)) for j in range(img.shape[1])]
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
def bgr_to_lab(bgr_color: tuple) -> tuple:
    """Converts a BGR color to a LAB color"""
    bgr_color = bgr_color[::-1]
    rgb_color = [
        100 * (((element / 255.0 + 0.055) / 1.055) ** 2.4 if element / 255.0 > 0.04045
               else (element / 255.0) / 12.92) for element in bgr_color]
    rgb2_xyz_mat = array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = [round(element, 4) for element in matmul(rgb2_xyz_mat, rgb_color)]
    xyz = matmul(array([[1 / 95.047, 0, 0], [0, 1 / 100.0, 0], [0, 0, 1 / 108.883]]), xyz)
    xyz = [element ** 0.33333 if element > 0.008856 else 7.787 * element + 16.0 / 116 for element in
           xyz]
    xyz2_lab_mat = array([[0, 116, 0], [500, -500, 0], [0, 200, -200]])
    final = tuple(round(element, 4) for element in matmul(xyz2_lab_mat, xyz) + array([-16, 0, 0]))
    return final


@cache
def ccv_distance(ccv_one: tuple, ccv_two: tuple) -> ndarray:
    """Calculates the distance between two CCV vectors"""
    return numpy_sum([3 * abs(ccv_one[i][0] - ccv_two[i][0]) + abs(ccv_one[i][1] - ccv_two[i][1])
                      for i in range(0, len(ccv_one))])


def lab_distance_3d(lab_one: tuple, lab_two: tuple) -> float:
    """Calculates the distance between two LAB colors"""
    lab_one, lab_two = bgr_to_lab(tuple(lab_one)), bgr_to_lab(tuple(lab_two))
    return ((lab_one[0] - lab_two[0]) ** 2.0) + ((lab_one[1] - lab_two[1]) ** 2.0) + (
            (lab_one[2] - lab_two[2]) ** 2.0) ** 0.5


@cache
def get_image_from_url(url: str) -> ndarray:
    """Gets an image from a URL and converts it to BGR"""
    response = get(url, timeout=5)
    img = Image.open(BytesIO(response.content))
    img = array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (16, 16))
    return img


def resort_loop(loop, func, total):
    """Reorders a loop to minimize the distance between the colors"""
    n_loop = [(i,) + tpl[1:] for i, tpl in enumerate(loop)]
    loop_length = len(n_loop)
    distance_matrix = zeros((loop_length, loop_length))
    for i in range(loop_length):
        for j in range(i):
            dist = func(n_loop[i][1], n_loop[j][1])
            distance_matrix[i][j] = distance_matrix[j][i] = dist
    pass_count = 0
    while pass_count < 150:  # max number of passes
        moving_loop_entry = n_loop.pop(-1)
        moving_index = moving_loop_entry[0]
        min_index = -1
        val = -1
        for i in range(loop_length - 1):
            if i in (0, loop_length - 2):
                behind_index = n_loop[loop_length - 3][0]
                ahead_index = n_loop[0][0]
            else:
                behind_index = n_loop[i - 1][0]
                ahead_index = n_loop[i + 1][0]
            avg_of_distances_i = (distance_matrix[behind_index, moving_index] + distance_matrix[
                ahead_index, moving_index]) / 2
            if total:
                total_distance_i = numpy_sum(
                    [distance_matrix[n_loop[k - 1][0], n_loop[k][0]] for k in
                     range(loop_length - 1)]
                )
                total_distance_i += distance_matrix[n_loop[0][0], n_loop[-1][0]]
                if min_index == -1 or total_distance_i < val:
                    val = total_distance_i
                    min_index = i
            else:
                if min_index == -1 or avg_of_distances_i < val:
                    val = avg_of_distances_i
                    min_index = i
        if min_index == loop_length - 2:
            n_loop.insert(0, moving_loop_entry)
        else:
            n_loop.insert(min_index + 1, moving_loop_entry)
        pass_count += 1
    return [(loop[tpl[0]][0],) + tpl[1:] for tpl in n_loop]


class App:
    """The main application class"""

    def __init__(self):
        self.window = Tk()
        self.window.title("Spotify Playlist Sorter")
        self.window.geometry('210x100')
        self.playlist_names = [playList['name'] for playList in playlists['items']]
        self.clicked = StringVar()
        self.clicked.set(self.playlist_names[0])
        drop = OptionMenu(self.window, self.clicked, *self.playlist_names)
        drop.pack()
        sort_button = Button(self.window, text="Sort",
                             command=lambda: self.sort_playlist(
                                 get_playlist_id(self.clicked.get())))
        sort_button.pack()
        self.progress = ttk.Progressbar(self.window, orient=HORIZONTAL, length=200,
                                        mode='determinate')
        self.progress.pack()
        self.window.mainloop()

    def update_progress_bar(self, value):
        """Updates the progress bar to the given value"""""
        self.progress['value'] = value
        self.window.update_idletasks()

    def loop_sort(self, entries: tuple, func: callable) -> list:
        """Sorts a list of entries by the function func"""
        loop = deque()
        loop.append(entries[0])
        entries = deque(entries[1:])
        length = len(entries)
        update_progress_bar = self.update_progress_bar
        for i in range(1, length + 1):
            item_one = loop[-1]
            item_two = entries
            j = find_minimum(item_one, func, item_two)
            loop.append(item_two[j])
            item_two.rotate(-j)
            item_two.popleft()
            update_progress_bar(60 + (i / length) * 20)
        return list(loop)

    def ccv_sort(self, playlist_id: str) -> list:
        """Sorts a playlist using CCVs"""
        entries = self.make_ccv_collection(get_playlist_items(playlist_id), ccv)
        self.update_progress_bar(60)
        loop = self.loop_sort(entries, ccv_distance)
        self.update_progress_bar(80)
        loop = resort_loop(loop, ccv_distance, True)
        return [loop[i][0] for i in range(0, len(loop))]

    def make_ccv_collection(self, playlist_items: tuple, data: callable) -> list:
        """Makes a collection of CCVs from a playlist"""""
        total = len(playlist_items)
        tuple_collection = []
        result_queue = Queue()

        def process_item(to_process: dict) -> None:
            """Processes an item in the playlist"""
            track = to_process['track']
            track_id = track['id']
            url = track['album']['images'][-1]['url']
            result_queue.put((track_id, data(url)))

        with ThreadPoolExecutor(max_workers=8) as executor:
            for index, item in enumerate(playlist_items):
                self.update_progress_bar(20 + (index / total) * 40)
                executor.submit(process_item, item)

        while not result_queue.empty():
            tuple_collection.append(result_queue.get())

        return tuple_collection

    def sort_playlist(self, playlist_id: str) -> None:
        """Sorts a playlist by the given algorithm"""
        self.update_progress_bar(20)
        sorted_track_ids = self.ccv_sort(playlist_id)
        self.update_progress_bar(90)
        reorder_playlist(playlist_id, sorted_track_ids)
        self.update_progress_bar(0)
        ccv_distance.cache_clear()


if __name__ == '__main__':
    app = App()
