"""This program is used to sort a Spotify playlist by the color of the album art of the songs"""
from io import BytesIO
from queue import Queue
from functools import cache
from threading import Thread
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from numpy import (
    sum as numpy_sum,
    ndarray,
    array,
    zeros,
    max as numpy_max,
    bincount,
    count_nonzero,
    argmin,
    empty,
    dot,
)
from PIL import Image
from skimage.measure import label
from cv2 import cvtColor, resize, COLOR_RGB2BGR
from customtkinter import (
    CTkImage,
    CTk,
    CTkFrame,
    CTkLabel,
    CTkButton,
    CTkComboBox,
    CTkProgressBar,
)
from spotify_api import SpotifyAPI, SpotifyAPIManager, public_get as client_get


# The Below Code is for the Spotify API, you will need to create a Spotify Developer Account and
# create an app to get the Client ID and Client Secret
SCOPE = (
    "user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private "
    "user-library-read"
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


def get_playlist_id(playlist_name: str) -> str:
    """Returns the ID of a playlist, or None if it doesn't exist"""
    return next(
        (ids["id"] for ids in playlists["items"] if ids["name"] == playlist_name), None
    )


def get_playlist_items(playlist_id: str) -> tuple:
    """Returns a list of the items in a playlist"""
    playlist_length = sp.playlist(playlist_id, fields="name,tracks.total")["tracks"][
        "total"
    ]
    playlist_items = sp.playlist_items(
        playlist_id,
        fields="items(track(id,track_number,album(images)))",
        playlist_length=playlist_length,
    )
    return tuple(playlist_items)


def get_playlist_art(playlist_id: str) -> CTkImage:
    """Returns the playlist preview image"""
    playlist_art = sp.playlist(playlist_id, fields="images")["images"][0]["url"]
    playlist_art = Image.open(BytesIO(client_get(playlist_art, timeout=5).content))
    playlist_art = CTkImage(playlist_art, size=(250, 250))
    return playlist_art


def reorder_playlist(playlist_id: str, sorted_track_ids: list) -> None:
    """Reorders a playlist to match the order of the sorted track IDs"""
    sp.playlist_remove_all_occurrences_of_items(playlist_id, sorted_track_ids)
    sp.playlist_add_items(playlist_id, sorted_track_ids)


def ccv(img_url: str) -> tuple:
    """Calculates the Color Coherence Vector of an image"""
    img = get_image_from_url(img_url)
    threshold = round(0.01 * img.shape[0] * img.shape[1])
    mac = rgb_to_mac(img)
    n_blobs, blob = blob_extract(array(mac))
    table = [
        [mac[i][j], table[blob[i][j] - 1][1] + 1] if blob[i][j] != 0 else [0, 0]
        for i in range(blob.shape[0])
        for j in range(blob.shape[1])
        for table in [[[0, 0] for _ in range(0, n_blobs)]]
    ]
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


def rgb_to_mac(img: ndarray) -> ndarray:
    """Converts an RGB image to a MAC image"""
    pixels = img.reshape(-1, 3)
    mac_indices = empty(pixels.shape[0], dtype="uint8")
    for i, pixel in enumerate(pixels):
        mac_indices[i] = find_minimum_macbeth(tuple(pixel), lab_distance_3d)
    mac_image = mac_indices.reshape(img.shape[:2])
    return mac_image


def find_minimum(p_entry: tuple, func: callable, q_entries: tuple) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    return (min(enumerate(q_entries), key=lambda x: func(p_entry[1], x[1][1])))[0]


@cache
def find_minimum_macbeth(p_entry: tuple, func: callable) -> int:
    """Finds the value of q_entries that minimizes the function func(p_entry, q_entry)"""
    macbeth_colors = array(
        [
            [115, 82, 68],
            [194, 150, 130],
            [98, 122, 157],
            [87, 108, 67],
            [133, 128, 177],
            [103, 189, 170],
            [214, 126, 44],
            [80, 91, 166],
            [193, 90, 99],
            [94, 60, 108],
            [157, 188, 64],
            [224, 163, 46],
            [56, 61, 150],
            [70, 148, 73],
            [175, 54, 60],
            [231, 199, 31],
            [187, 86, 149],
            [8, 133, 161],
            [243, 243, 242],
            [200, 200, 200],
            [160, 160, 160],
            [122, 122, 121],
            [85, 85, 85],
            [52, 52, 52],
        ]
    )
    distances = array([func(p_entry, tuple(q_entry)) for q_entry in macbeth_colors])
    return argmin(distances)


@cache
def bgr_to_lab(bgr_color: tuple) -> tuple:
    """Converts a BGR color to a CIELAB color"""
    bgr_color = array(bgr_color, dtype=float) / 255.0
    mask = bgr_color > 0.04045
    bgr_color[mask] = ((bgr_color[mask] + 0.055) / 1.055) ** 2.4
    bgr_color[~mask] /= 12.92
    bgr_matrix = array(
        [[0.1805, 0.3576, 0.4124], [0.0722, 0.7152, 0.2126], [0.9505, 0.1192, 0.0193]]
    )
    xyz = dot(bgr_matrix, bgr_color)
    xyz_n = array([0.95047, 1.0, 1.08883])
    xyz_r = (xyz / xyz_n) ** (1 / 3)
    mask = xyz_r <= 0.008856
    xyz_r[mask] = (7.787 * xyz_r[mask]) + (16 / 116)
    lab_color = (
        116 * xyz_r[1] - 16,
        500 * (xyz_r[0] - xyz_r[1]),
        200 * (xyz_r[1] - xyz_r[2]),
    )
    return lab_color


@cache
def ccv_distance(ccv_one: tuple, ccv_two: tuple) -> ndarray:
    """Calculates the distance between two CCV vectors"""
    return numpy_sum(
        [
            3 * abs(ccv_one[i][0] - ccv_two[i][0]) + abs(ccv_one[i][1] - ccv_two[i][1])
            for i in range(0, len(ccv_one))
        ]
    )


def lab_distance_3d(lab_one: tuple, lab_two: tuple) -> float:
    """Estimates the distance between two BGR colors in LAB space"""
    l_1, a_1, b_1 = bgr_to_lab(lab_one)
    l_2, a_2, b_2 = bgr_to_lab(lab_two)
    return abs(l_1 - l_2) + abs(a_1 - a_2) + abs(b_1 - b_2)


def get_image_from_url(url: str) -> ndarray:
    """Gets an image from a URL and converts it to BGR"""
    response = client_get(url, timeout=5)
    img = Image.open(BytesIO(response.content))
    img = cvtColor(array(img), COLOR_RGB2BGR)
    img = resize(img, (16, 16))
    return img


def get_n_loop(loop: list) -> list:
    """Converts a loop to a list of tuples with the index and the color"""
    return [(i,) + tpl[1:] for i, tpl in enumerate(loop)]


def generate_distance_matrix(n_loop: list, func: callable, loop_length: int) -> ndarray:
    """Generates a distance matrix for a loop"""
    distance_matrix = zeros((loop_length, loop_length))
    for i in range(loop_length):
        for j in range(i):
            distance_matrix[i][j] = distance_matrix[j][i] = func(
                n_loop[i][1], n_loop[j][1]
            )
    return distance_matrix


def resort_loop(loop, func, loop_length):
    """Reorders a loop to minimize the distance between the colors"""
    n_loop = deque(get_n_loop(loop))
    distance_matrix = generate_distance_matrix(n_loop, func, loop_length)
    while True:
        moving_loop_entry = n_loop.pop()
        behind_indices = array([n_loop[i - 1][0] for i in range(1, loop_length - 1)])
        ahead_indices = array([n_loop[i + 1][0] for i in range(loop_length - 2)])
        behind_distances = distance_matrix[behind_indices, moving_loop_entry[0]]
        ahead_distances = distance_matrix[ahead_indices, moving_loop_entry[0]]
        avg_of_distances = (behind_distances + ahead_distances) / 2
        min_index = argmin(avg_of_distances)
        if min_index == loop_length - 3:
            n_loop.appendleft(moving_loop_entry)
        else:
            n_loop.rotate(-(min_index + 1))
            n_loop.appendleft(moving_loop_entry)
            n_loop.rotate(min_index + 1)
        if n_loop[0][0] == 0:
            break
    return [(loop[tpl[0]][0],) + tpl[1:] for tpl in n_loop]


def remove_duplicates(items: list) -> list:
    """Removes duplicate items from a return API call"""
    seen = set()
    final_items = []
    for item in items:
        if item["track"]["id"] not in seen:
            seen.add(item["track"]["id"])
            final_items.append(item)
    return tuple(final_items)


def loop_sort(entries: tuple, func: callable) -> list:
    """Sorts a list of entries by the function func"""
    loop = deque()
    loop.append(entries[0])
    entries = deque(entries[1:])
    length = len(entries)
    for _ in range(1, length + 1):
        item_one: deque = loop[-1]
        item_two: deque = entries
        j = find_minimum(item_one, func, item_two)
        loop.append(item_two[j])
        item_two.rotate(-j)
        item_two.popleft()
    return list(loop)


def ccv_sort(playlist_id: str) -> list:
    """Sorts a playlist using CCVs"""
    items = get_playlist_items(playlist_id)
    items = remove_duplicates(items)
    entries = make_ccv_collection(items, ccv)
    loop = loop_sort(entries, ccv_distance)
    loop = resort_loop(loop, ccv_distance, len(loop))
    return [loop[i][0] for i in range(0, len(loop))]


def make_ccv_collection(playlist_items: tuple, data: callable) -> list:
    """Makes a collection of CCVs from a playlist"""
    tuple_collection = []
    result_queue = Queue()

    def process_item(to_process: dict) -> None:
        """Processes an item in the playlist"""
        track = to_process["track"]
        track_id = track["id"]
        url = track["album"]["images"][-1]["url"]
        result_queue.put((track_id, data(url)))
        print(f"Processed {track['name']}")

    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in playlist_items:
            executor.submit(process_item, item)

    while not result_queue.empty():
        tuple_collection.append(result_queue.get())

    return tuple_collection


def sort_playlist(playlist_id: str) -> None:
    """Sorts a playlist by the given algorithm"""
    sorted_track_ids = ccv_sort(playlist_id)
    reorder_playlist(playlist_id, sorted_track_ids)
    ccv_distance.cache_clear()
    app.progress_bar.set(100, 100)


class App(CTk):
    """The main application class"""

    def __init__(self):
        green = "#1DB954"
        black = "#191414"
        super().__init__()
        self.title("Spotify Playlist Sorter")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)
        self.title_label = CTkLabel(self, text="Spotify Playlist Sorter")
        self.title_label.pack(padx=10, pady=10)
        self.center_frame = CTkFrame(self)
        self.center_frame.pack(padx=10, pady=10)
        self.album_art = CTkLabel(self.center_frame, text="")
        self.album_art.configure(width=250, height=250)
        self.album_art.pack()
        self.dropdown = CTkComboBox(
            self,
            values=[playList["name"] for playList in playlists["items"]],
            command=self.dropdown_changed,
        )
        self.dropdown.configure(state="readonly")
        self.dropdown.pack(padx=10, pady=10)
        self.button = CTkButton(
            self,
            text="Sort",
            command=self.sort_playlist,
            fg_color=green,
            text_color=black,
        )
        self.button.pack(padx=10, pady=10)
        self.progress_bar = CTkProgressBar(
            self, mode="indeterminate", progress_color=green, bg_color=black
        )
        self.progress_bar.set(0, 100)
        self.progress_bar.pack(padx=10, pady=10)
        self.dropdown_changed(None)

    def dropdown_changed(self, _):
        """Called when the dropdown is changed"""
        album_art_thread = Thread(target=self.get_album_art)
        album_art_thread.start()

    def sort_playlist(self):
        """Sorts a playlist"""
        playlist_id = get_playlist_id(self.dropdown.get())
        sorting_thread = Thread(target=sort_playlist, args=(playlist_id,))
        sorting_thread.start()
        self.button.configure(state="disabled")
        self.progress_bar.start()
        while sorting_thread.is_alive():
            self.update()
        self.button.configure(state="normal")
        self.progress_bar.stop()
        self.progress_bar.destroy()
        self.progress_bar = CTkProgressBar(self, mode="indeterminate")
        self.progress_bar.set(0, 100)
        self.progress_bar.pack(padx=10, pady=10)

    def get_album_art(self):
        """Gets the album art for the selected playlist"""
        playlist_id = get_playlist_id(self.dropdown.get())
        art = get_playlist_art(playlist_id)
        self.album_art.destroy()
        self.album_art = CTkLabel(self.center_frame, text="")
        self.album_art.configure(image=art)
        self.album_art.image = art
        self.album_art.pack(fill="both", expand="yes")


if __name__ == "__main__":
    app = App()
    app.mainloop()
