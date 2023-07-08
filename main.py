"""The main file for the Spotify Playlist Sorter"""
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from customtkinter import (
    CTk,
    CTkFrame,
    CTkLabel,
    CTkButton,
    CTkComboBox,
    CTkProgressBar,
)
from image_processing import ccv, ccv_distance
from playlist_utils import (
    get_playlist_items,
    get_playlist_art,
    reorder_playlist,
    remove_duplicates,
    playlists,
    get_playlist_id,
)
from loop_sorting import loop_sort, resort_loop


def ccv_sort(playlist_id: str) -> list[str]:
    """Sorts a playlist using CCVs"""
    items = get_playlist_items(playlist_id)
    items = remove_duplicates(items)
    entries = make_ccv_collection(items, ccv)
    loop = loop_sort(entries, ccv_distance)
    loop = resort_loop(loop, ccv_distance, len(loop))
    return [loop[i][0] for i in range(0, len(loop))]


def make_ccv_collection(playlist_items: tuple, calculate_ccv: callable) -> list[str, str]:
    """Makes a collection of CCVs from a playlist"""
    tuple_collection: list[str, str] = []
    result_queue: Queue[tuple[str, str]] = Queue()
    url_list = []

    def process_item(to_process: dict[str, str]) -> None:
        """Processes an item in the playlist"""
        track: dict[str, str] = to_process["track"]
        track_id: str = track["id"]
        url: str = track["album"]["images"][-1]["url"]
        url_list.append(url)
        result_queue.put((track_id, calculate_ccv(url)))

    with ThreadPoolExecutor(max_workers=8) as executor:
        for item in playlist_items:
            executor.submit(process_item, item)

    while not result_queue.empty():
        tuple_collection.append(result_queue.get())

    return tuple_collection


def sort_playlist(playlist_id: str) -> None:
    """Sorts a playlist by the given algorithm"""
    sorted_track_ids: list[str] = ccv_sort(playlist_id)
    reorder_playlist(playlist_id, sorted_track_ids)
    ccv_distance.cache_clear()
    app.progress_bar.set(100, 100)


class App(CTk):
    """The main application class"""

    def __init__(self) -> None:
        super().__init__()
        green: str = "#1DB954"
        black: str = "#191414"
        self.title("Spotify Playlist Sorter")
        self.geometry(f"{1100}x{580}")
        self.resizable(False, False)
        self.title_label: CTkLabel = CTkLabel(self, text="Spotify Playlist Sorter")
        self.title_label.pack(padx=10, pady=10)
        self.center_frame: CTkFrame = CTkFrame(self)
        self.center_frame.pack(padx=10, pady=10)
        self.album_art: CTkLabel = CTkLabel(self.center_frame, text="")
        self.album_art.configure(width=250, height=250)
        self.album_art.pack()
        self.dropdown: CTkComboBox = CTkComboBox(
            self,
            values=[playList["name"] for playList in playlists["items"]],
            command=self.dropdown_changed,
        )
        self.dropdown.configure(state="readonly")
        self.dropdown.pack(padx=10, pady=10)
        self.button: CTkButton = CTkButton(
            self,
            text="Sort",
            command=self.sort_playlist,
            fg_color=green,
            text_color=black,
        )
        self.button.pack(padx=10, pady=10)
        self.progress_bar: CTkProgressBar = CTkProgressBar(
            self, mode="indeterminate", progress_color=green, bg_color=black
        )
        self.progress_bar.set(0, 100)
        self.progress_bar.pack(padx=10, pady=10)
        self.dropdown_changed(None)

    def dropdown_changed(self, _: any) -> None:
        """Called when the dropdown is changed"""
        album_art_thread = Thread(target=self.get_album_art)
        album_art_thread.start()

    def sort_playlist(self) -> None:
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

    def get_album_art(self) -> None:
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
