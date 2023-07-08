# Spotify Album Sorter

![GitHub](https://img.shields.io/github/license/benny-nottonson/spotifySort?color=green)
![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)

This project, coded by Benny Nottonson using Python, implements LRU Caching and CCV Vectors for sorting images.

## Acknowledgements

- [miniSOM](https://github.com/JustGlowing/minisom/blob/master/minisom.py)
- [HSY Concept](https://www.alanzucconi.com/2015/09/30/colour-sorting/)
- [ML for Visual Sorting](https://towardsdatascience.com/machine-learning-to-visually-sort-7349d3660e1)
- [CCV Sorting](https://github.com/thjsimmons/SortByColor)

## Features

- Allows selection of playlist
- Works for public and private playlists
- Runtime generally <= 5s
- Scalable for large playlists

## Usage/Examples

```python
# Inside ./playlist_utils
# This can be created at https://developer.spotify.com/dashboard/applications
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="########################",
                                               client_secret="#########################",
                                               redirect_uri="http://example.com",
                                               scope="user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private user-library-read"))
```

## Getting Started

Follow the steps below to get started with the Spotify Album Sorter:

1. Clone the repository:

   ```shell
   git clone https://github.com/benny-nottonson/spotifySort.git
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

3. Run the program:

   ```shell
   python main.py
   ```

## Contributing

Contributions are welcome! If you find any bugs or want to enhance the project, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.