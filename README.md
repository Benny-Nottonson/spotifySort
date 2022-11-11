# Spotify Album Sorter 

A project coded by Benny Nottonson using Python and the miniSOM implementation by JustGlowing



## Acknowledgements

 - [miniSOM](https://github.com/JustGlowing/minisom/blob/master/minisom.py)
 - [HSY Concept](https://www.alanzucconi.com/2015/09/30/colour-sorting/)
 - [ML for Visual Sorting](https://towardsdatascience.com/machine-learning-to-visually-sort-7349d3660e1)



[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)



## Features

- Allows selection of album
- Preloaded training data
- Works for public and private playlists
- Runtime generally <= 30s
- Scalable for large playlists


## Usage/Examples

```python
# This can be created at https://developer.spotify.com/dashboard/applications
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id="########################",
                                               client_secret="#########################",
                                               redirect_uri="http://example.com",
                                               scope="user-library-modify playlist-modify-public ugc-image-upload playlist-modify-private user-library-read"))
```
