### Video downloading
Example usage:

`python .\download_video.py "https://www.youtube.com/watch?v=abcdefg" --output .\clips\video.mkv`

`python .\download_video.py "https://www.twitch.tv/videos/1234567890" --output .\clips\video.mkv`

### Load detection
Example usage:

`python .\extract_loads.py .\clips\video.mkv --start 00:00:07 --end 01:35:14 --bars bars.json`

You can also pass in a `--output-dir` arg and the script will output png files of the beginning and end frames of a detected load. Can use to ensure the script is detecting loads correctly.

There is a more verbose `--debug-dir` arg that will output a png for each frame, overlaid with debug info about the loading bar locations it's looking for and which pixels are being detected correctly.
