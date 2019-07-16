### Code to create a mashup dataset using DAMP karaoke tracks and MUSDB instrumental tracks.   
Since tempo estimation on vocal-only track is very difficult, I queried Spotify API with the song title of the karaoke track and get the tempo information from there. The information is in `damp_tempo.csv`.     


### How to use 
1. Make sure the directory to MUSDB instrumental tracks directory and to DAMP audio directory under `../damp_config.py` is correctly set. 
2. Run the following
```
python mashability.py 
```
3. The output should be in the path you specified for `mix_audio_dir` in `damp_config.py`

### General code
Checkout [vocalMashup](https://github.com/kyungyunlee/vocalMashup). I will probably add better mashup code for future work. 
