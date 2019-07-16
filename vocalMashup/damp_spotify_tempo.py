''' Finds the tempo of the DAMP dataset by searching the spotify API with the song title. You need client id and key for using spotify api'''
import os
import sys
import pandas as pd
import spotipy
import time
import json
import pickle
from spotipy.oauth2 import SpotifyClientCredentials

sys.path.append('../')
import damp_config as config

df = pd.read_csv(config.damp_perf_csv)  
song_list = df['songid'].tolist()


def fix_song_name(song) : 
    if song == 'sitting on the dock of th bay': 
        song = 'sitting on the dock of the bay'
    elif song == 'sittin on dock of th bay':
        song ='sitting on the dock of the bay'
    elif song == 'strongr what doesnt kill':
        song = "stronger what doesn't kill" 
    elif song == 'love you lk a love song': 
        song = 'love you like a love song'
    elif song =='can you feel love tngt':
        song = 'can you feel the love tonight'
    elif song =='this little lght of mine': 
        song = 'this little light of mine'
    elif song == 'hashtag that power': 
        song = '# that power'
    elif song == 'treasure bm': 
        song = 'treasure'
    elif song == 'save th last dance fr me':
        song = 'save the last dance for me'
    elif song == 'hit me with your best sht': 
        song = 'hit me with your best shot' 
    elif song == 'hit me wth your best sht': 
        song = 'hit me with your best shot' 
    elif song == 'just the way you are bm': 
        song = 'just the way you are'
    elif song == 'i wanna dance with smbdy':
        song = 'i wanna dance with somebody'

    return song 




# all the unique songs in the dataset 
song_list = [" ".join(song[1:].split('_')) for song in song_list]
new_song_list = [] 
for song in song_list : 
    new_song_list.append(fix_song_name(song))

song_list = new_song_list
song_list = list(set(song_list))

# map perf_key to song name 
perf_to_song_name = {} 
for i, row in df.iterrows():
    perf_to_song_name[row['perf_key']] = fix_song_name(" ".join(row['songid'][1:].split('_')))


# specify your own client id and key 
client_credentials_manager = SpotifyClientCredentials(client_id='', client_secret='')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
sp.trace=False

# find tempo 
song_name_to_tempo = {} 
for song in song_list : 
    print ("curr song:", song)

    results = sp.search(q=song, limit=3)
    tids = []
    for i, t in enumerate(results['tracks']['items']):
        print(' ', i, t['name'])
        tids.append(t['uri'])

    start = time.time()
    features = sp.audio_features(tids[0])
    tempo = features[0]["tempo"]
    print (tempo)
    song_name_to_tempo[song] = tempo


row_list = [] 
for k,v in perf_to_song_name.items():
    row_list.append({'perf_key': k, 'song_name': v, 'tempo': song_name_to_tempo[v]})

new_df = pd.DataFrame(row_list, columns=['perf_key', 'song_name', 'tempo'])
new_df.to_csv('damp_tempo.csv', index=False)
