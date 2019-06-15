''' 
run beat, chroma algorithm and save the result to npy 
'''

import os
import librosa
import numpy as np
from pathlib import Path
from multiprocessing import Pool 

from utils import load_data_segment 


def beat_chroma(song_path, beat_save_dir, chroma_save_dir):
    ''' 
    Calculates beat tracking and chromagram algorithm from librosa.
    Used as a mapping function for multiprocessing 
    Args: 
        song_path : posix path 
        beat_save_dir : path to save beat result
        chroma_save_dir : path to save chroma result
    Return :
        None
    '''
    beat_savepath = os.path.join(beat_save_dir, song_path.stem + '.npy')
    chroma_savepath = os.path.join(chroma_save_dir, song_path.stem + '.npy')
    
    if os.path.exists(beat_savepath):
        print ('beat already computed')
        return 

    if os.path.exists(chroma_savepath):
        print ('chroma already computed')
        return 

    y, _ = librosa.load(str(song_path), sr=22050)
    # beat tracking 
    tempo, beat = librosa.beat.beat_track(y, sr=22050)
    print (beat_savepath, tempo, beat.shape)
    feat = np.append(beat, tempo)
    np.save(beat_savepath, feat)
    
    # chroma 
    chromagram = librosa.feature.chroma_stft(y, sr=22050, n_chroma=12, n_fft=1024)
    print (chroma_savepath, chromagram.shape)
    np.save(chroma_savepath, chromagram)




if __name__ == '__main__' : 
    # gather damp data and musdb background tracks 
    vocal_path = 'data/damp_audio/' # path to damp dataset 
    bg_path = 'data/musdb_accompaniment_combined/' # path to musdb dataset 

    vocal_beat_path = 'data/damp_beat/' # path to save beat detection result 
    bg_beat_path = 'data/musdb_beat/' # path to save beat detection result 

    if not os.path.exists(vocal_beat_path):
        os.makedirs(vocal_beat_path)

    if not os.path.exists(bg_beat_path):
        os.makedirs(bg_beat_path)

    vocal_chroma_path = 'data/damp_chroma/' # path to save chroma result
    bg_chroma_path = 'data/musdb_chroma/' # path to save chroma result 

    if not os.path.exists(vocal_chroma_path):
        os.makedirs(vocal_chroma_path)
    if not os.path.exists(bg_chroma_path):
        os.makedirs(bg_chroma_path)


    vocal_songs = [song for song in Path(vocal_path).glob('*.m4a')]
    bg_songs = [song for song in Path(bg_path).glob('*.wav')]
    print (len(vocal_songs), len(bg_songs)) # 34329, 150 
    
    # load damp training data 
    train_artists = np.load('data/artist_1000.npy')
    train_list, _ = load_data_segment('data/train_artist_track_1000.pkl', train_artists)
    valid_list, _ = load_data_segment('data/valid_artist_track_1000.pkl', train_artists)

    training_tracks_to_mix = set()
    for i in range(len(train_list)):
        _, feat_path, _ = train_list[i]
        training_tracks_to_mix.add(feat_path)

    for i in range(len(valid_list)):
        _, feat_path, _ = valid_list[i]
        training_tracks_to_mix.add(feat_path)

    training_tracks_to_mix = list(training_tracks_to_mix)
    print('length of data used for training:' , len(training_tracks_to_mix))


    # load damp testing data 
    unseen_artists = np.load('data/unseen_artist_300_2.npy')
    unseen_model_list, _ = load_data_segment('data/unseen_model_artist_track_300_2.pkl', unseen_artists) # for modeling the artist 
    unseen_eval_list, _ = load_data_segment('data/unseen_eval_artist_track_300_2.pkl', unseen_artists) # for performing tests 

    testing_tracks_to_mix = set()
    for i in range(len(unseen_model_list)):
        _, feat_path, _ = unseen_model_list[i]
        testing_tracks_to_mix.add(feat_path)

    for i in range(len(unseen_eval_list)):
        _, feat_path, _ = unseen_eval_list[i]
        testing_tracks_to_mix.add(feat_path)

    testing_tracks_to_mix = list(testing_tracks_to_mix)
    print('length of data used for testing:' , len(testing_tracks_to_mix))

    ### finally  run beat detection and save 
    training_vocal_args = [(Path(vocal_path + song.replace('.npy', '.m4a')), vocal_beat_path, vocal_chroma_path) for song in training_tracks_to_mix]
    testing_vocal_args = [(Path(vocal_path + song.replace('.npy', '.m4a')), vocal_beat_path, vocal_chroma_path) for song in testing_tracks_to_mix]
    bg_args = [(song, bg_beat_path, bg_chroma_path) for song in bg_songs]

    with Pool(5) as p:
        p.starmap(beat_chroma, training_vocal_args)
    print ("finished training data")

    with Pool(5) as p:
        p.starmap(beat_chroma, testing_vocal_args)
    print ("finished testing data")

    with Pool(5) as p : 
        p.starmap(beat_chroma, training_bg_args)
    
    
    print ("finished all")

