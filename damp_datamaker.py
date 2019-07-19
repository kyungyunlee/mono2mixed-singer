import os 
import sys
import pandas as pd 
import librosa 
import numpy as np
import tensorflow as tf 
from multiprocessing import Pool
from pathlib import Path
import pickle
import random
import csv
random.seed(20190718)

import damp_config as config 


def _vocal_detection(path):
    input_frame_len = config.input_frame_len
    input_hop_len = config.input_frame_len // 2 
    try : 
        y, _ = librosa.load(path, sr=config.sr)
    except : 
        return None, None  
    rmse = librosa.feature.rmse(y, frame_length=config.n_fft, hop_length=config.hop_length, center=True)
    rmse = rmse[0]

    threshold = 0.04
    
    vocal_segments = np.where(rmse>threshold)[0]
    binary = rmse>threshold * 1
    vocal_segments = [] 

    for i in range(0, binary.shape[0], input_hop_len):
        curr_segment = binary[i:i+input_frame_len]
        vocal_ratio = np.sum(curr_segment) / input_frame_len
        if vocal_ratio > 0.7 :
            start_time = librosa.frames_to_time(i, hop_length=config.hop_length, n_fft=config.n_fft)
            vocal_segments.append(round(start_time, 2))
            # vocal_segments.append(start_time)

    return path, vocal_segments 


def perform_vocal_detection():

    df = pd.read_csv(os.path.join(config.data_dir, 'perfs.csv'))

    song_list = df['perf_key'].tolist()
    print (len(song_list))

    args = [os.path.join(config.vocal_audio_dir, song + '.m4a')  for song in song_list]
    # result = _vocal_detection('/mnt/nfs/analysis/interns/klee/DAMP_audio/122055398_34614878.m4a')
    # print (Path(result[0]).stem, result[1])
    # sys.exit()

    with Pool(20) as p:
        vocal_seg_results = p.map(_vocal_detection, args)

    
    results = {}  
    for result in vocal_seg_results: 
        if result[0] is not None : 
            results[Path(result[0]).stem] = result[1]

    pickle.dump(results, open(os.path.join(config.data_dir, 'damp_SVD.pkl'), 'wb'))



def make_csv():
    
    df = pd.read_csv(os.path.join(config.data_dir, 'perfs.csv'))
    track_svd = pickle.load(open(os.path.join(config.data_dir, 'damp_SVD.pkl'), 'rb'))


    svd_list_csv = open(os.path.join(config.data_dir, 'damp_svd_data.csv'), 'w')
    svd_list_writer = csv.DictWriter(svd_list_csv, fieldnames=['perf_key', 'plyrid', 'svds', 'n_svd'])
    svd_list_writer.writeheader()
    '''
    all_data_csv = open(os.path.join(config.data_dir, 'damp_all_data.csv'), 'w')
    all_data_writer = csv.DictWriter(all_data_csv, fieldnames=['perf_key', 'plyrid', 'seg_start','seg_duration'])
    all_data_writer.writeheader()
    '''
    for track_id, svds in track_svd.items():
        # for svd in svds : 
        svd_list_writer.writerow({'perf_key': track_id,
                        'plyrid': track_id.split('_')[0],
                        'svds': svds,
                        'n_svd': len(svds)
                        # 'seg_start': svd, 
                        # 'seg_duration': 3.0,
                        })
        '''
        for svd in svds : 
            all_data_writer.writerow({'perf_key': track_id,
                        'plyrid': track_id.split('_')[0],
                        'seg_start': svd, 
                        'seg_duration': 3.0})
        '''

    return 


def data_split():
    
    # select 1300 singers 

    df = pd.read_csv(os.path.join(config.data_dir, 'damp_svd_data.csv'))
    # select singers with many svd detected songs 
    all_singers = list(set(df['plyrid'].tolist()))
    suff_singers = [] 
    for singer in all_singers : 
        singer_df = df[df['plyrid'] == singer]
        singer_n_svds = np.array(singer_df['n_svd'].tolist())
        singer_n_svds[singer_n_svds <10] = 0 
        singer_n_svds[singer_n_svds>=10] = 1 
        sum_n_svd = np.sum(singer_n_svds)
        if sum_n_svd == 10 : 
            suff_singers.append(singer)


    print (len(all_singers))
    print (len(suff_singers))

    random.shuffle(suff_singers)

    n_train = 1000 
    n_test = 300 

    train_singers = suff_singers[:n_train]
    test_singers = suff_singers[n_train : n_train+ n_test]

    


    df_train = df[df['plyrid'].isin(train_singers)]
    df_test = df[df['plyrid'].isin(test_singers)]


    # split tracks per singer 
    train_tracks = [] 
    valid_tracks = [] 
    for singer in train_singers : 
        singer_df = df_train[df_train['plyrid'] == singer]
        singer_tracks = list(set(singer_df['perf_key']))
        random.shuffle(singer_tracks)
        train_tracks.extend( singer_tracks[:8])
        valid_tracks.extend (singer_tracks[8:])
    
    model_tracks = [] 
    eval_tracks = []
    for singer in test_singers:
        singer_df = df_test[df_test['plyrid'] == singer]
        singer_tracks = list(set(singer_df['perf_key']))
        random.shuffle(singer_tracks)
        model_tracks.extend(singer_tracks[:6])
        eval_tracks.extend(singer_tracks[6:])
    
    
    df_train_train = df_train[df_train['perf_key'].isin(train_tracks)]
    df_train_valid = df_train[df_train['perf_key'].isin(valid_tracks)]

    df_test_model = df_test[df_test['perf_key'].isin(model_tracks)]
    df_test_eval = df_test[df_test['perf_key'].isin(eval_tracks)]


    df_train_train.to_csv(os.path.join(config.data_dir, 'damp_train_train.csv'))
    df_train_valid.to_csv(os.path.join(config.data_dir, 'damp_train_valid.csv'))

    df_test_model.to_csv(os.path.join(config.data_dir, 'damp_test_model.csv'))
    df_test_eval.to_csv(os.path.join(config.data_dir, 'damp_test_eval.csv'))



if __name__ == '__main__': 
    # perform_vocal_detection()
    make_csv()
    data_split()
