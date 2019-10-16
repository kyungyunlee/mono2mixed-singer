import sys
import librosa 
import numpy as np
import pickle
from tensorflow.keras import backend as K

sys.path.append('../')
import damp_config as config 


def load_data_segment(picklefile, artist_list):
    train_data = []
    artist_names = []

    f = pickle.load(open(picklefile, 'rb'))
    artist_to_id = {}
    for u in range(len(artist_list)):
        artist_to_id[artist_list[u]] = u

    for artist_id, tracks in f.items():
        for track_id, svd in tracks.items():
            center_segs = svd[len(svd)//2 - 10 : len(svd)//2 + 10]
            # center_segs = svd[len(svd)//2 - 5 : len(svd)//2 + 5]
            start_frames = librosa.time_to_frames(center_segs, sr=22050, hop_length=512, n_fft=1024)
            for i in range(len(start_frames)):
                start_frame = start_frames[i]
                if start_frame < 0:
                    start_frame = 0
                # train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame))
                ### augmentation
                train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame))
                # train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame, 1 ))
                artist_names.append(artist_id)
                artist_names.append(artist_id)

    return train_data, artist_names


def load_siamese_data (picklefile, artist_list, num_train_artist):

    artist_track_segs = {}
    f = pickle.load(open(picklefile, 'rb'))
    artist_to_id = {}
    for u in range(len(artist_list)):
        artist_to_id[artist_list[u]] = u

    train_data = []
    y_data = []
    for artist_id, tracks in f.items() :
        artist_track_segs[artist_to_id[artist_id]] = {}

        for track_id, svd in tracks.items() :
            center_segs = svd[len(svd) // 2 - 10 : len(svd) //2 + 10]
            start_frames = librosa.time_to_frames(center_segs, sr=22050, hop_length=512, n_fft=1024)
            artist_track_segs[artist_to_id[artist_id]][track_id + '.npy'] = start_frames        
            for i in range(len(start_frames)):
                start_frame = start_frames [i]
                if start_frame < 0 :
                    start_frame = 0

                train_data.append((track_id + '.npy', start_frame))
                y_data.append(artist_to_id[artist_id])

    train_data = np.array(train_data)
    y_data = np.array(y_data)

    return train_data, y_data, artist_track_segs




def compute_gain(spec): 
    rms = librosa.feature.rmse(S=np.abs(spec), frame_length=1024)[0]
    rms_filter_ind = np.where(rms >= 0.04) # remove silent frmaes 
    rms_filter = rms[rms_filter_ind]
    mean_rms = np.mean(rms_filter)
    return mean_rms 


def hinge_loss(y_true, y_pred):
    y_pos = y_pred[:, :config.num_pos_tracks]
    y_neg = y_pred[:, config.num_pos_tracks:]

    total_loss = 0.0
    for i in range(config.num_pos_tracks):
        loss = K.sum(K.maximum(0., config.margin - y_pos[:, i:i+1] + y_neg))
        total_loss += loss
    return total_loss


