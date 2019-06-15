import os
import sys
import numpy as np
from random import shuffle

import dataloader
sys.path.append('../')
import damp_config as config 


def load_test_data():
    # load training tracks for building artist model 
    artist_list = np.load('../data/unseen_artist_300_2.npy')
    train_list, _ = dataloader.load_data_segment('../data/unseen_model_artist_track_300_2.pkl', artist_list)
    test_list, _ = dataloader.load_data_segment('../data/unseen_eval_artist_track_300_2.pkl', artist_list)
    print (len(train_list), len(test_list))
    
    # reformat train. test to increase test size
    all_list = train_list + test_list 
    print (len(all_list))
    sorted_all_list = sorted(all_list, key=lambda x: (x[0], x[1]))
    
    train_list = [] 
    test_list = []
    for i in range(0, len(sorted_all_list), 200):
        train_list.extend(sorted_all_list[i: i+120])
        test_list.extend(sorted_all_list[i+120:i+200])

    print (len(train_list), len(test_list))
    return train_list, test_list


def load_mix_feature(feat_path, start_frame):
    feat_path_tmp = os.path.join(config.mix_mel_dir, feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
    feat = np.load(feat_path_tmp)
    feat = feat[:, :config.input_frame_len]
    feat = feat.T
    feat -= config.mix_total_mean
    feat /= config.mix_total_std
    feat = np.expand_dims(feat, 0)
    return feat 


def load_mono_feature(feat_path, start_frame):
    feat = np.load(os.path.join(config.vocal_mel_dir, feat_path))
    feat = feat[:, start_frame:start_frame + config.input_frame_len]
    feat = feat.T
    feat -= config.vocal_total_mean
    feat /= config.vocal_total_std
    feat = np.expand_dims(feat, 0)
    return feat


