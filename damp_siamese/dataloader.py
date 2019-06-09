from __future__ import print_function
import os
import numpy as np
import random
import librosa
import tensorflow as tf 
import keras
import threading
import argparse
import pickle
import random
from pathlib import Path

import config 


def fast_sample(array): 
    idx = np.arange(len(array))
    np.random.shuffle(idx)
    return array[idx[0]] 


class FrameDataGenerator(keras.utils.Sequence):
    def __init__(self, train_list, y_list, scenario,  mel_path, artist_tracks_segments, feat_mean, feat_std, num_singers, batch_size, input_frame_len, num_neg_artist, num_pos_tracks, shuffle):
        '''
        Args : 
            train_list: list of tracks and corresponding list of vocal segments ex.(track_path, vocal_segment)
            y_list : list of artist_id - matching train_list
            mel_path : 
            artist_tracks_segments : dict of artist to tracks to vocal segments 
            feat_mean : 
            feat_std : 
        '''
        self.train_list = train_list 
        self.y_list = y_list
        self.scenario = scenario
        assert scenario in ['mono2mono', 'mix2mix']
        self.mel_path = mel_path
        self.artist_tracks_segments = artist_tracks_segments 
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.num_singers = num_singers
        self.batch_size = batch_size
        self.input_frame_len = input_frame_len
        self.num_neg_artist = num_neg_artist
        self.num_pos_tracks = num_pos_tracks
        self.shuffle=shuffle
        
        self.on_epoch_end()

        

    def __len__(self):
        return len(self.train_list) // self.batch_size
    
    def __getitem__(self, idx):
        # returns (6, 32, 129, 128), (32, 5) 
        list_of_x = self.train_list[idx*self.batch_size : (idx+1) * self.batch_size]
        list_of_y = self.y_list[idx*self.batch_size : (idx+1) * self.batch_size]
        x,y = self.__data_generation(list_of_x, list_of_y)
        return x,y 

    

    def __data_generation(self, list_of_x, list_of_y):
        
        x_anchor_batch = []
        x_pos_batch = []
        x_negs_batch = [] 
        y_train_batch_siamese = np.zeros((self.batch_size, config.num_neg_artist + self.num_pos_tracks))

        for i, (x_item, y_item) in enumerate(zip(list_of_x, list_of_y)):
            # anchor 
            curr_artist_id = y_item 
            feat_path, start_frame = x_item
            start_frame = int(start_frame)
            
            if self.scenario == 'mix2mix' : 
                feat_path_name = os.path.join(self.mel_path, feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
                feat = np.load(feat_path_name)[:, :self.input_frame_len]
            else :
                feat_path_name = os.path.join(self.mel_path, feat_path)
                feat = np.load(feat_path_name)[:, start_frame:start_frame+self.input_frame_len]
            
            feat = feat.T
            feat -= self.feat_mean
            feat /= self.feat_std

            x_anchor_batch.append(feat)

            # pos item 
            pos_candidates = list(self.artist_tracks_segments[curr_artist_id])

            x_pos_tmp = []
            for j in range(self.num_pos_tracks):
                pos_feat_path = fast_sample(pos_candidates)
                pos_start_frames = self.artist_tracks_segments[curr_artist_id][pos_feat_path]
                pos_start_frame = int(fast_sample(pos_start_frames))
                
                if self.scenario == 'mix2mix': 
                    pos_path_name = os.path.join(self.mel_path, pos_feat_path.replace('.npy', '_' + str(pos_start_frame) + '.npy'))
                    pos_feat = np.load(pos_path_name)[:, :self.input_frame_len]
                else :
                    pos_path_name = os.path.join(self.mel_path, pos_feat_path)
                    pos_feat = np.load(pos_path_name)[:, pos_start_frame:pos_start_frame + self.input_frame_len]

                pos_feat = pos_feat.T
                pos_feat -= self.feat_mean
                pos_feat /= self.feat_std
            
                x_pos_tmp.append(pos_feat)
            x_pos_batch.append(x_pos_tmp)

                
            # neg
            neg_artist_candidates = list(set(self.artist_tracks_segments.keys()) - set([curr_artist_id]))
            neg_artist_ids = [fast_sample(neg_artist_candidates) for _ in range(self.num_neg_artist)]
            x_negs_tmp = []
            for j in range(self.num_neg_artist):
                # neg_artist_id = fast_sample(neg_artist_candidates)
                neg_artist_id = neg_artist_ids[j]
                candidate_tracks = list(self.artist_tracks_segments[neg_artist_id])
                neg_feat_path = fast_sample(candidate_tracks)
                start_frame = int(fast_sample(self.artist_tracks_segments[neg_artist_id][neg_feat_path]))
               
                if self.scenario == 'mix2mix': 
                    neg_path_name = os.path.join(self.mel_path, neg_feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
                    neg_feat= np.load(neg_path_name)[:,:self.input_frame_len]
                else :
                    neg_path_name = os.path.join(self.mel_path, neg_feat_path)
                    neg_feat = np.load(neg_path_name)[:, start_frame:start_frame+self.input_frame_len]
                neg_feat = neg_feat.T
                neg_feat -= self.feat_mean
                neg_feat /= self.feat_std

                x_negs_tmp.append(neg_feat)

            x_negs_batch.append(x_negs_tmp)


        x_anchor_batch = np.array(x_anchor_batch)
        x_pos_batch = np.array(x_pos_batch)
        x_negs_batch = np.array(x_negs_batch)
        
    
        x_train_batch = [x_anchor_batch] + [x_pos_batch[:,k,:,:] for k in range(self.num_pos_tracks)] + [x_negs_batch[:,k,:,:] for k in range(self.num_neg_artist)]

        y_train_batch_siamese[:, :self.num_pos_tracks]= 1

        return x_train_batch, y_train_batch_siamese, 


    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.arange(len(self.train_list))
            np.random.shuffle(self.indexes)
            self.train_list = self.train_list[self.indexes]
            self.y_list = self.y_list[self.indexes]




class FrameDataGenerator_mono2mix(keras.utils.Sequence):
    def __init__(self, train_list, y_list, mel_path, artist_tracks_segments, feat_mean, feat_std, num_singers, batch_size, input_frame_len, num_neg_artist, num_pos_tracks, shuffle):
        '''
        Args : 
            train_list: list of tracks and corresponding list of vocal segments ex.(track_path, vocal_segment)
            y_list : list of artist_id - matching train_list
            mel_path :
            artist_tracks_segments : dict of artist to tracks to vocal segments 
        '''
        self.train_list = train_list 
        self.y_list = y_list
        self.vocal_mel_path = mel_path[0]
        self.mix_mel_path = mel_path[1]
        self.artist_tracks_segments = artist_tracks_segments 
        self.vocal_feat_mean = feat_mean[0]
        self.vocal_feat_std = feat_std[0]
        self.mix_feat_mean = feat_mean[1]
        self.mix_feat_std = feat_std[1] 
        self.num_singers = num_singers
        self.batch_size = batch_size
        self.input_frame_len = input_frame_len
        self.num_neg_artist = num_neg_artist
        self.num_pos_tracks = num_pos_tracks
        self.shuffle=shuffle
        self.idx_ = 0 
        
        if shuffle:
            self.on_epoch_end()

        


    def __len__(self):
        return len(self.train_list) // self.batch_size
    
    def __getitem__(self, idx):
        self.idx_ = idx 
        # returns (6, 32, 129, 128), (32, 5) 
        list_of_x = self.train_list[idx*self.batch_size : (idx+1) * self.batch_size]
        list_of_y = self.y_list[idx*self.batch_size : (idx+1) * self.batch_size]
        x,y = self.__data_generation(list_of_x, list_of_y)
        return x,y 

    

    def __data_generation(self, list_of_x, list_of_y):
        
        x_anchor_batch = []
        x_pos_batch = []
        x_negs_batch = [] 
        y_train_batch_siamese = np.zeros((self.batch_size, config.num_neg_artist + self.num_pos_tracks)) # cosine 

        for i, (x_item, y_item) in enumerate(zip(list_of_x, list_of_y)):
            # anchor 
            curr_artist_id = y_item 
            feat_path, start_frame = x_item
            start_frame = int(start_frame)
            # start_frame = int(fast_sample(start_frames))
            
            feat = np.load(os.path.join(self.vocal_mel_path, feat_path))[:, start_frame : start_frame + config.input_frame_len]
            feat = feat.T

            feat -= self.vocal_feat_mean
            feat /= self.vocal_feat_std
            
            x_anchor_batch.append(feat)
            
            # pos 
            pos_candidates = list(self.artist_tracks_segments[curr_artist_id])

            x_pos_tmp = []
            for j in range(self.num_pos_tracks):
                pos_feat_path = fast_sample(pos_candidates)
                pos_start_frames = self.artist_tracks_segments[curr_artist_id][pos_feat_path]
                pos_start_frame = int(fast_sample(pos_start_frames))

                pos_path_name = os.path.join(self.mix_mel_path, pos_feat_path.replace('.npy', '_' + str(pos_start_frame) + '.npy'))
                pos_feat = np.load(pos_path_name)[:, : self.input_frame_len]
                pos_feat = pos_feat.T

                pos_feat -= self.mix_feat_mean
                pos_feat /= self.mix_feat_std

                if pos_feat.shape != (129, 128):
                    print("oops", pos_path_name, pos_start_frame)
                    return self.__getitem__(self.idx_ -1) 

                x_pos_tmp.append(pos_feat)
            

            x_pos_batch.append(x_pos_tmp)

                
            # neg
            neg_artist_candidates = list(set(self.artist_tracks_segments.keys()) - set([curr_artist_id]))
            neg_artist_ids = [fast_sample(neg_artist_candidates) for _ in range(self.num_neg_artist)]
            x_negs_tmp = []
            for j in range(self.num_neg_artist):
                # neg_artist_id = fast_sample(neg_artist_candidates)
                neg_artist_id = neg_artist_ids[j]
                candidate_tracks = list(self.artist_tracks_segments[neg_artist_id])
                neg_feat_path = fast_sample(candidate_tracks)
                start_frame = int(fast_sample(self.artist_tracks_segments[neg_artist_id][neg_feat_path]))
                    
                neg_path_name = os.path.join(self.mix_mel_path, neg_feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
                neg_feat = np.load(neg_path_name)[:, :self.input_frame_len]
                # neg_feat = np.load(os.path.join(self.mix_mel_path, neg_feat_path))[:,start_frame:start_frame + self.input_frame_len] 
                neg_feat = neg_feat.T

                neg_feat -= self.mix_feat_mean
                neg_feat /= self.mix_feat_std

                x_negs_tmp.append(neg_feat)

                if neg_feat.shape != (129, 128):
                    print("oops", neg_path_name, start_frame)
                    return self.__getitem__(self.idx_ - 1) 

            x_negs_batch.append(x_negs_tmp)

        
        
        x_anchor_batch = np.array(x_anchor_batch)
        x_pos_batch = np.array(x_pos_batch)
        x_negs_batch = np.array(x_negs_batch)
        

        x_train_batch = [x_anchor_batch] + [x_pos_batch[:,k,:,:] for k in range(self.num_pos_tracks)] + [x_negs_batch[:,k,:,:] for k in range(self.num_neg_artist)]

        y_train_batch_siamese[:, :self.num_pos_tracks]= 1 # cosine 
        return x_train_batch, y_train_batch_siamese



    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.arange(len(self.train_list))
            np.random.shuffle(self.indexes)
            self.train_list = self.train_list[self.indexes]
            self.y_list = self.y_list[self.indexes]






if __name__ == '__main__':
    from load_siamese_data_frame import load_siamese_data

    x_train, y_train, train_artist_tracks_segments = load_siamese_data(config.data_dir + 'msd_train_data_1000_b.csv')
    # x_valid, y_valid, valid_artist_tracks_segments = load_siamese_data(config.data_dir + 'msd_valid_data_1000_b.csv')


    feat_mean = np.load('mix_train_mean.npy')
    feat_std = np.load('mix_train_std.npy')
    # valx, valy = load_valid_identification(x_valid, feat_mean, feat_std, 500, config.mel_path)
    gen = FrameDataGenerator(x_train, y_train, train_artist_tracks_segments, feat_mean, feat_std, 1000, config.batch_size, config.input_frame_len, len(x_train) // config.batch_size,4,1)
    print (gen[0])
    '''
    artist_id, feat_path, vocal_idx = x_train[0]
    feat = np.load(os.path.join(config.mel_path, feat_path))
    print (feat.shape)
    print (artist_id, vocal_idx)
    '''



