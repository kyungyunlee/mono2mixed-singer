from __future__ import print_function
import os
import sys
import numpy as np
import random
import librosa
import csv 
import tensorflow as tf 
import tensorflow.keras as keras
import argparse
import ast 
import pickle
import random
sys.path.append('../')
import msd_config as config



def fast_sample(array): 
    idx = np.arange(len(array))
    np.random.shuffle(idx)
    return array[idx[0]] 



class FrameDataGenerator(keras.utils.Sequence):
    def __init__(self, train_list, y_list, artist_tracks_segments, feat_mean, feat_std, num_singers, batch_size, input_frame_len, num_neg_singers, num_pos_tracks, shuffle):
        '''
        Args : 
            train_list: list of tracks and corresponding list of vocal segments ex.(track_path, vocal_segment)
            y_list : list of artist_id - matching train_list
            artist_tracks_segments : dictionary of artist (key) to tracks and thei corresponding vocal segments 
            feat_mean : mean value of the input melspectrograms of training data  
            feat_std : standard deviation of the input melspectrograms of training data
            num_singers : number of singers training with ex. 1000 singers
            batch_size : 
            input_frame_len : length of the input melspectrogram
            num_neg_singers : number of negative sampling artists ex. 4 singers
            num_pos_tracks : number of tracks from the anchor artist ex. 1 track
            shuffle : whether to shuffle the artist or not

        '''
        self.train_list = train_list 
        self.y_list = y_list
        self.train_artist_tracks_segments = train_artist_tracks_segments 
        self.feat_mean = feat_mean
        self.feat_std = feat_std
        self.num_singers = num_singers
        self.batch_size = batch_size
        self.input_frame_len = input_frame_len
        self.num_neg_singers = num_neg_singers 
        self.num_pos_tracks = num_pos_tracks
        self.shuffle=shuffle
        
        if shuffle:
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
        y_train_batch_siamese = np.zeros((self.batch_size, config.num_neg_singers + self.num_pos_tracks))
        y_train_batch_classification = np.zeros((self.batch_size, self.num_singers ))

        for i, (x_item, y_item) in enumerate(zip(list_of_x, list_of_y)):
            # anchor 
            curr_artist_id = y_item 
            feat_path, start_frame = x_item
            start_frame = int(start_frame)

            feat = np.load(os.path.join(config.mel_path, feat_path))[:,start_frame:start_frame + self.input_frame_len] 
            feat = feat.T

            feat -= self.feat_mean
            feat /= self.feat_std

            x_anchor_batch.append(feat)
            
            y_train_batch_classification[i, curr_artist_id] = 1
            # pos 
            pos_candidates = list(self.train_artist_tracks_segments[curr_artist_id])

            x_pos_tmp = []
            for j in range(self.num_pos_tracks):
                pos_feat_path = fast_sample(pos_candidates)
                pos_start_frames = self.train_artist_tracks_segments[curr_artist_id][pos_feat_path]
                pos_start_frame = int(fast_sample(pos_start_frames))

                pos_feat = np.load(os.path.join(config.mel_path, pos_feat_path))[:, pos_start_frame : pos_start_frame +self.input_frame_len] 
                pos_feat = pos_feat.T

                pos_feat -= self.feat_mean
                pos_feat /= self.feat_std
                x_pos_tmp.append(pos_feat)

            x_pos_batch.append(x_pos_tmp)
            
            # neg
            neg_artist_candidates = list(set(self.train_artist_tracks_segments.keys()) - set([curr_artist_id]))
            neg_artist_ids = [fast_sample(neg_artist_candidates) for _ in range(self.num_neg_singers)]
            x_negs_tmp = []
            for j in range(self.num_neg_singers):
                neg_artist_id = neg_artist_ids[j]
                candidate_tracks = list(self.train_artist_tracks_segments[neg_artist_id])
                neg_feat_path = fast_sample(candidate_tracks)
                start_frame = int(fast_sample(self.train_artist_tracks_segments[neg_artist_id][neg_feat_path]))

                neg_feat = np.load(os.path.join(config.mel_path, neg_feat_path))[:,start_frame:start_frame + self.input_frame_len] 
                neg_feat = neg_feat.T

                neg_feat -= self.feat_mean
                neg_feat /= self.feat_std

                x_negs_tmp.append(neg_feat)

            x_negs_batch.append(x_negs_tmp)


        x_anchor_batch = np.array(x_anchor_batch)
        x_pos_batch = np.array(x_pos_batch)
        x_negs_batch = np.array(x_negs_batch)
        

        x_train_batch = [x_anchor_batch] + [x_pos_batch[:,k,:,:] for k in range(self.num_pos_tracks)] + [x_negs_batch[:,k,:,:] for k in range(self.num_neg_singers)]

        y_train_batch_siamese[:, :self.num_pos_tracks]= 1
        return x_train_batch, y_train_batch_siamese

    def on_epoch_end(self):
        if self.shuffle:
            self.indexes = np.arange(len(self.train_list))
            np.random.shuffle(self.indexes)
            self.train_list = self.train_list[self.indexes]
            self.y_list = self.y_list[self.indexes]





def load_siamese_data (csvfilepath, num_train_artist):
    ''' 
    Args:
    Return: 
    ''' 
        
    artist_tracks_segments = {} # dict of artist to tracks to vocal segments 
    with open(csvfilepath, 'r') as csv_file : 
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            curr_artist = int(row['artist_index'])
            path_to_feat = config.id7d_to_path[config.idmsd_to_id7d[row['track_id']]].replace('.mp3','.npy')
            start_frames = librosa.time_to_frames(ast.literal_eval(row['vocal_segments']), sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft)
            if start_frames[0] < 0:
                start_frames[0] = 0
            
            try : 
                artist_tracks_segments[curr_artist][path_to_feat] = start_frames
            except :
                artist_tracks_segments[curr_artist] = {}
                artist_tracks_segments[curr_artist][path_to_feat] = start_frames



    track_list = [] 
    y_list = [] 

    with open(csvfilepath, 'r') as csv_file : 
        singer_list = np.arange(num_train_artist)
        print ('num_singers:', len(singer_list))
        csv_reader = csv.DictReader(csv_file)

        line_count = 0 
        for row in csv_reader:
            if line_count == 0 :
                line_count +=1
            curr_artist_id = int(row['artist_index'])
            path_to_feat = config.id7d_to_path[config.idmsd_to_id7d[row['track_id']]].replace('.mp3', '.npy')

            start_frames = librosa.time_to_frames(ast.literal_eval(row['vocal_segments']), sr=config.sr, hop_length=config.hop_length, n_fft=config.n_fft)
            
            # train with all vocal segments 
            for i in range(len(start_frames)):
                
                if start_frames[i] < 0:
                    start_frames[i] = 0

                track_list.append((path_to_feat, start_frames[i]))
                
                y_list.append(curr_artist_id)


    track_list = np.array(track_list)
    y_list = np.array(y_list)

    
    return track_list, y_list, artist_tracks_segments 




