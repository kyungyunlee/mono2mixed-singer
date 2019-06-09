from __future__ import print_function
import os
import sys
import numpy as np
import keras
import random
import librosa

import config 


class Datagenerator(keras.utils.Sequence):
    def __init__(self,train_list, data_type, mel_path, feat_mean, feat_std, num_singers, batch_size, input_frame_len, shuffle=True):
        self.train_list = train_list
        self.data_type = data_type
        self.mel_path = mel_path
        self.feat_mean = feat_mean
        self.feat_std = feat_std 

        self.num_singers = num_singers
        self.batch_size= batch_size
        self.input_frame_len =  input_frame_len
        self.shuffle = shuffle

        if shuffle: 
            self.on_epoch_end() 
    
    def __getitem__(self, idx):
        list_of_data = self.train_list[idx*self.batch_size : (idx+1)*self.batch_size]
        x, y = self.__data_generation(list_of_data)
        return x,y

    def __data_generation(self, list_of_data):

        x_train_batch = []
        y_train_batch = np.zeros((self.batch_size, self.num_singers))

        for i, item_iter in enumerate(list_of_data):
            artist_id, feat_path, start_frame = item_iter
            start_frame = int(start_frame)
            
            if self.data_type == 'mix' : 
                feat_path = os.path.join(self.mel_path, feat_path.replace('.npy', '_' + str(start_frame) + '.npy')) 
                feat = np.load(feat_path) [:, : self.input_frame_len]
            else : 
                feat = np.load(os.path.join(self.mel_path, feat_path))[:,start_frame:start_frame + self.input_frame_len] 

            feat = feat.T
            feat -= self.feat_mean
            feat /= self.feat_std

            x_train_batch.append(feat)
            y_train_batch[i, artist_id] = 1 

        x_train_batch = np.array(x_train_batch)

        return x_train_batch, y_train_batch


    def on_epoch_end(self):
        if self.shuffle: 
            random.shuffle(self.train_list)

    def __len__(self):
        return len(self.train_list) // self.batch_size

