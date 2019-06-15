import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf 
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from keras.layers import Input 
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.models import load_model, Model
import argparse
# print (K.tensorflow_backend._get_available_gpus())

import model
import dataloader 
sys.path.append('../')
import damp_config as config

os.environ["CUDA_VISIBLE_DEVICES"] = "6" 


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_type', type=str, choices=['mix', 'mono'])
args = parser.parse_args()
print("data type", args.data_type)
print("model name", args.model_name)


if args.data_type == 'mix' : 
    feat_mean = config.mix_total_mean
    feat_std = config.mix_total_std
    mel_path = config.mix_mel_dir
else : 
    feat_mean = config.vocal_total_mean
    feat_std = config.vocal_total_std
    mel_path = config.vocal_mel_dir 


train_artist_list = np.load('../data/artist_1000.npy')
train_list, train_artist_names  = dataloader.load_data_segment('../data/train_artist_track_1000.pkl', train_artist_list)
valid_list, _  = dataloader.load_data_segment('../data/valid_artist_track_1000.pkl', train_artist_list)


def train():
    global train_list, valid_list,  feat_mean, feat_std, mel_path  
    
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, decay=1e-6)

    mymodel =  model.basic_cnn(config.input_frame_len, config.num_singers)
    mymodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', metrics.top_k_categorical_accuracy])


    steps_per_epoch = len(train_list) // config.batch_size
    validation_steps = len(valid_list) // config.batch_size
    
    weight_name = 'models/'+ args.model_name + '.{epoch:02d}.h5'

    if not os.path.exists(os.path.dirname(weight_name)):
        os.makedirs(os.path.dirname(weight_name))


    checkpoint = ModelCheckpoint(monitor='val_loss', # val_loss 
                                 filepath=weight_name, 
                                 verbose=1, 
                                 save_best_only=True, 
                                 save_weights_only=False, 
                                 mode='auto')
    earlystopping = EarlyStopping(monitor='val_loss',
                                  patience=10,
                                  verbose=1,
                                  mode='auto')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=3,
                                  verbose=1,
                                  min_lr=1e-6)
    callbacks = [checkpoint, earlystopping, reduce_lr]

    
    train_generator = dataloader.Datagenerator(train_list, args.data_type, mel_path, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len)
    valid_generator = dataloader.Datagenerator(valid_list, args.data_type, mel_path, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len)
    

    
    mymodel.fit_generator(train_generator,
                          shuffle=False,
                          steps_per_epoch=steps_per_epoch,
                          max_queue_size=10,
                          workers=5,
                          use_multiprocessing=False,
                          epochs=config.num_epochs,
                          verbose=1,
                          callbacks=callbacks,
                          # validation_data=(x_valid,y_valid))
                          validation_data=valid_generator,
                          validation_steps=validation_steps)

    print("finished training")


if __name__ == '__main__':
    train()

