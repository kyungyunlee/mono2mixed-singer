# Code for training 3 scenarios : mono2mono, mix2mix and mono2mix 
# 
#
import os
import sys
import numpy as np
import tensorflow as tf
from keras import backend as K
import keras 
import argparse 
import model
import dataloader
sys.path.append('../')
import damp_config as config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_type', type=str, choices=['mono', 'mix', 'cross'], required=True)
parser.add_argument('--pretrained_model', type=str, default=None, help="path to the pretrained model")
args = parser.parse_args()
print("model name", args.model_name)
print("model type", args.model_type)
print("pretrained model path", args.pretrained_model)



def train():
    # load data 
    train_artists = np.load(os.path.join(config.data_dir, 'artist_1000.npy'))
    x_train, y_train, train_artist_tracks_segments = dataloader.load_siamese_data(os.path.join(config.data_dir, 'train_artist_track_1000.pkl'), train_artists, 1000)
    x_valid, y_valid, valid_artist_tracks_segments = dataloader.load_siamese_data(os.path.join(config.data_dir, 'valid_artist_track_1000.pkl'), train_artists, 1000)

    print (x_train[0], y_train[0])
    print (len(x_train), len(y_train))

    # initialize  model 
    if args.model_type in ['mono', 'mix']:
        if args.pretrained_model: 
            pretrained_model = keras.models.load_model(args.pretrained_model)
            mymodel_tmp = keras.models.Model(pretrained_model.inputs, pretrained_model.layers[-2].output)
            mymodel_tmp.set_weights(pretrained_model.get_weights())
            mymodel = model.finetuning_siamese_cnn(mymodel_tmp, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks)
        else : 
            mymodel = model.siamese_cnn(config.input_frame_len, config.num_neg_artist, config.num_pos_tracks)

    elif args.model_type == 'cross':
        if args.pretrained_model : 
            vocal_pretrained = keras.models.load_model(args.pretrained_model)
            mix_pretrained = keras.models.load_model(args.pretrained_model)
            vocal_model_tmp = keras.models.Model(vocal_pretrained.inputs, vocal_pretrained.layers[-2].output)
            vocal_model_tmp.set_weights(vocal_pretrained.get_weights())
            mix_model_tmp = keras.models.Model(mix_pretrained.inputs, mix_pretrained.layers[-2].output)
            mix_model_tmp.set_weights(mix_pretrained.get_weights())

            mymodel = model.finetuning_mono2mix(vocal_model_tmp, mix_model_tmp, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks)

        else: 
            mymodel = model.siamese_cnn_mono2mix(config.input_frame_len, config.num_neg_artist, config.num_pos_tracks)

    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    mymodel.compile(optimizer=sgd, loss=model.hinge_loss, metrics=['accuracy'])
    print(mymodel.summary())

    weight_name = os.path.join('models', args.model_name + '.{epoch:02d}.h5')
    
    if not os.path.exists(os.path.dirname(weight_name)):
        os.makedirs(os.path.dirname(weight_name))

    checkpoint = keras.callbacks.ModelCheckpoint(
                    monitor='val_loss',
                    filepath=weight_name,
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                    mode='auto')
    earlystopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1,
                    mode='auto')
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=2,
                    verbose=1,
                    min_lr=0.000016)
    callbacks = [checkpoint, earlystopping, reduce_lr]
   

    if args.model_type == 'mono':
            mel_dir = config.vocal_mel_dir
            feat_mean = config.vocal_total_mean
            feat_std = config.vocal_total_std

            train_generator = dataloader.FrameDataGenerator(x_train, y_train, 'mono', mel_dir, train_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=True)
            valid_generator = dataloader.FrameDataGenerator(x_valid, y_valid, 'mono', mel_dir, valid_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=False)

    elif args.model_type  == 'mix':
            mel_dir = config.mix_mel_dir
            feat_mean = config.mix_total_mean
            feat_std = config.mix_total_std

            train_generator = dataloader.FrameDataGenerator(x_train, y_train, 'mix', mel_dir, train_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=True)
            valid_generator = dataloader.FrameDataGenerator(x_valid, y_valid, 'mix', mel_dir, valid_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=False)

    elif args.model_type == 'cross':
        mel_dir = [config.vocal_mel_dir, config.mix_mel_dir]
        feat_mean = [config.vocal_total_mean, config.mix_total_mean]
        feat_std = [config.vocal_total_std, config.mix_total_std]

        train_generator = dataloader.FrameDataGenerator_cross(x_train, y_train, mel_dir, train_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=True)
        valid_generator = dataloader.FrameDataGenerator_cross(x_valid, y_valid, mel_dir, valid_artist_tracks_segments, feat_mean, feat_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_artist, config.num_pos_tracks, shuffle=False)

    
    train_steps = int(len(x_train) / config.batch_size)
    valid_steps = int(len(x_valid) / config.batch_size)

    del x_train, y_train, x_valid, y_valid

    mymodel.fit_generator(
            generator=train_generator,
            steps_per_epoch=train_steps,
            max_queue_size=10,
            shuffle=False,
            workers=config.num_parallel,
            use_multiprocessing=True,
            epochs=config.num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_generator,
            validation_steps=valid_steps)

 



if __name__ == '__main__' : 
    train()
