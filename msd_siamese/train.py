import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model, Model
import argparse
# print (K.tensorflow_backend._get_available_gpus())

import model
import dataloader 
sys.path.append('../')
import msd_config as config


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # 4 for ss, 5 for mix


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--pretrained_model', type=str, default=None, help='path to the pretrained model')
args = parser.parse_args()
print("model name", args.model_name)
print("pretrained model path", args.pretrained_model)


def train():
    
    x_train, y_train, train_artist_tracks_segments = dataloader.load_siamese_data(os.path.join(config.data_dir, 'msd_train_data_'+str(config.num_singers) + '_d.csv'), config.num_singers)
    x_valid, y_valid, valid_artist_tracks_segments = dataloader.load_siamese_data(os.path.join(config.data_dir, 'msd_valid_data_' + str(config.num_singers) + '_d.csv'), config.num_singers)


    
    if args.pretrained_model : 
        pretrained_model = load_model(args.pretrained_model)
        mymodel_tmp = Model(pretrained_model.layers[0].input, pretrained_model.layers[-2].output)
        mymodel_tmp.set_weights(pretrained_model.get_weights())
        mymodel = model.finetuning_siamese_cnn(mymodel_tmp, config.input_frame_len, config.num_neg_singers, config.num_pos_tracks)

    else : 
        mymodel =  model.siamese_cnn(config.input_frame_len, config.num_neg_singers, config.num_pos_tracks)
    
    print (mymodel.summary())

    # compile model 
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # adam = Adam(lr=0.0001, decay=1e-6)
    
    mymodel.compile(optimizer=sgd, loss=model.hinge_loss, metrics=['accuracy'])

    
    print ('train', len(x_train), 'valid', len(x_valid))
    train_steps = int(len(x_train) / config.batch_size)
    valid_steps = int(len(x_valid) / config.batch_size)
    
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
                                  patience=2,
                                  verbose=1,
                                  min_lr=0.000016)
    callbacks = [checkpoint, earlystopping, reduce_lr]
   
    train_generator = dataloader.FrameDataGenerator(x_train, y_train, train_artist_tracks_segments, config.total_mean, config.total_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_singers, config.num_pos_tracks, shuffle=True)

    valid_generator = dataloader.FrameDataGenerator(x_valid, y_valid, valid_artist_tracks_segments,  config.total_mean, config.total_std, config.num_singers, config.batch_size, config.input_frame_len, config.num_neg_singers, config.num_pos_tracks, shuffle=False)

    del x_train 
    del y_train
    del x_valid
    del y_valid 
    del train_artist_tracks_segments 
    del valid_artist_tracks_segments 

    mymodel.fit_generator(generator=train_generator,
                      steps_per_epoch=train_steps,
                      max_queue_size=10,
                      shuffle=False,
                      workers=config.num_parallel,
                      use_multiprocessing=False,
                      epochs=config.num_epochs,
                      verbose=1,
                      callbacks=callbacks,
                      validation_data=valid_generator,
                      validation_steps=valid_steps)

    print("finished training")


if __name__ == '__main__':
    train()

