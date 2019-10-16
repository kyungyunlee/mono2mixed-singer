import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import metrics
import argparse
# print (K.tensorflow_backend._get_available_gpus())

import model
import dataloader 
sys.path.append('../')
import msd_config as config

os.environ["CUDA_VISIBLE_DEVICES"] = "5" # 4 for ss, 5 for mix


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
args = parser.parse_args()
print("model name", args.model_name)



def train():
    # load data
    train_list, valid_list, _ = np.load(os.path.join(config.data_dir, 'generator_dcnn_train_data_1000_d.npy'))
    print ('train, test', len(train_list), len(valid_list))

    mymodel =  model.basic_cnn(config.input_frame_len, config.num_singers)

    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, decay=1e-6)
    mymodel.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', metrics.top_k_categorical_accuracy])

    print (mymodel.summary())

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

    train_generator = dataloader.Datagenerator(train_list, config.total_mean, config.total_std, config.num_singers, config.batch_size, config.input_frame_len)
    valid_generator = dataloader.Datagenerator(valid_list, config.total_mean, config.total_std, config.num_singers, config.batch_size, config.input_frame_len)
    
    mymodel.fit_generator(train_generator,
                          shuffle=False,
                          steps_per_epoch=steps_per_epoch,
                          max_queue_size=10,
                          workers=config.num_parallel,
                          use_multiprocessing=False,
                          epochs=config.num_epochs,
                          verbose=1,
                          callbacks=callbacks,
                          validation_data=valid_generator,
                          validation_steps=validation_steps)


    print("finished training")


if __name__ == '__main__':
    train()

