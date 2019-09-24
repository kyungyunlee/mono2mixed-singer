# Code for training 3 scenarios : mono2mono, mix2mix and mono2mix 
# 
#
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import argparse 
import model
# import dataloader
from data_pipeline import get_dataset

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



def get_train_dataset():
    dataset, n_data = get_dataset(
            os.path.join(config.data_dir, 'damp_train_train.csv'), 
            model_type=args.model_type, 
            input_frame_len=config.input_frame_len, 
            n_mels=config.n_mels, 
            batch_size=config.batch_size,
            shuffle=True,
            infinite_generator=True,
            num_parallel_calls=32,
            # cache_dir='/mnt/nfs/analysis/interns/klee/tf_cache/%s_train'%args.model_type)
            cache_dir='/mnt/nfs/analysis/interns/klee/tf_cache/trash')
    
    return dataset, n_data 


def get_valid_dataset():
    dataset, n_data = get_dataset(
            os.path.join(config.data_dir, 'damp_train_valid.csv'), 
            model_type=args.model_type, 
            input_frame_len=config.input_frame_len, 
            n_mels=config.n_mels, 
            batch_size=config.batch_size,
            shuffle=False,
            infinite_generator=True,
            num_parallel_calls=32,
            cache_dir='/mnt/nfs/analysis/interns/klee/tf_cache/%s_valid'%args.model_type)
    
    return dataset, n_data 



def train():
    # load data 
    train_data, n_train = get_train_dataset()
    valid_data, n_valid = get_valid_dataset()
    print (n_train,n_valid)

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
   

    train_steps = int(n_train / config.batch_size) * 5
    valid_steps = int(n_valid / config.batch_size) * 5

    print (train_steps, valid_steps)
    mymodel.fit(
            train_data,
            steps_per_epoch=train_steps,
            shuffle=False,
            epochs=config.num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_data,
            validation_steps=valid_steps)

 



if __name__ == '__main__' : 
    train()
