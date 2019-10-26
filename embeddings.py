''' Using the trained model to compute singing voice embeddings ''' 
import os
import sys
import numpy as np
import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.losses
from sklearn.metrics.pairwise import cosine_similarity 
from collections import OrderedDict
import argparse 
from multiprocessing import Pool
from pathlib import Path

import damp_config as config 
from feature_extract import parallel_mel
import damp_siamese.model as model 

# args 
parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, choices=['mono', 'mixed'], required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
args = parser.parse_args()
print("model path:", args.model_path)
print("audio path:", args.audio_path)
print("audio file domain:", args.domain)

N_WORKERS=5

def extract_embeddings(model_path, audio_path, domain):
    # load model and correct layers 
    mymodel = load_model(model_path, custom_objects={'hinge_loss' : model.hinge_loss})
    
    if domain == 'mono' : 
        layer_output = mymodel.get_layer('model_1').get_output_at(0)
        layer_input = mymodel.get_layer('model_1').get_input_at(0)
        mean = config.vocal_total_mean
        std = config.vocal_total_std 

    elif domain == 'mixed' : 
        layer_output = mymodel.get_layer('model_2').get_output_at(0)
        layer_input = mymodel.get_layer('model_2').get_input_at(0)
        mean = config.mix_total_mean
        std = config.mix_total_std 

    get_last_layer_outputs = K.function([layer_input, K.learning_phase()],[layer_output])

    # process audio files in the audio_path directory 
    list_of_audiofiles = [f for f in Path(audio_path).glob('*')]
    print ("Number of audiofiles to process :", len(list_of_audiofiles))
    list_of_data = [] 
    ext = '.' + str(list_of_audiofiles[0]).split('.')[-1]
    mel_path = Path(audio_path).stem + '_mel' 
    for audiofile in list_of_audiofiles : 
        list_of_data.append([Path(audiofile).stem + ext, audio_path, mel_path, ext])
    with Pool(N_WORKERS) as p : 
        p.starmap(parallel_mel, list_of_data)
    
    # load data 
    list_of_track_features = [] 
    for audiofile in list_of_audiofiles : 
        feat = np.load(os.path.join(mel_path, Path(audiofile).stem + '.npy'))
        n_segments = feat.shape[1] // config.input_frame_len  
        track_feats = []
        for n in range (n_segments):
            track_feats.append(feat[:, n * config.input_frame_len : (n+1) * config.input_frame_len])
        track_feats = np.array(track_feats)
        list_of_track_features.append(track_feats)
    list_of_track_features = np.array(list_of_track_features)
    
    # compute embeddings 
    print ("Computing embeddings...")
    savedir = Path(audio_path).stem + '_result' 
    if not os.path.exists(savedir) : 
        os.makedirs(savedir, exist_ok=True)
    print ("Saving results to %s"%savedir)

    for i in range(len(list_of_track_features)):
        track_embeddings = [] 
        filename = Path(list_of_audiofiles[i]).stem  
        print("processing", filename)
        for feat in list_of_track_features[i] : 
            feat = feat.T
            feat -= mean
            feat /= std 
            feat = np.expand_dims(feat, 0)
            pred = get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]
            track_embeddings.append(pred)
        track_embeddings = np.mean(track_embeddings, axis=0)
        np.save(os.path.join(savedir, filename +'.npy'), track_embeddings)
    print ("Results saved in ", savedir)    

if __name__ == '__main__':
    extract_embeddings(args.model_path, args.audio_path, args.domain)


