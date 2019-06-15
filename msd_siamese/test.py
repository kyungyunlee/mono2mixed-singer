import os
import sys
import numpy as np
from random import shuffle

import tensorflow as tf 
import keras
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from keras.optimizers import SGD, Adam
from keras import metrics
from keras.models import load_model, Model
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances
from operator import itemgetter 
import csv
from collections import OrderedDict
import argparse
# print (K.tensorflow_backend._get_available_gpus())

import model
sys.path.append('../')
import msd_config as config 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,required=True)
parser.add_argument('--pretrained', action='store_true')
args = parser.parse_args()


keras.losses.hinge_loss = model.hinge_loss


def build_singer_model(model, train_list, feat_mean, feat_std, mel_path, num_singers, forBuildingModel):
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    if args.pretrained : 
        layer_output = model.get_layer('model_1').get_output_at(0)
        get_last_layer_outputs = K.function([model.get_layer('model_1').get_input_at(0), K.learning_phase()], [layer_output])
    else : 
        get_last_layer_outputs = K.function([model.layers[0].input, K.learning_phase()], [layer_dict['global_average_pooling1d_1'].output])

    artist_to_track_model = dict.fromkeys(range(0,500), None)
    for i in range(len(train_list)):
        artist_id, feat_path, start_frame = train_list[i]
        feat = np.load(os.path.join(mel_path, feat_path))
        
        if artist_to_track_model[artist_id] == None:
            artist_to_track_model[artist_id] = {}
        tmp_feat = feat[:, start_frame:start_frame + config.input_frame_len]
        tmp_feat = tmp_feat.T
        tmp_feat -= feat_mean
        tmp_feat /= feat_std
        tmp_feat = np.expand_dims(tmp_feat, 0)
        pred = get_last_layer_outputs([tmp_feat, 0])[0]
        pred = pred[0]

        try:
            artist_to_track_model[artist_id][feat_path].append(pred)
        except:
            artist_to_track_model[artist_id][feat_path] = []
            artist_to_track_model[artist_id][feat_path].append(pred)
    
    if forBuildingModel:
        embs = [] 
        artist_track_answer = [] 
        for k,track_dict in artist_to_track_model.items():
            artist_all_feat= []
            count_tracks = 0 
            for tid,v in track_dict.items():

                artist_all_feat.extend(v)
                for _ in range(len(v)):
                    artist_track_answer.append(k)
                count_tracks +=1 

            artist_all_feat = np.array(artist_all_feat)
            mean = np.mean(artist_all_feat, axis=0)
            embs.append(mean)
            
            track_answer = [] 
        
        embs = np.array(embs)

        return embs, artist_track_answer 

    else :
        embs = []
        track_answer = [] 
        for k,track_dict in artist_to_track_model.items():
            for tid,v in track_dict.items():
                v = np.array(v)
                mean = np.mean(v, axis=0)
                embs.append(mean)
                track_answer.append(k)
        embs = np.array(embs)
        track_answer = np.array(track_answer)
        return embs, track_answer 

        






def test_unseen():
    train_list , test_list = np.load(os.path.join(config.data_dir, 'gen_dcnn_unseen_data_500_d.npy'))
    print (len(train_list), len(test_list))

    mymodel = load_model(args.model_path)

    artist_embeddings, artist_track_answer = build_singer_model(mymodel, train_list, config.total_mean, config.total_std, config.mel_dir, 500, forBuildingModel=True)
    track_embeddings, track_answer = build_singer_model(mymodel, test_list, config.total_mean,config.total_std, config.mel_dir, 500, forBuildingModel=False)

    cos_sim = cosine_similarity(track_embeddings, artist_embeddings)
    pred_score = np.argmax(np.array(cos_sim), axis=1)


    sim_correct = sum(pred_score == track_answer) # correct:1, incorrect:0

    sim_acc = sim_correct / len(track_embeddings)
    print ('Similarity Correct: %d/%d, Acc:%.4f'%(sim_correct , len(track_embeddings), sim_acc))
    
    # top-k accuracy 
    k=5
    pred_scores = np.argsort(-np.array(cos_sim), axis=1)
    top_k_correct= 0
    for i in range(len(pred_scores)):
        if track_answer[i] in pred_scores[i][:k]:
            top_k_correct += 1

    print ("top k: %d/%d, acc:%.4f"%(top_k_correct, len(track_embeddings), top_k_correct/len(track_embeddings)))




if __name__ == '__main__':
    test_unseen()

