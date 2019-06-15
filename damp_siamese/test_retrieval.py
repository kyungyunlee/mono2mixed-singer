''' Track retrieval based on singer similarity 
Given track as query, retrieve tracks sung by the same singer. 

scenario1: mono2mono
* query = mono
* database = mono 

scenario2: mix2mix
* query = mix 
* database = mix 

scenario3: mono2mix
* query = mono
* database = mix 

Model type 
    * mono : model trained only with monophonic vocal track
    * mix : model trained only with mixed track
    * cross : model trained with both mono and mixed tracks 
'''
import os
import sys
import numpy as np
from random import shuffle
import tensorflow as tf 
from keras import backend as K
from keras import metrics
from keras.models import load_model, Model
import keras.losses
from sklearn.metrics.pairwise import cosine_similarity 
from collections import OrderedDict
import argparse

import model
import dataloader 
import test_utils


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
keras.losses.hinge_loss = model.hinge_loss

# args 
parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, choices=['mono2mono', 'mix2mix', 'mono2mix'], help='test scenario', required=True)
parser.add_argument('--model_type', type=str, choices=['mono', 'mix', 'cross'], required=True)
parser.add_argument('--pretrained', action='store_true', help='using model trained with pretrained model?')
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()
print ("scenario:", args.scenario)
print("model path:", args.model_path)
print ("pretrained:", args.pretrained)



def test(model_path, model_type, pretrained, scenario):
    # load test data
    train_list, test_list = test_utils.load_test_data()

    mymodel = load_model(model_path)

    if model_type in ['mono', 'mix']:
        if pretrained:
            layer_output = mymodel.get_layer('model_1').get_output_at(0)
            layer_input = mymodel.get_layer('model_1').get_input_at(0)

        else:
            layer_output = mymodel.get_layer('global_average_pooling1d_1').get_output_at(0)
            layer_input = mymodel.layers[0].input

        get_last_layer_outputs = K.function([layer_input, K.learning_phase()], [layer_output])

    elif model_type == 'cross':
        if pretrained :
            mono_layer_output = mymodel.get_layer('model_1').get_output_at(0)
            mono_layer_input = mymodel.get_layer('model_1').get_input_at(0)
            mix_layer_output = mymodel.get_layer('model_2').get_output_at(0)
            mix_layer_input = mymodel.get_layer('model_2').get_input_at(0)
        else:
            mono_layer_output = mymodel.get_layer('global_average_pooling1d_1').get_output_at(0)
            mono_layer_input = mymodel.layers[0].input
            mix_layer_output = mymodel.get_layer('global_average_pooling1d_2').get_output_at(0)
            mix_layer_input = mymodel.layers[1].input

        mono_get_last_layer_outputs = K.function([mono_layer_input, K.learning_phase()],[mono_layer_output])
        mix_get_last_layer_outputs = K.function([mix_layer_input, K.learning_phase()],[mix_layer_output])

    print(mymodel.summary())

    ###################### inference #########################################
    #### mem efficient #### 

    #### track modeling ####
    artist_to_tracks_model = dict.fromkeys(range(0, 300), None)
    for i in range(len(train_list)):
        artist_id, feat_path, start_frame = train_list[i]

        if artist_to_tracks_model[artist_id] == None:
            artist_to_tracks_model[artist_id] = {}
        
        # which data type to load : mix or monophonic?
        if scenario == 'mix2mix':
            feat = test_utils.load_mix_feature(feat_path, start_frame)
        elif scenario == 'mono2mono':
            feat = test_utils.load_mono_feature(feat_path, start_frame)
        else:
            feat = test_utils.load_mix_feature(feat_path, start_frame)
        
        # which portion of cross model to use for inference 
        if model_type == 'cross' : 
            if scenario == 'mix2mix':
                pred = mix_get_last_layer_outputs([feat,0])[0]
            elif scenario == 'mono2mono':
                pred = mono_get_last_layer_outputs([feat,0])[0]
            else : 
                pred = mix_get_last_layer_outputs([feat,0])[0]
            pred = pred[0]
        else:
            pred = get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]

        try:
            artist_to_tracks_model[artist_id][feat_path].append(pred)
        except:
            artist_to_tracks_model[artist_id][feat_path] = []
            artist_to_tracks_model[artist_id][feat_path].append(pred)
    artist_to_tracks_model = OrderedDict(sorted(artist_to_tracks_model.items(), key=lambda t: t[0]))
    
    del train_list 

    # For creating database - take mean at track level   
    database_embs = []
    # artist_track_embs = [] # for knn  
    database_answer = [] 
    for k, track_dict in artist_to_tracks_model.items():
        artist_all_feat =[]
        count_tracks = 0  # for testing effect of number of tracks per artist 

        for tid, v in track_dict.items():
            v = np.array(v)
            mean = np.mean(v, axis=0)
            database_embs.append(mean)
            database_answer.append(k)
            # artist_track_embs.extend(v)
            count_tracks += 1 

    del artist_to_tracks_model


    artist_to_tracks_eval = dict.fromkeys(range(0, 300), None)
    for i in range(len(test_list)):
        artist_id, feat_path, start_frame = test_list[i]

        if artist_to_tracks_eval[artist_id] == None:
            artist_to_tracks_eval[artist_id] = {}


        if scenario  == 'mix2mix':
            feat = test_utils.load_mix_feature(feat_path, start_frame)

        elif scenario == 'mono2mono':
            feat = test_utils.load_mono_feature(feat_path, start_frame)

        elif scenario == 'mono2mix':
            feat = test_utils.load_mono_feature(feat_path, start_frame)


        if model_type == 'cross':
            if scenario == 'mix2mix':
                pred = mix_get_last_layer_outputs([feat,0])[0]
            elif scenario == 'mono2mono':
                pred = mono_get_last_layer_outputs([feat,0])[0]
            else : 
                pred = mono_get_last_layer_outputs([feat,0])[0]
            pred = pred[0]
        else:
            pred = get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]
        
        
        try:
            artist_to_tracks_eval[artist_id][feat_path].append(pred)
        except:
            artist_to_tracks_eval[artist_id][feat_path] = []
            artist_to_tracks_eval[artist_id][feat_path].append(pred)
    

    del test_list 
    # artist emb
    track_embs = []
    track_answer = [] 
    for k, track_dict in artist_to_tracks_eval.items():
        for tid, v in track_dict.items():
            v = np.array(v)
            mean = np.mean(v, axis=0)
            track_embs.append(mean)
            track_answer.append(k)

    del artist_to_tracks_eval


    track_embs = np.array(track_embs)
    database_embs = np.array(database_embs)
    database_answer = np.array(database_answer)
    track_answer = np.array(track_answer)
    print (track_embs.shape, database_embs.shape)
    sim = cosine_similarity(track_embs, database_embs)
    # sim = euclidean_distances(track_embs, artist_embs)


    k=5
    pred_scores = np.argsort(-np.array(sim), axis=1)
    # print ('pred scores: 600 x 2400?', pred_scores.shape) # 600 x 2400

    pred_scores_k = pred_scores[:, :k]
    # print ('pred scores at k : this is index with highest sim', pred_scores_k[0])
    pred_id = database_answer[pred_scores_k] # y_score 
    # print ('pred id: this is actual artist id with highest sim',  pred_id[0])
    # print ('track answer : this is the actual artist id', track_answer[0])

    # to binary 
    track_answer_ext = track_answer[np.newaxis].T
    for _ in range(k-1) : 
        track_answer_ext = np.hstack((track_answer_ext, track_answer[np.newaxis].T))
    # print ('track answer ext: 600 x k?',  track_answer_ext.shape)

    
    pred_binary = (pred_id == track_answer_ext) * 1
    # print ('pred to binary : 1 if the predicted track artist is same as the query track artist', pred_binary[0])

    
    ######  MAP ###### 
    all_ap = [] 
    n_relevant = 8 
    for i in range(pred_binary.shape[0]):
        # ap = average_precision_score(pred_true[i], pred_binary[i])
        total  = [] 
        for j in range(pred_binary.shape[1]) : 
            total.append( np.sum(pred_binary[i][:j+1])/(j+1) )
        
        total = sum(total) / n_relevant 
        # print (total)
        all_ap.append(total)
    print (np.mean(np.array(all_ap)))
    

    ##### PR@k, Recall@k
    all_precision = [] 
    all_recall = [] 
    
    for i in range(pred_binary.shape[0]):
        all_precision.append(np.sum(pred_binary[i]) / k)
        all_recall.append(np.sum(pred_binary[i]) / n_relevant)

    mean_precision = np.mean(all_precision)
    mean_recall = np.mean(all_recall)
    print ('mean precision @%d, %.4f'%(k, mean_precision))
    print ('mean recall @%d, %.4f'%(k, mean_recall))



if __name__ == '__main__':
    test(args.model_path, args.model_type, args.pretrained, args.scenario)

