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

import model
import dataloader 
import test_utils

# args 
parser = argparse.ArgumentParser()
parser.add_argument('--scenario', type=str, choices=['mono2mono', 'mix2mix', 'mono2mix'], required=True)
parser.add_argument('--model_type', type=str, choices=['mono', 'mix', 'cross'], required=True)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args()
print ("scenario:", args.scenario)
print("model path:", args.model_path)
print ("pretrained:", args.pretrained)



def test(model_path, model_type, pretrained, scenario):
    # load test data 
    train_list, test_list = test_utils.load_test_data()
    print ("test data loaded")
    
    # load model and correct layers 
    mymodel = load_model(model_path, custom_objects={'hinge_loss' : model.hinge_loss})
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
    
    # print (mymodel.summary())
    
    ###################### inference #########################################
   
    ####  mem efficient #### 
    print ("Computing embeddings...")
    artist_to_tracks_model = dict.fromkeys(range(0, 300), None)
    for i in range(len(train_list)):
        artist_id, feat_path, start_frame = train_list[i]

        if artist_to_tracks_model[artist_id] == None:
            artist_to_tracks_model[artist_id] = {}
        
        if scenario == 'mix2mix':
            feat = test_utils.load_mix_feature(feat_path, start_frame)

        elif scenario == 'mono2mono' : 
            feat = test_utils.load_mono_feature(feat_path, start_frame)

        else:
            feat = test_utils.load_mix_feature(feat_path, start_frame)


        if model_type == 'cross':
            if scenario == 'mix2mix':
                pred = mix_get_last_layer_outputs([feat, 0])[0]
            elif scenario == 'mono2mono':
                pred = mono_get_last_layer_outputs([feat, 0])[0]
            else :
                pred = mix_get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]
        else :
            pred = get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]



        try:
            artist_to_tracks_model[artist_id][feat_path].append(pred)
        except:
            artist_to_tracks_model[artist_id][feat_path] = []
            artist_to_tracks_model[artist_id][feat_path].append(pred)
    artist_to_tracks_model = OrderedDict(sorted(artist_to_tracks_model.items(), key=lambda t: t[0]))
    
    del train_list 
    # artist emb
    artist_embs = []
    artist_id_answer = [] 
    artist_track_answer = [] 
    for k, track_dict in artist_to_tracks_model.items():
        artist_all_feat =[]
        count_tracks = 0  # for testing effect of number of tracks per artist 
        for tid, v in track_dict.items():
            artist_all_feat.extend(v)
            for _ in range(len(v)): 
                artist_track_answer.append(k)
            count_tracks += 1 

        artist_all_feat = np.array(artist_all_feat)
        mean = np.mean(artist_all_feat, axis=0)
        artist_embs.append(mean)
        artist_id_answer.append(k)

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
                pred = mix_get_last_layer_outputs([feat, 0])[0]
            elif scenario == 'mono2mono':
                pred = mono_get_last_layer_outputs([feat, 0])[0]
            else :
                pred = mono_get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]
        else:
            pred = get_last_layer_outputs([feat, 0])[0]
            pred = pred[0]


        try:
            artist_to_tracks_eval[artist_id][feat_path].append(pred)
        except:
            artist_to_tracks_eval[artist_id][feat_path] = []
    

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
    artist_embs = np.array(artist_embs)
    artist_id_answer = np.array(artist_id_answer)
    track_answer = np.array(track_answer)
    print (track_embs.shape, artist_embs.shape)
    sim = cosine_similarity(track_embs, artist_embs)

    pred_score = np.argmax(np.array(sim), axis=1) # 600 query x 1 closest artist 
    # pred_score = np.argmin(np.array(sim), axis=1) # 600 query x 1 closest artist 

    # convert arg to artist id 
    pred_score_id = artist_id_answer[pred_score] 
    # print ('pred score, pred score id, size should be the same', pred_score.shape, pred_score_id.shape)

    sim_correct = sum(pred_score == track_answer) # correct:1, incorrect:0
    sim_acc = sim_correct / len(track_embs)
    print ('Similarity Correct: %d/%d, Acc:%.4f'%(sim_correct , len(track_embs), sim_acc))

    # top-k score 
    k=5
    pred_scores = np.argsort(-np.array(sim), axis=1)
    # pred_scores = np.argsort(np.array(sim), axis=1)
    # print (pred_scores.shape) # 2500 x k
    sim_correct = 0 
    for i in range(len(pred_scores)):
        # print (pred_scores[i][:k], track_answer[i])
        if track_answer[i] in pred_scores[i][:k]:
            sim_correct += 1
    print ('top k : %d/%d, acc:%.4f'%(sim_correct, len(track_embs), sim_correct/len(track_embs)))




if __name__ == '__main__':
    test(args.model_path, args.model_type, args.pretrained, args.scenario)


