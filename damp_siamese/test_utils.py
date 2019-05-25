import os
import sys
import numpy as np
from random import shuffle

from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances 
from sklearn.metrics import confusion_matrix, pairwise_distances
from operator import itemgetter 
from collections import OrderedDict
import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import argparse 

import config
sys.path.append('../')
import utils


def load_test_data():
    # load training tracks for building artist model 
    artist_list = np.load('../data/unseen_artist_300_2.npy')
    train_list, _ = utils.load_data_segment('../data/unseen_model_artist_track_300_2.pkl', artist_list)
    test_list, _ = utils.load_data_segment('../data/unseen_eval_artist_track_300_2.pkl', artist_list)
    print (len(train_list), len(test_list))
    
    # reformat train. test to increase test size
    all_list = train_list + test_list 
    print (len(all_list))
    sorted_all_list = sorted(all_list, key=lambda x: (x[0], x[1]))
    
    train_list = [] 
    test_list = []
    for i in range(0, len(sorted_all_list), 200):
        train_list.extend(sorted_all_list[i: i+120])
        test_list.extend(sorted_all_list[i+120:i+200])

    print (len(train_list), len(test_list))
    return train_list, test_list


def load_mix_feature(feat_path, start_frame):
    feat_path_tmp = os.path.join(config.mix_mel_path, feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
    feat = np.load(feat_path_tmp)
    feat = feat[:, :config.input_frame_len]
    feat = feat.T
    feat -= config.mix_total_mean
    feat /= config.mix_total_std
    feat = np.expand_dims(feat, 0)
    return feat 


def load_mono_feature(feat_path, start_frame):
    feat = np.load(os.path.join(config.vocal_mel_path, feat_path))
    feat = feat[:, start_frame:start_frame + config.input_frame_len]
    feat = feat.T
    feat -= config.vocal_total_mean
    feat /= config.vocal_total_std
    feat = np.expand_dims(feat, 0)
    return feat



def test_unseen(model_path, pretrained, scenario):
    global mel_path, feat_mean, feat_std

    mymodel = load_model(model_path)
    print(mymodel.summary())

    if scenario in ['mono2mono', 'mix2mix']:
        if pretrained:
            layer_output = mymodel.get_layer('model_2').get_output_at(0)
            layer_input = mymodel.get_layer('model_2').get_input_at(0)

        else:
            layer_output = mymodel.get_layer('global_average_pooling1d_1').get_output_at(0)
            layer_input = mymodel.layers[0].input
        
        get_last_layer_outputs = K.function([layer_input, K.learning_phase()], [layer_output])

    elif scenario == 'mono2mix':
        mono_layer_output = mymodel.get_layer('').get_output_at(0)
        mix_layer_output = mymodel.get_layer('').get_output_at(0)
        mono_get_last_layer_outputs = K.function([],[])
        mix_get_last_layer_outputs = K.function([],[])


    
    ###################### retrieval #########################################
   
    ####  mem efficient #### 
    artist_to_tracks_model = dict.fromkeys(range(0, 300), None)
    for i in range(len(train_list)):
        artist_id, feat_path, start_frame = train_list[i]

        if artist_to_tracks_model[artist_id] == None:
            artist_to_tracks_model[artist_id] = {}
        
        if scenario in ['mix2mix', 'mono2mix']:
            feat_path_tmp = os.path.join(mel_path, feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
            feat = np.load(os.path.join(mel_path, feat_path_tmp))
            feat = feat[:, :config.input_frame_len]
            feat = feat.T
            feat -= config.mix_total_mean
            feat /= config.mix_total_std

        elif scenario == 'mono2mono' : 
            feat_path_tmp = feat_path 
            feat = np.load(os.path.join(mel_path, feat_path_tmp))
            feat = feat[:, start_frame:start_frame + config.input_frame_len]
            feat = feat.T
            feat -= config.vocal_total_mean
            feat /= config.vocal_total_std

        
        feat = np.expand_dims(feat, 0)
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
    artist_track_embs = [] # for knn  
    artist_track_answer = [] 
    for k, track_dict in artist_to_tracks_model.items():
        artist_all_feat =[]
        count_tracks = 0  # for testing effect of number of tracks per artist 
        for tid, v in track_dict.items():
            # if count_tracks > 4:
            #    break
            artist_all_feat.extend(v)
            # track_mean = np.mean(np.array(v), axis=0) # for training knn 
            # artist_track_embs.append(track_mean)
            artist_track_embs.extend(v)
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
        

        if scenario in ['mix2mix', 'mono2mix']:
            feat_path_tmp = os.path.join(mel_path, feat_path.replace('.npy', '_' + str(start_frame) + '.npy'))
            feat = np.load(os.path.join(mel_path, feat_path_tmp))
            feat = feat[:, :config.input_frame_len]
            feat = feat.T
            feat -= config.mix_total_mean
            feat /= config.mix_total_std

        elif scenario == 'mono2mono': 
            feat_path_tmp = feat_path
            feat = np.load(os.path.join(mel_path, feat_path_tmp))
            feat = feat[:, start_frame:start_frame + config.input_frame_len]
            feat = feat.T
            feat -= config.vocal_total_mean
            feat /= config.vocal_total_std

        
        feat = np.expand_dims(feat, 0)
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
    artist_embs = np.array(artist_embs)
    artist_id_answer = np.array(artist_id_answer)
    track_answer = np.array(track_answer)
    print (track_embs.shape, artist_embs.shape)
    sim = cosine_similarity(track_embs, artist_embs)

    pred_score = np.argmax(np.array(sim), axis=1) # 600 query x 1 closest artist 
    # pred_score = np.argmin(np.array(sim), axis=1) # 600 query x 1 closest artist 

    # convert arg to artist id 
    pred_score_id = artist_id_answer[pred_score] 
    print ('pred score, pred score id, size should be the same', pred_score.shape, pred_score_id.shape)
    print (pred_score[0], pred_score_id[0])

    sim_correct = sum(pred_score == track_answer) # correct:1, incorrect:0
    sim_acc = sim_correct / len(track_embs)
    print ('Similarity Correct: %d/%d, Acc:%.4f'%(sim_correct , len(track_embs), sim_acc))

    # top-k score 
    k=5
    pred_scores = np.argsort(-np.array(sim), axis=1)
    # pred_scores = np.argsort(np.array(sim), axis=1)
    print (pred_scores.shape) # 2500 x k
    sim_correct = 0 
    for i in range(len(pred_scores)):
        # print (pred_scores[i][:k], track_answer[i])
        if track_answer[i] in pred_scores[i][:k]:
            sim_correct += 1
    print ('top k : %d/%d, acc:%.4f'%(sim_correct, len(track_embs), sim_correct/len(track_embs)))




if __name__ == '__main__':
    test_unseen(args.model_path, args.pretrained, args.scenario)

