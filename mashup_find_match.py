''' 
Code to find matching background segment and save the pair data into csv 
'''

import os
import sys
import librosa
import numpy as np
from pathlib import Path
from multiprocessing import Pool 
import random
import scipy.linalg
import scipy.stats
import csv 
import pickle 
import argparse

from utils import load_data_segment


vocal_path = 'data/audio/'
bg_path = 'data/musdb_accompaniment_combined/'

# tempo, beat feature path 
vocal_feat_path = 'data/damp_beat/'
bg_feat_path = 'data/musdb_beat/'

# chromagram feature path 
vocal_chroma_path = 'data/damp_chroma/'
bg_chroma_path = 'data/musdb_chroma/'
 
bg_wav_files = [song for song in Path(bg_path).glob('*.wav')]
bg_tempo_files = [ song for song in Path(bg_feat_path).glob('*.npy')]


major = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#','B']
minor = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#','B']

cannot_match_counter = 0 
done = 0 



def load_tracks_to_mix():
   # load train data 
    train_artists = np.load('data/artist_1000.npy')
    train_list, _ = load_data_segment('data/train_artist_track_1000.pkl', train_artists)
    valid_list, _ = load_data_segment('data/valid_artist_track_1000.pkl', train_artists)

    training_tracks_to_mix = []
    for i in range(len(train_list)):
        _, feat_path, start_frame = train_list[i]
        training_tracks_to_mix.append((feat_path, start_frame))

    for i in range(len(valid_list)):
        _, feat_path, start_frame = valid_list[i]
        training_tracks_to_mix.append((feat_path, start_frame))
    # pickle.save('training_tracks_to_mix.npy', training_tracks_to_mix)


    # load unseen test data  
    unseen_artists = np.load('data/unseen_artist_300_2.npy')
    unseen_model_list, _ = load_data_segment('data/unseen_model_artist_track_300_2.pkl', unseen_artists)
    unseen_eval_list, _ = load_data_segment('data/unseen_eval_artist_track_300_2.pkl', unseen_artists)
    
    testing_tracks_to_mix = []
    for i in range(len(unseen_model_list)):
        _, feat_path, start_frame = unseen_model_list[i]
        testing_tracks_to_mix.append((feat_path, start_frame))

    for i in range(len(unseen_eval_list)):
        _, feat_path, start_frame = unseen_eval_list[i]
        testing_tracks_to_mix.append((feat_path, start_frame))


    training_tracks_to_mix = list(training_tracks_to_mix)
    testing_tracks_to_mix = list(testing_tracks_to_mix)
    print (len(training_tracks_to_mix), len(testing_tracks_to_mix))

    return training_tracks_to_mix, testing_tracks_to_mix 




def ks_key(X): 
    # code from  'https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd'

    X = scipy.stats.zscore(X)
        
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)
                            
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)
                                        
    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)
                                                        
    return major.T.dot(X), minor.T.dot(X)


def find_bg_track(tempo, key, ismajor, check_key=True) : 
    ''' Function to find a background track with given tempo, key, etc
    Return :
        return nothing if there is no match found
    '''

    global bg_wav_files 
    global bg_feat_path, bg_chroma_path 

    random.shuffle(bg_wav_files)

    for s in range(len(bg_wav_files)) : 
        try : 
            bg = bg_wav_files[s]
            bg_tempo_beat = np.load(bg_feat_path + bg.stem + '.npy')
            bg_tempo = round(bg_tempo_beat[-1])
            bg_beat = librosa.frames_to_time(bg_tempo_beat[:-1], sr=22050, hop_length=512)
            bg_chromagram = np.load(bg_chroma_path + bg.stem + '.npy')
         
        except:
            continue
        
        # check for same tempo track
        if tempo == bg_tempo :
            # if same tempo track is found, get the start and end beat
            for i in range(len(bg_beat)):
                if i+1 >= len(bg_beat):
                    break

                start_beat = bg_beat[i]
                next_beat_index = i 
                end_beat = bg_beat[next_beat_index]
                while end_beat - start_beat < 6.0: # if length of segment is less than 6 seconds
                    next_beat_index +=1 
                    if next_beat_index >= len(bg_beat):
                        break
                    end_beat = bg_beat[next_beat_index]

                start_frame = librosa.time_to_frames(start_beat, sr=22050, hop_length=512)
                end_frame = librosa.time_to_frames(end_beat, sr=22050, hop_length=512)
                
                # detect key and find the matching key 
                if check_key : 
                    bg_chroma_seg = np.sum(bg_chromagram[:, start_frame: end_frame], axis=1)
                    maj_, min_ = ks_key(bg_chroma_seg)
                    if np.max(maj_) > np.max(min_):
                        # major key 
                        bg_key = np.argmax(maj_)
                        bg_key_ = major[bg_key]
                        bg_ismajor = True
                    else : 
                        # minor key 
                        bg_key = np.argmax(min_)
                        bg_key_ = minor[bg_key]
                        bg_ismajor = False
                    
                    if key == bg_key and ismajor == bg_ismajor:
                        print ("same key found", key, bg_key, "ismajor:", ismajor)
                        return bg, bg_tempo, bg_beat, start_beat, end_beat  
                else : 
                    print ("not checking key")
                    return bg, bg_tempo, bg_beat, start_beat, end_beat
    
    
    return None, None, None, None, None 

     

def perform_match(song, start_frame) : 
    '''
    Find the matching tempo and key for given 3 second vocal damp data.
    Write the result to csv file.
    Args: 
        song : 
        start_frame :
    Return:
        None


    '''
    global vocal_feat_path, bg_feat_path ,vocal_chroma_path, bg_chroma_path 
    global major, minor 
    global cannot_match_counter, done 

    # print (song)
    isfound = False
    
    start_sec = librosa.frames_to_time(start_frame, sr=22050, hop_length=512)
    tempo_beat = np.load(vocal_feat_path + song)
    tempo = round(tempo_beat[-1])
    beat = librosa.frames_to_time(tempo_beat[:-1], sr=22050, hop_length=512)

    # find closest beat from the svd start frame 
    svd_start_time = None 
    for b in range(1, len(tempo_beat[:-1])):
        prev_b = tempo_beat[:-1][b-1]
        curr_b = tempo_beat[:-1][b]
        if curr_b > start_frame: 
            # print (start_frame, curr_b, prev_b)
            svd_start_time = librosa.frames_to_time(prev_b, sr=22050, hop_length=512)
            break

    if svd_start_time == None : 
        svd_start_time = librosa.frames_to_time(start_frame, sr=22050, hop_length=512) 
        

    # find key of the 3 sec vocal segment 
    chromagram = np.sum(np.load(vocal_chroma_path + song)[:, start_frame : start_frame + 129], axis=1)
    maj_, min_ = ks_key(chromagram)
    if np.max(maj_) > np.max(min_):
        # major key 
        key = np.argmax(maj_)
        key_ = major[key]
        ismajor = True
    else : 
        # minor key 
        key = np.argmax(min_)
        key_ = minor[key]
        ismajor = False
    # print (key_, ismajor)
    

    # print (tempo, tempo_beat.shape, beat.shape)
    # findo a bg track with same tempo 
    new_tempo = tempo 
    key_change = key
    ismajor_change = ismajor
    bg, bg_tempo, bg_beat, start_time, end_time = find_bg_track(tempo, key, ismajor)
    if bg == None : 
        new_tempo = tempo + 1 
        bg, bg_tempo, bg_beat, start_time, end_time = find_bg_track(new_tempo, key, ismajor)
    if bg == None : 
        new_tempo = tempo -1 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key, ismajor) 

    if bg == None : 
        new_tempo = tempo +2 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key, ismajor)
    if bg == None : 
        new_tempo = tempo -2
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key, ismajor)

    # change to minor key or major 
    if ismajor : 
        key_change = (key + 9)%12
        ismajor_change = False
    else :
        key_change = (key + 3) %12
        ismajor_change = True

    if bg == None : 
        new_tempo = tempo  
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change)

    if bg == None : 
        new_tempo = tempo +1 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change)
    if bg == None : 
        new_tempo = tempo -1 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change)
    if bg == None : 
        new_tempo = tempo +2  
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change)
    if bg == None : 
        new_tempo = tempo -2  
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change)
    
    # ok...dont check key 
    if bg == None : 
        new_tempo = tempo 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change, check_key=False)
    if bg == None : 
        new_tempo = tempo +1 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change, check_key=False)
    if bg == None : 
        new_tempo = tempo -1
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change, check_key=False)
    if bg == None : 
        new_tempo = tempo +2 
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change, check_key=False)
    if bg == None : 
        new_tempo = tempo -2  
        bg, bg_tempo, bg_beat, start_time, end_time  = find_bg_track(new_tempo, key_change, ismajor_change, check_key=False)


    if bg == None :
        print ("oh no") 
        print (song, key_, ismajor, tempo) 
        cannot_match_counter += 1 
    
    # write to csv file 
    csvwriter.writerow({'vocal_track_path':song,
                      'svd_start_sec':start_sec,
                      'svd_beat_start_sec': svd_start_time,
                      'tempo': tempo,
                      'key': key,
                      'maj_min' : ismajor,
                      'bg_track_path':bg,
                      'beat_match_start_sec':start_time,
                      'beat_match_end_sec':end_time,
                      'bg_tempo':bg_tempo,
                      'bg_key':key_change,
                      'bg_maj_min':ismajor_change,
                      })

    done +=1 
    print ("done:", done)



if __name__ == '__main__' : 
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_type', type=str, choices=['train', 'test'],  help='Choose between "train" or "test" set', required=True)
    parser.add_argument('--curr_batch', type=int, help='Indicate the index of the batch you want to process. It should be between 0 and (total_batch - 1)', required=True)
    parser.add_argument('--total_batch', type=int, help='Data is big, so it is better to divide it into batches and process one batch at a time. Indicate the total number of batches you want to subdivide the data', required=True)
    args = parser.parse_args()
    print (args)

    if args.curr_batch >= args.total_batch:
        print ("curr_batch should be between 0 and total_batch -1")
        sys.exit()

    index = args.curr_batch

    if args.set_type == 'train':
        tracks_to_mix, _ = load_tracks_to_mix()
        csvfile = open('data/mashup_pair_data_train_' + str(index) + '.csv', mode='w')
    else :
        _, tracks_to_mix = load_tracks_to_mix()
        csvfile = open('data/mashup_pair_data_unseen_' + str(index) + '.csv', mode='w')

    batch_size = len(tracks_to_mix) // args.total_batch # number of tracks to processnow

    
    # the info about the matched files are saved in a csv file
    fieldnames = ['vocal_track_path', 'svd_start_sec', 'svd_beat_start_sec', 'tempo', 'key', 'maj_min', 'bg_track_path', 'beat_match_start_sec', 'beat_match_end_sec', 'bg_tempo', 'bg_key', 'bg_maj_min']
    csvwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
    csvwriter.writeheader()


    if index +1  == args.total_batch : 
        todo_list = tracks_to_mix[index * batch_size : ] 
    else : 
        todo_list =  tracks_to_mix[index * batch_size : (index + 1) * batch_size]

    for track_and_frame in todo_list: 
        perform_match(track_and_frame[0], int(track_and_frame[1]))
        print ("curr_batch index is %d out of total %d"%(index, args.total_batch))


    print ("number of vocal tracks that could not find a match:", cannot_match_counter)




