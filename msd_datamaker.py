import os
import sys
import pickle
import numpy as np
import csv 
import random

singer_data_dir = 'data/msd_data'

artist_id_to_name = pickle.load(open(os.path.join(singer_data_dir, 'artist_id_to_name.pkl'), 'rb'))


train_artists = [ artist_id.strip('\n') for artist_id in open(os.path.join(singer_data_dir, 'train_artists.txt'), 'r').readlines()]
random.shuffle(train_artists)
print (len(train_artists))


artist_to_train_songs = pickle.load(open(os.path.join(singer_data_dir, 'msd_artist_train_SVD.pkl'), 'rb'))
artist_to_valid_songs = pickle.load(open(os.path.join(singer_data_dir, 'msd_artist_valid_SVD.pkl'), 'rb'))
artist_to_test_songs = pickle.load(open(os.path.join(singer_data_dir, 'msd_artist_test_SVD.pkl'), 'rb'))



num_train_tracks = 15
num_valid_tracks = 18


# select which artist to train and eval with (unseen)
num_test_artist = 500 # num of unseen artists for all evaluation 
num_train_artist = 1000 # num of training artists
version = '_a' # indicate data version 

test_artists = train_artists[:num_test_artist] # change number 
print (len(test_artists))
print (len(train_artists))

train_artists = train_artists[num_test_artist:num_test_artist+num_train_artist] 

test_artist_to_int = {}
test_artist_to_name = {} 
for a in range(len(test_artists)):
    # print (artist_id_to_name[test_artists[a]]) # convert artist id to name 
    test_artist_to_int[test_artists[a]] = a
    test_artist_to_name[test_artists[a]] = artist_id_to_name[test_artists[a]]


train_artist_to_int = {}
train_artist_to_name = {} 
for a in range(len(train_artists):
    train_artist_to_int[train_artists[a]] = a
    train_artist_to_name[train_artists[a]] = artist_id_to_name[train_artists[a]]



# csv files 
data_path = 'data'
fieldnames = ['artist_index', 'artist_id', 'artist_name', 'track_id', 'vocal_segments']

train_f = open(data_path + 'msd_train_data_' + str(num_train_singers) + version + '.csv', mode='w')
valid_f = open(data_path + 'msd_valid_data_' + str(num_train_singers) + version + '.csv', mode='w')
test_f = open(data_path + 'msd_test_data_' + str(num_train_singers) + version + '.csv', mode='w')

unseen_train_f = open(data_path + 'msd_unseen_model_data_'  + str(num_test_artist) + version + '.csv', mode='w')
unseen_test_f = open(data_path + 'msd_unseen_eval_data_'  + str(num_test_artist) + version + '.csv', mode='w')

train_writer = csv.DictWriter(train_f, fieldnames=fieldnames)
train_writer.writeheader()
valid_writer = csv.DictWriter(valid_f, fieldnames=fieldnames)
valid_writer.writeheader()
# test_writer_1000 = csv.DictWriter(test_f_1000, fieldnames=fieldnames)
# test_writer_1000.writeheader()

unseen_train_writer = csv.DictWriter(unseen_train_f, fieldnames=fieldnames)
unseen_train_writer.writeheader()
unseen_test_writer = csv.DictWriter(unseen_test_f, fieldnames=fieldnames)
unseen_test_writer.writeheader()




# artist for training 
for i in range(len(train_artists)):
    curr_artist_id = train_artists[i]
    artist_train_songs = list(artist_to_train_songs[curr_artist_id].keys())
    artist_valid_songs = list(artist_to_valid_songs[curr_artist_id].keys())
    artist_test_songs = list(artist_to_test_songs[curr_artist_id].keys())
    all_artist_songs = []
    all_artist_songs.extend(artist_train_songs)
    all_artist_songs.extend(artist_valid_songs)
    all_artist_songs.extend(artist_test_songs)

    random.shuffle(all_artist_songs)

    # num_train_tracks = random.randint(5, 18)
    # num_valid_tracks = num_train_tracks + 2 
    artist_train_songs = all_artist_songs[:num_train_tracks]
    artist_valid_songs = all_artist_songs[num_train_tracks : num_valid_tracks]
    # artist_test_songs = all_artist_songs[num_valid_tracks:]

    for j in range(len(artist_train_songs)):
        curr_track = artist_train_songs[j]
        segments = artist_to_songs[curr_artist_id][curr_track]
        middle10 = segments[len(segments)//2 -5 : len(segments)//2 +5]
        if (len(middle10) < 10):
            print ("warning, less than 10 segments")
        print ('artist_id:', curr_artist_id, 'track_id:', curr_track, 'middle10:', middle10)
        train_writer.writerow({'artist_index': train_artist_to_int[curr_artist_id], 'artist_id': curr_artist_id, 'artist_name': train_artist_to_name[curr_artist_id], 'track_id': curr_track, 'vocal_segments': middle10})

    
    for j in range(len(artist_valid_songs)):
        curr_track = artist_valid_songs[j]
        segments = artist_to_songs[curr_artist_id][curr_track]
        middle10 = segments[len(segments)//2 -5 : len(segments)//2 +5]
        if (len(middle10) < 10):
            print ("warning, less than 10 segments")
        print ('artist_id:', curr_artist_id, 'track_id:', curr_track, 'middle10:', middle10)
        valid_writer.writerow({'artist_index': train_artist_to_int[curr_artist_id], 'artist_id': curr_artist_id, 'artist_name': train_artist_to_name[curr_artist_id], 'track_id': curr_track, 'vocal_segments': middle10})


    '''
    for j in range(len(artist_test_songs)):
        curr_track = artist_test_songs[j]
        segments = artist_to_songs[curr_artist_id][curr_track]
        middle10 = segments[len(segments)//2 -5 : len(segments)//2 +5]
        if (len(middle10) < 10):
            print ("warning, less than 10 segments")
        print ('artist_id:', curr_artist_id, 'track_id:', curr_track, 'middle10:', middle10)
        test_writer_1000.writerow({'artist_index': train_artist_to_int[curr_artist_id], 'artist_id': curr_artist_id, 'artist_name': train_artist_to_name[curr_artist_id], 'track_id': curr_track, 'vocal_segments': middle10})

    '''

# artist as unseen data
for i in range(len(test_artists)):
    curr_artist_id = test_artists[i]
    artist_train_songs = list(artist_to_train_songs[curr_artist_id].keys())
    artist_valid_songs = list(artist_to_valid_songs[curr_artist_id].keys())
    artist_test_songs = list(artist_to_test_songs[curr_artist_id].keys())
    all_artist_songs = []
    all_artist_songs.extend(artist_train_songs)
    all_artist_songs.extend(artist_valid_songs)
    all_artist_songs.extend(artist_test_songs)

    random.shuffle(all_artist_songs)
    artist_train_songs = all_artist_songs[:15]
    artist_test_songs = all_artist_songs[15:]

    for j in range(len(artist_train_songs)):
        curr_track = artist_train_songs[j]
        segments = artist_to_songs[curr_artist_id][curr_track]
        middle10 = segments[len(segments)//2 -5 : len(segments)//2 +5]
        if (len(middle10) < 10):
            print ("warning, less than 10 segments")
        print ('artist_id:', curr_artist_id, 'track_id:', curr_track, 'middle10:', middle10)
        unseen_train_writer.writerow({'artist_index': test_artist_to_int[curr_artist_id], 'artist_id': curr_artist_id, 'artist_name': test_artist_to_name[curr_artist_id], 'track_id': curr_track, 'vocal_segments': middle10})


    for j in range(len(artist_test_songs)):
        curr_track = artist_test_songs[j]
        segments = artist_to_songs[curr_artist_id][curr_track]
        middle10 = segments[len(segments)//2 -5 : len(segments)//2 +5]
        if (len(middle10) < 10):
            print ("warning, less than 10 segments")
        print ('artist_id:', curr_artist_id, 'track_id:', curr_track, 'middle10:', middle10)
        unseen_test_writer.writerow({'artist_index': test_artist_to_int[curr_artist_id], 'artist_id': curr_artist_id, 'artist_name': test_artist_to_name[curr_artist_id], 'track_id': curr_track, 'vocal_segments': middle10})




