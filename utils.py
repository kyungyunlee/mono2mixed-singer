import librosa 
import pickle


def load_data_segment(picklefile, artist_list):
    train_data = []
    artist_names = []

    f = pickle.load(open(picklefile, 'rb'))
    artist_to_id = {}
    for u in range(len(artist_list)):
        artist_to_id[artist_list[u]] = u

    for artist_id, tracks in f.items():
        for track_id, svd in tracks.items():
            center_segs = svd[len(svd)//2 - 10 : len(svd)//2 + 10]
            # center_segs = svd[len(svd)//2 - 5 : len(svd)//2 + 5]
            start_frames = librosa.time_to_frames(center_segs, sr=22050, hop_length=512, n_fft=1024)
            for i in range(len(start_frames)):
                start_frame = start_frames[i]
                if start_frame < 0:
                    start_frame = 0
                # train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame))
                ### augmentation
                train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame))
                # train_data.append((artist_to_id[artist_id], track_id + '.npy', start_frame, 1 ))
                artist_names.append(artist_id)
                artist_names.append(artist_id)

    return train_data, artist_names

