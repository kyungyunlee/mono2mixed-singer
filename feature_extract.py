import os
import sys
import librosa 
import numpy as np
from multiprocessing import Pool

N_WORKERS = 5 

def parallel_mel(track, save_dir, audio_dir):

    audiofile = os.path.join(audio_dir, track)
    savefile = os.path.join(save_dir, track.replace(ext, '.npy'))
    
    if not os.path.exists(os.path.dirname(savefile)):
        os.makedirs(os.path.dirname(savefile))

    if os.path.exists(savefile):
        print (savefile, ":already exists")
        return 

    y, _ = librosa.load(audiofile, sr=config.sr)
    S = librosa.core.stft(y, n_fft=config.n_fft, hop_length=config.hop)
    X = np.abs(S)
    mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.n_fft, n_mels=config.n_mels)
    mel_S = np.dot(mel_basis, X)
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.astype(np.float32)
    print (mel_S.shape, savefile)
    np.save(savefile,  mel_S)

 

def process_msd_singer():
    import msd_config 
    global N_WORKERS 

    data_dir = msd_config.data_dir
    save_dir = msd_config.mel_dir 
    audio_dir = msd_config.audio_dir
    ext = '.mp3' # .wav for ss .mp3 for mix 

    #### training data 
    x_train, x_valid, x_test = np.load(os.path.join(config.data_dir, 'generator_dcnn_train_data_1000_d.npy'))

    all_tracks = [] 
    for track in x_train:
        all_tracks.append((track[1].replace('.npy', ext),save_dir, audio_dir))

    for track in x_valid:
        all_tracks.append((track[1].replace('.npy', ext), save_dir, audio_dir))

    for track in x_test:
        all_tracks.append((track[1].replace('.npy', ext), save_dir, audio_dir))

    print (len(all_tracks))

    all_tracks = [(all_tracks[i]) for i in range(len(all_tracks))]

    with Pool(N_WORKERS) as p:
        p.starmap(parallel_mel, all_tracks)
    
    print ("training data done")

    del all_tracks

    #### testing data 
    x_unseen_train, x_unseen_test = np.load(os.path.join(config.data_dir, 'gen_dcnn_unseen_data_500_d.npy'))
    
    all_tracks = [] 
    for track in x_train:
        all_tracks.append(track[1].replace('.npy', ext))

    for track in x_test:
        all_tracks.append(track[1].replace('.npy', ext))

    all_tracks = [(all_tracks[i]) for i in range(len(all_tracks))]

    with Pool(N_WORKERS) as p:
        p.starmap(parallel_mel, all_tracks)
       
    print ("testing data done") 
    del x_unseen_train, x_unseen_test, all_tracks
    

    print ("computing mean, std...")
    all_mels = [] 
    for track in x_train : 
        artist_id, feat_path, start_frame = track 
        feat = np.load(config.mel_path + feat_path.replace(ext, '.npy'))[:, start_frame : start_frame + config.input_frame_len]
        all_mels.append(feat)

    print ("mean:",np.mean(all_mels), "std:", np.std(all_mels))



def process_damp(audio_dir, mel_dir, ext):
    import damp_config 
    from utils import load_data_segment
    global N_WORKERS 

    train_artists = np.load(os.path.join(damp_config.data_dir, 'artist_1000.npy'))
    train_list, _ = load_data_segment(os.path.join(damp_config.data_dir, 'train_artist_track_1000.pkl'), train_artists)
    valid_list, _ = load_data_segment(os.path.join(damp_config.data_dir, 'valid_artist_track_1000.pkl'), train_artists)

    unseen_train_artists = np.load(os.path.join(damp_config.data_dir, 'unseen_artist_300_2.npy'))
    unseen_train_list, _ = load_data_segment(os.path.join(damp_config.data_dir, 'unseen_model_artist_track_300_2.pkl'), unseen_train_artists)
    unsene_valid_list, _ = load_data_segment(os.path.join(damp_config.data_dir, 'unseen_eval_artist_track_300_2.pkl'), unseen_train_artists)

    all_tracks = set()
    for i in range(len(train_list)):
        _, feat_path, _ = train_list[i]
        feat_path = feat_path.replace('.npy', ext)
        all_tracks.add((feat_path, mel_dir, audio_dir))
    for i in range(len(valid_list)):
        _, feat_path, _ = valid_list[i]
        feat_path = feat_path.replace('.npy',ext)
        all_tracks.add((feat_path, mel_dir, audio_dir))

    all_tracks = list(all_tracks)
    print (len(all_tracks))
    
    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)

    with Pool(N_WORKERS) as p:
        p.starmap(parallel_mel, all_tracks)
    


    all_tracks = set()
    for i in range(len(unseen_train_list)):
        _, feat_path, _ = unseen_train_list[i]
        feat_path =feat_path.replace('.npy', ext)
        all_tracks.add((feat_path, mel_dir, audio_dir))
    for i in range(len(unseen_valid_list)):
        _, feat_path, _ = unseen_valid_list[i]
        feat_path = feat_path.replace('.npy', ext)
        all_tracks.add((feat_path, mel_dir, audio_dir))

    all_tracks = list(all_tracks)
    print (len(all_tracks))
    
    with Pool(N_WORKERS) as p:
        p.starmap(parallel_mel, all_tracks)
    

def process_damp_mix():
    import damp_config
    process_damp(damp_config.mix_mel_dir, damp_config.mix_audio_dir, '.wav')

def process_damp_vocal():
    import damp_config
    process_damp(damp_config.vocal_mel_dir, damp_config.vocal_audio_dir, '.m4a')


if __name__ == '__main__' : 
    process_msd_singer()
    # process_damp_mix()
    # process_damp_vocal()
