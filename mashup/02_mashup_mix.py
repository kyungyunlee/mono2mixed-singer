'''
Create a mix of vocal and baackground data
'''
import os
import sys
import csv 
import librosa
import numpy as np
import pickle
import random
from pathlib import Path
# from multiprocessing import Pool

from utils import compute_gain

vocal_dir = 'data/damp_stft/'
bg_dir = 'data/musdb_accompaniment_combined_stft/'

wav_dir = 'data/damp_mashup_mix_snr_v2/'
stft_dir = 'data/damp_mashup_mix_snr_stft_v2/'
mel_dir = 'data/damp_mashup_mix_snr_mel_v2/'






filename = ''

bg_files = [song for song in Path(bg_dir).glob('*.npy')]

if not os.path.exists(stft_dir):
    os.makedirs(stft_dir)

if not os.path.exists(mel_dir):
    os.makedirs(mel_dir)

if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

def csv_to_pickle() : 
    track_to_match = {} 
    filename = 'mashup_pair_data_unseen_2.csv'
    with open(filename,'r') as csvfile : 
        csvreader = csv.DictReader(csvfile)
        line_count = 0 
        for row in csvreader : 
            if line_count == 0 :
                line_count +=1 
            else :

                # handle empty ones by choosing random background files 
                vocal_track_path = row['vocal_track_path']
                vocal_svd_time = float(row['svd_start_sec'])
                vocal_tempo = float(row['tempo'])
                svd_beat_start_sec = float(row['svd_beat_start_sec'])

                if row['beat_match_start_sec'] == '' : 
                    print ("empty")
                    found = False
                    for s in range(len(bg_files)):
                        try : 
                            bg = bg_track_path[s]
                            bg_tempo_beat = np.load('data/musdb_beat/' + bg.stem + '.npy')
                            bg_tempo = round(bg_tempo_beat[-1])
                            bg_beat = librosa.frames_to_times(bg_tempo_beat[:-1], sr=22050, hop_length=512)
                        except:
                            continue

                        if abs(tempo - bg_tempo) < 10  : 
                            found = True
                            for i in range(len(bg_beat)):
                                if i+ 1 >= len(bg_beat):
                                    break
                                start_beat = bg_beat[i]
                                next_beat = i
                                end_beat = bg_beat[next_beat]
                                while end_beat - start_beat < 6.0:
                                    next_beat += 1
                                    if next_beat >= len(bg_beat):
                                        break
                                    end_beat = bg_beat[next_beat]
                            
                                
                            bg_track_path = bg 
                            bg_start_time = start_beat
                            bg_end_time = end_beat 
                            break
                    
                    
                    if not found : 
                        print ("within tempo range not found")
                        bg = random.choice(bg_files) 
                        bg_tempo_beat = np.load('data/musdb_beat/' + bg.stem + '.npy')

                        bg_tempo = round(bg_tempo_beat[-1])
                        bg_beat = librosa.frames_to_time(bg_tempo_beat[:-1], sr=22050, hop_length=512)
                        start_beat = bg_beat[0]
                        next_beat = 10
                        end_beat = bg_beat[next_beat]
                        while end_beat - start_beat < 6.0:
                            next_beat += 1
                            if next_beat >= len(bg_beat):
                                break
                            end_beat = bg_beat[next_beat]

                        bg_track_path = bg 
                        bg_start_time = start_beat
                        bg_end_time = end_beat 

                else : 
                    bg_track_path = row['bg_track_path']
                    bg_start_time = float(row['beat_match_start_sec'])
                    bg_end_time = float(row['beat_match_end_sec'])

                print (vocal_track_path, vocal_svd_time, bg_track_path)
                
                try : 
                    track_to_match[vocal_track_path][vocal_svd_time] = [bg_track_path, bg_start_time, bg_end_time, svd_beat_start_sec]
                except : 
                    track_to_match[vocal_track_path] = {} 
                    track_to_match[vocal_track_path][vocal_svd_time] = [bg_track_path, bg_start_time, bg_end_time, svd_beat_start_sec]


    pickle.dump(track_to_match, open(filename.replace('.csv', '.pkl'), 'wb'))






def bc_mix(sound1, sound2):
    ''' 
    sound1 : vocal
    sound2 : bg 
    '''
    gain1 = compute_gain(sound1)  
    gain2 = compute_gain(sound2)
    
    # amplify background 
    while gain2 < 1.3 : 
        sound2 *= 1.2 
        gain2 = compute_gain(sound2)
    # print (gain1, gain2)

    while gain1 < 1.3 : 
        sound1 *= 1.2 
        gain1 = compute_gain(sound1)

    
    if gain1 > gain2 : 
        r = gain2 / gain1 
        sound1 *= r 
    else : 
        r = gain1 / gain2 
        sound2 *= r 
    
    sound1 = sound1 * 1.1
    
    # SNR 
    mult = np.power(10, (0 / 20.0)) # -6, -3, 0, 3, 6 db 
    sound1 *= mult  

    gain1 = compute_gain(sound1)
    gain2 = compute_gain(sound2)
    print ('middle', gain1, gain2)

    sound = sound1 + sound2 
   
    mix_gain =compute_gain(sound)
    while mix_gain < 3.8 :
        sound = sound * 1.1 
        mix_gain = compute_gain(sound)

    

    return sound


missing = 0 

def parallel_mash(vocal_track_path, svd_pairs): 
    global filename
    global vocal_dir, bg_dir, mel_dir, stft_dir, wav_dir 
    global gains, vocal_gains, bg_gains 
    global missing 

    vocal_y = np.load(vocal_dir + vocal_track_path) 
    
    for vocal_svd_time, bg_pair in svd_pairs.items() :
        

        vocal_svd_frame = int(librosa.time_to_frames(vocal_svd_time, sr=22050, n_fft=1024, hop_length=512))
        if vocal_svd_frame < 0 : 
            vocal_svd_frame = 0 

        mash_stft_path = stft_dir + vocal_track_path.replace('.npy', '_' + str(vocal_svd_frame) + '.npy')
        mash_mel_path = mel_dir + vocal_track_path.replace('.npy', '_' + str(vocal_svd_frame) + '.npy')

        if os.path.exists(mash_stft_path):
            print("already mixed")
            
            continue
        
        missing += 1
        
        bg_track_path, bg_start_time, bg_end_time, svd_beat_start_sec = bg_pair

        svd_beat_start_frame = librosa.time_to_frames(svd_beat_start_sec, sr=22050, hop_length=512, n_fft=1024)
        bg_start_frame = librosa.time_to_frames(bg_start_time, sr=22050, hop_length=512, n_fft=1024)
        bg_end_frame = librosa.time_to_frames(bg_end_time, sr=22050, hop_length=512, n_fft=1024)

        if bg_end_frame - bg_start_frame < 129 * 2 : 
            bg_end_frame = int(bg_start_frame  + 129*2)
        
        bg_y = np.load(bg_dir + Path(bg_track_path).stem + '.npy')
        bg_y_tmp = bg_y[:, bg_start_frame : bg_end_frame] 
        
        

        vocal_seg = vocal_y[:, svd_beat_start_frame : svd_beat_start_frame + (bg_end_frame - bg_start_frame)] 

        while bg_y_tmp.shape[1] < vocal_seg.shape[1] :
            bg_y_tmp = np.concatenate((bg_y_tmp, bg_y_tmp), axis=1)
        
        bg_y_tmp = bg_y_tmp[:, :vocal_seg.shape[1]]


        mix_out = bc_mix(vocal_seg, bg_y_tmp)

    
        if mix_out.shape[1] < 129*2:
            print (mix_out.shape)
        
        print (mix_out.shape)
        # save wav 
        wav = librosa.istft(mix_out, hop_length=512, win_length=1024)
        librosa.output.write_wav(wav_dir + vocal_track_path.replace('.npy', '_' + str(vocal_svd_frame) + '.wav'), wav, sr=22050)

        # save stft 
        np.save(mash_stft_path, mix_out)
        
        # save mel 
        S = np.abs(mix_out)
        mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=128)
        mel_S = np.dot(mel_basis, S)
        mel_S = np.log10(1+10*mel_S)
        mel_S = mel_S.astype(np.float32)

        np.save(mash_mel_path, mel_S)




gains = [] 
bg_gains = [] 
vocal_gains = [] 
def mash() : 
    global filename
    global missing 

    # filename = 'mashup_pair_data_unseen_2_partial.pkl'
    filename = 'mashup_pair_data_unseen_ver2_2.pkl'
    # filename ='mashup_pair_data_1_partial2.pkl'
    # filename ='mashup_pair_data_4.pkl'
    '''
    filename ='mashup_pair_data_0.pkl'
    filename ='mashup_pair_data_1.pkl'
    filename ='mashup_pair_data_1_partial.pkl'
    filename ='mashup_pair_data_1_partial2.pkl'
    filename ='mashup_pair_data_2.pkl'
    filename ='mashup_pair_data_2_parrtial.pkl'
    filename ='mashup_pair_data_2_partial2.pkl'
    filename ='mashup_pair_data_3.pkl'
    filename ='mashup_pair_data_4.pkl'
    '''


    mash_pickle = pickle.load(open(filename, 'rb'))

    args = [(vocal_track_path, svd_pairs) for vocal_track_path, svd_pairs in mash_pickle.items()]

    
    for i in range(len(args)):
        parallel_mash(args[i][0],args[i][1])
    


if __name__ == '__main__' : 
    from damp_dcnn.load_data import load_data_segment
    csv_to_pickle() 
    mash()


    


