'''
Given a vocal recording and a list of background recordings, select the "mashable" background tracks with the vocal recording. 
'''
import os
import sys
import librosa
import random
import numpy as np
import scipy
import pickle
from pathlib import Path
from multiprocessing import Pool

NUM_PARALLEL = 20

AUDIO_PARAMS = {
        'sr': 22050,
        'n_fft': 1024,
        'hop_length': 512,
        'frame_length': 1024,
        'n_chroma': 12
        }

MAJOR = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#','B']
MINOR = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#','B']
MAJ_MIN_RELATIVES = {'B': 'G#',
                     'F#': 'D#',
                     'C#':'A#',
                     'G#':'F',
                     'D#': 'C',
                     'A#': 'G',
                     'F': 'D', 
                     'C':'A',
                     'G': 'E',
                     'D': 'B',
                     'A': 'F#',
                     'E': 'C#', 
                     'B': 'G#',
                     'F#': 'D#', 
                     'C#': 'A#'}
MIN_MAJ_RELATIVES = {v:k for k,v in MAJ_MIN_RELATIVES.items()}


def vocal_detection(y, input_frame_len, input_hop_len):
    rmse = librosa.feature.rmse(y, frame_length=AUDIO_PARAMS['frame_length'], hop_length=AUDIO_PARAMS['hop_length'], center=True)
    rmse = rmse[0]

    threshold = 0.04
    
    vocal_segments = np.where(rmse>threshold)[0]
    binary = rmse>threshold * 1
    vocal_segments = [] 

    for i in range(0, binary.shape[0], input_hop_len):
        curr_segment = binary[i:i+input_frame_len]
        vocal_ratio = np.sum(curr_segment) / input_frame_len
        if vocal_ratio > 0.8 :
            start_time = librosa.frames_to_samples(i, hop_length=AUDIO_PARAMS['hop_length'])
            # vocal_segments.append(round(start_time, 2))
            vocal_segments.append(start_time)

    return vocal_segments 



def ks_key(X):
    #  code from  'https://gist.github.com/bmcfee/1f66825cef2eb34c839b42dddbad49fd'
    X = scipy.stats.zscore(X)

    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)

    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)

    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)

    return major.T.dot(X), minor.T.dot(X)



def compute_beat(y):
    # beat tracking 
    tempo, beat = librosa.beat.beat_track(y, sr=22050, units='samples')
    return {'tempo': round(tempo), 'beat': beat} 


def compute_key(y): 
    # chroma 
    chromagram = librosa.feature.chroma_stft(y, sr=22050, n_chroma=12, n_fft=1024)
    chroma_sum = np.sum(chromagram, axis=1)
    
    # key 
    maj_key, min_key = ks_key(chroma_sum)
    if np.max(maj_key) > np.max(min_key):
        # major
         key_idx = np.argmax(maj_key)
         key = MAJOR[key_idx]
         ismajor = True
    else : 
        key_idx = np.argmax(min_key)
        key = MINOR[key_idx]
        ismajor = False


    return key, ismajor


def bg_handler(bg_track, duration=3.0):

    bg_y, _ = librosa.load(str(bg_track), sr=AUDIO_PARAMS['sr'])
    bg_tempo_beat = compute_beat(bg_y)
    segment_key = {} 
    for onset_sample in bg_tempo_beat['beat'] : 
        seg = bg_y[onset_sample : onset_sample + int(duration * AUDIO_PARAMS['sr'])]
        key, ismajor = compute_key(seg)
        if len(seg) == int(duration * AUDIO_PARAMS['sr']): 
            segment_key[onset_sample]  = (key, ismajor)
    print (bg_track.stem)
    return bg_track.stem, {'y':bg_y, 'tempo': bg_tempo_beat['tempo'], 'beat': segment_key}



def precompute_bg_info(bg_dir, duration=3.0):
    
    list_of_bg = [bg_track for bg_track in Path(bg_dir).glob('*.wav')]

    with Pool(NUM_PARALLEL) as p : 
        res = p.map(bg_handler, list_of_bg)
    
    bg_info = {} 
    for result in res : 
        bg_info[result[0]] = result[1]

    print('saving to bg_info.pkl')
    pickle.dump(bg_info, open('bg_info.pkl', 'wb'))




def compute_loudness(y):
    '''
    Args : 
        y : audio signal in samples
    Return : 
        mean_rms : (float) 
    '''
    rms = librosa.feature.rmse(y=y, frame_length=AUDIO_PARAMS['frame_length'])
    rms_filter_ind = np.where(rms >= 0.04)
    rms_filter = rms[rms_filter_ind]
    mean_rms = np.mean(rms_filter)
    return mean_rms




def find_mashup_pairs(vocal_path, bg_dir, duration=3.0, num_segments=10):
    '''
    For each vocal segment, find the best matching background music
    Args:
        vocal_path: path to vocal recording
        bg_dir: path to directory containing background tracks
        duration : duration of segment to perform mashup in seconds (default=3.0)
        num_segments: number of maximum vocal segments to perform mashup (default=10)
    Return : 
        matching_pair_result : 
        
    '''
    print('Finding background pairs for %s'%vocal_path)
    input_frame_len = librosa.time_to_frames(duration, sr=AUDIO_PARAMS['sr'], hop_length=AUDIO_PARAMS['hop_length'])
    input_hop_len = input_frame_len // 2

    vocal_y, _ = librosa.load(str(vocal_path), sr=AUDIO_PARAMS['sr'])
    # tempo, beat detection
    vocal_tempo_beat = compute_beat(vocal_y)
    print ('Vocal track tempo:', vocal_tempo_beat['tempo'])
    # print (vocal_tempo_beat['beat'])

    # vocal detection 
    vocal_segments = vocal_detection(vocal_y, input_frame_len, input_hop_len) 
    # print (vocal_segments) 
    vocal_segments = vocal_segments[:num_segments]
    
    
    # load bg info 
    if os.path.exists('bg_info.pkl'):
        bg_info = pickle.load(open('bg_info.pkl', 'rb'))
        
    else : 
        print ('bg_info.pkl does not exist, Computing info...')
        precompute_bg_info(bg_dir)
        print ('Done. Run the script again')
        sys.exit()


    # find bg track with same (or similar tempo)
    list_of_bg = [bg_track for bg_track in Path(bg_dir).glob('*.wav')]
    random.shuffle(list_of_bg)
    same_tempo = []
    near_tempo = [] 
    for bg_track in list_of_bg:
        bg_track_info= bg_info[bg_track.stem]
        if bg_track_info['tempo']== vocal_tempo_beat['tempo'] : 
            same_tempo.append((bg_track, bg_track_info))
        elif abs(bg_track_info['tempo'] - vocal_tempo_beat['tempo']) < 4: 
            near_tempo.append((bg_track, bg_track_info))

    if len(same_tempo) == 0 and len(near_tempo) == 0 : 
        print ("no  matching background music found for given vocal recording's tempo..try finding more background tracks.")
        sys.exit()

    else :

        if len(same_tempo) > 0 : 
            random.shuffle(same_tempo)
            candidate_bg = same_tempo.copy()
            candidate_bg = candidate_bg + near_tempo

        else :
            random.shuffle(near_tempo)
            candidate_bg = near_tempo.copy()

        print ('Finding pairs from %d candidate tracks with same (or similar) tempo'%len(candidate_bg))
        
        matching_pair_result = {} 
        # find key matching segment for each vocal segment 
        for i in range(len(vocal_segments)):
            start_sample = vocal_segments[i]
            curr_seg_y = vocal_y[start_sample : start_sample + int(duration * AUDIO_PARAMS['sr'])]
            vocal_key, vocal_ismajor = compute_key(curr_seg_y)
            print ("curr vocal segment key :", vocal_key, vocal_ismajor)
           
            matching_pair_found = False 

            random.shuffle(candidate_bg)
           
            while not matching_pair_found  : 
                for bg in candidate_bg : 
                    bg_y = bg[1]['y'] 
                    bg_beat = bg[1]['beat'] 
                    candidate_start_samples = [] 
                    for bg_onset_sample, (bg_key, bg_ismajor) in bg_beat.items() :
                        # key matching 
                        if vocal_key == bg_key and (vocal_ismajor and bg_ismajor) :
                            # same key found 
                            candidate_start_samples.append((bg_onset_sample, bg_key, bg_ismajor))
                     
                        else : 
                            # use the relative keys 
                            if vocal_ismajor : 
                                vocal_min_key = MAJ_MIN_RELATIVES[vocal_key]
                                if vocal_min_key == bg_key : 
                                    candidate_start_samples.append((bg_onset_sample, bg_key, bg_ismajor))
                            else:
                                vocal_maj_key = MIN_MAJ_RELATIVES[vocal_key]
                                if vocal_maj_key == bg_key : 
                                    candidate_start_samples.append((bg_onset_sample, bg_key, bg_ismajor))

                    if len(candidate_start_samples) > 0 : 
                        random.shuffle(candidate_start_samples)
                        matching_pair_result[vocal_segments[i]] = (bg[0], candidate_start_samples[0]) 
                        matching_pair_found = True
                break

            
            if not matching_pair_found :
                print ("no matching pair")
            else:      
                # print (matching_pair_result)
                print ("success")

                

        return matching_pair_result 






def mash(vocal_path, vocal_start, bg_path, bg_start, duration):
    '''
    Mix vocal and background tracks at the corresponding segment. 
    Args: 
        vocal_path: path to vocal track
        vocal_start: onset in samples for vocal segment 
        bg_path: path to background track 
        bg_start : onset in samples for background segment 
        duration : duration of the mix
    Return : 
        output : mixed signal in samples 
    '''
    input_sample_len = int(duration * AUDIO_PARAMS['sr'])
    vocal_y, _ = librosa.load(str(vocal_path), sr=AUDIO_PARAMS['sr'])
    vocal_y = vocal_y[vocal_start : vocal_start + input_sample_len] 
    bg_y, _ = librosa.load(str(bg_path), sr=AUDIO_PARAMS['sr'])
    bg_y = bg_y[bg_start : bg_start + input_sample_len]
    
    # adjust loudness 
    vocal_gain = compute_loudness(vocal_y)
    bg_gain = compute_loudness(bg_y)
    print (vocal_gain, bg_gain)
    
    while bg_gain < 0.3:
        bg_y *= 1.1 
        bg_gain = compute_loudness(bg_y)

    while vocal_gain < 0.3 : 
        vocal_y *= 1.1
        vocal_gain = compute_loudness(vocal_y)

    if vocal_gain > bg_gain : 
        r = bg_gain / vocal_gain
        vocal_y *= r
    else : 
        r = vocal_gain / bg_gain
        bg_y *= r 

    output = vocal_y + bg_y 
    output_gain = compute_loudness(output)
    print ("output gain", output_gain)
    
    while output_gain > 0.4 : 
        output = output / 1.1
        output_gain = compute_loudness(output)

    return output 
        

def process_damp_data(artist_tracks_file):
    sys.path.append('../')
    import damp_config 

    damp_data_dir = damp_config.vocal_audio_dir
    musdb_data_dir = damp_config.bg_audio_dir
    # musdb_data_dir = 'background_tracks' 

    if not os.path.exists('damp_mashup_output'):
        os.makedirs('damp_mashup_output')
 

    train_dict = pickle.load(open(artist_tracks_file, 'rb'))

    
    vocal_paths = []
    for artist_id, track_list in train_dict.items():
        for track_id in track_list : 
            vocal_track_path = os.path.join(damp_data_dir, track_id + '.m4a') 

            mashability_result = find_mashup_pairs(vocal_track_path, musdb_data_dir)
           
            for start_sample, (bg_track, (bg_start_sample, bg_key, bg_ismajor)) in mashability_result.items(): 
                print (start_sample, bg_track, bg_start_sample)
                mixed_output = mash(vocal_track_path, start_sample, bg_track, bg_start_sample, 3.0)
                start_frame = librosa.samples_to_frames(start_sample, hop_length =damp_config.hop_length, n_fft=damp_config.n_fft)
                # librosa.output.write_wav(os.path.join('damp_mashup_output', Path(vocal_path).stem + '_' + str(start_frame) +'.wav'), mixed_output, sr=AUDIO_PARAMS['sr'])






if __name__ == '__main__':
    import pickle
    train_data_path = '../data/train_artist_track_1000.pkl'
    valid_data_path = '../data/valid_artist_track_1000.pkl'
    unseen_model_data_path = '../data/unseen_model_artist_tracks_300.pkl'
    unseen_eval_data_path = '../data/unseen_eval_artist_tracks_300.pkl'
    
    print ("processing train data")
    process_damp_data(train_data_path)
    print ("processing valid data")
    process_damp_data(valid_data_path)
    print ("processing unseen model data")
    process_damp_data(unseen_model_data_path)
    print ("processing unseen eval data")
    process_damp_data(unseen_eval_data_path)


