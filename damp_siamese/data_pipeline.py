import os 
import sys
import pandas as pd 
import librosa 
import tensorflow as tf 
import numpy as np
import ast 
import random

sys.path.append('../')
import damp_config as config 


AUDIO_PARAMS = {
        "sr": config.sr,
        "n_fft": config.n_fft,
        "hop_length": config.hop_length,
        "n_mels": config.n_mels
        } 


def librosa_mel_filter(feat_config):

    sr, n_fft, n_mels = feat_config["sr"], feat_config["n_fft"] , feat_config["n_mels"]
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return mel_basis.astype(np.float32) 


def librosa_stft(audio_path, audio_dir, start, feat_config):
    with tf.device("/cpu:0") : 
        sr = feat_config["sr"]
        n_fft = feat_config["n_fft"]
        hop_length = feat_config["hop_length"] 

        input_args = [audio_path,  audio_dir, start, sr, n_fft, hop_length] 

        def _librosa_stft(audio_path, audio_dir, start, sr, n_fft, hop_length):
            y, _ = librosa.load(os.path.join(audio_dir, audio_path), sr=sr, offset=start, duration=config.input_second)
            mel = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
            # mel = np.expand_dims(mel, 0)
            mel = mel.astype(np.float32)
            # print ('inside', mel.shape)
            return mel

        return tf.py_func(_librosa_stft, input_args, (tf.float32))


def get_pos_tracks(plyr_perf_dict, plyrid, perf_key):
    with tf.device('/cpu:0'):
        args = [plyr_perf_dict, plyrid, perf_key]
        def _get_pos_tracks(plyr_perf_dict, plyrid, perf_key):
            # tracks = df[df['plyrid'] == plyrid]['perf_key'].tolist()
            tracks = plyr_perf_dict[plyrid]
            tracks.remove(perf_key)
            random.shuffle(tracks)

            return tracks[0]

        return tf.py_func(_get_pos_tracks, args, [tf.string])


def pick_start_svd(svds):
    with tf.device('/cpu:0'):
        args = [svds]
        def _pick(svds):
            svds = svds.decode()
            svds = ast.literal_eval(svds)
            random.shuffle(svds)
            return np.float32(svds[0])
                    
    return tf.py_func(_pick, args, tf.float32)



def get_dataset(input_csv,  model_type, input_frame_len, n_mels, batch_size, normalize=False, shuffle=True, infinite_generator=True, num_parallel_calls=10, cache_dir='/mnt/nfs/analysis/interns/klee/tf_cache/trash'):
    
    MEL_FILTER = librosa_mel_filter(AUDIO_PARAMS)
    MEL_FILTER = tf.convert_to_tensor(MEL_FILTER, dtype=tf.float32)
    
    df = pd.read_csv(input_csv)
    n_data = len(df)

    if model_type == 'mono': 
        anchor_path = config.vocal_audio_dir
        other_path = config.vocal_audio_dir
    elif model_type == 'mix': 
        anchor_path = config.mix_audio_dir
        other_path = config.mix_audio_dir
    elif model_type == 'cross': 
        anchor_path = config.vocal_audio_dir
        other_path = config.mix_audio_dir
 

    dataset = tf.data.Dataset.from_tensor_slices({key:df[key].values for key in df})


    dataset = dataset.take(30)
    
    
    '''
    # test the dataset
    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()


    with tf.Session() as sess:
        for k in range(10):
            sample_value = sess.run(sample)
            
            # print (sample_value["anchor"]["plyrid"])
            # print (sample_value["pos"]["plyrid"])
            # print(sample_value["neg1"]["plyrid"], sample_value["neg2"]["plyrid"], sample_value["neg3"]["plyrid"], sample_value["neg4"]["plyrid"])
            # print (sample_value["neg1_features"].shape)
            # print (sample_value["anchor"]["seg_start"])
            print (sample_value["seg_start"])
            print ("---------------------------")
    sys.exit()
    '''


    if shuffle:
        dataset = dataset.shuffle(buffer_size=n_data,reshuffle_each_iteration=True)

    dataset_positive = dataset.shuffle(buffer_size=n_data, reshuffle_each_iteration=True)
    dataset_negative = dataset.shuffle(buffer_size=n_data, reshuffle_each_iteration=True)


    # select start svd 
    dataset = dataset.map(lambda sample: dict(sample, seg_start=pick_start_svd(sample["svds"]))) 
    dataset_positive = dataset_positive.map(lambda sample: dict(sample, seg_start=pick_start_svd(sample["svds"]))) 
    dataset_negative = dataset_negative.map(lambda sample: dict(sample, seg_start=pick_start_svd(sample["svds"]))) 

   
    dataset = dataset.interleave(
        lambda anchor : tf.data.Dataset.zip((
            tf.data.Dataset.from_tensors(anchor).repeat(n_data),
            dataset_positive.filter(lambda positive : tf.logical_and(tf.equal(positive['plyrid'], anchor['plyrid']), tf.logical_not(tf.equal(positive['perf_key'], anchor['perf_key'])))).shuffle(n_data).take(n_data),

            dataset_negative.filter(lambda negative: tf.logical_not(tf.equal(negative['plyrid'], anchor['plyrid']))).shuffle(n_data).take(n_data), 
            dataset_negative.filter(lambda negative: tf.logical_not(tf.equal(negative['plyrid'], anchor['plyrid']))).shuffle(n_data).take(n_data), 
            dataset_negative.filter(lambda negative: tf.logical_not(tf.equal(negative['plyrid'], anchor['plyrid']))).shuffle(n_data).take(n_data), 
            dataset_negative.filter(lambda negative: tf.logical_not(tf.equal(negative['plyrid'], anchor['plyrid']))).shuffle(n_data).take(n_data))), 
        cycle_length=n_data, block_length=1)

    
    dataset = dataset.map(lambda anchor,pos, neg1, neg2, neg3, neg4 : {"anchor":anchor, "pos":pos, "neg1":neg1, "neg2": neg2, "neg3": neg3, "neg4": neg4})

    #### ANCHOR ##### 

    # get mel 
    dataset = dataset.map(lambda sample: dict(sample, anchor_features=librosa_stft(sample["anchor"]["perf_key"] + '.m4a', anchor_path, sample["anchor"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, anchor_features=tf.matmul(MEL_FILTER, sample["anchor_features"])))
    dataset = dataset.map(lambda sample: dict(sample, anchor_features=tf.expand_dims(sample["anchor_features"], axis=0)))
    
    ### POS #### 
    dataset = dataset.map(lambda sample: dict(sample, pos_features=librosa_stft(sample["pos"]["perf_key"] + '.m4a', other_path, sample["pos"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, pos_features=tf.matmul(MEL_FILTER, sample["pos_features"])))
    dataset = dataset.map(lambda sample: dict(sample, pos_features=tf.expand_dims(sample["pos_features"], axis=0)))
    
    #### NEG ### 
    dataset = dataset.map(lambda sample: dict(sample, neg1_features=librosa_stft(sample["neg1"]["perf_key"] + '.m4a', other_path, sample["neg1"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, neg1_features=tf.matmul(MEL_FILTER, sample["neg1_features"])))
    dataset = dataset.map(lambda sample: dict(sample, neg1_features=tf.expand_dims(sample["neg1_features"], axis=0)))

    dataset = dataset.map(lambda sample: dict(sample, neg2_features=librosa_stft(sample["neg2"]["perf_key"] + '.m4a', other_path, sample["neg2"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, neg2_features=tf.matmul(MEL_FILTER, sample["neg2_features"])))
    dataset = dataset.map(lambda sample: dict(sample, neg2_features=tf.expand_dims(sample["neg2_features"], axis=0)))

    dataset = dataset.map(lambda sample: dict(sample, neg3_features=librosa_stft(sample["neg3"]["perf_key"] + '.m4a', other_path, sample["neg3"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, neg3_features=tf.matmul(MEL_FILTER, sample["neg3_features"])))
    dataset = dataset.map(lambda sample: dict(sample, neg3_features=tf.expand_dims(sample["neg3_features"], axis=0)))

    dataset = dataset.map(lambda sample: dict(sample, neg4_features=librosa_stft(sample["neg4"]["perf_key"] + '.m4a', other_path, sample["neg4"]["seg_start"], AUDIO_PARAMS))) 
    dataset = dataset.map(lambda sample: dict(sample, neg4_features=tf.matmul(MEL_FILTER, sample["neg4_features"])))
    dataset = dataset.map(lambda sample: dict(sample, neg4_features=tf.expand_dims(sample["neg4_features"], axis=0)))


    # CACHE 
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        dataset = dataset.cache(cache_dir)
   
    if infinite_generator:
        dataset = dataset.repeat(count=-1)

    dataset = dataset.map(
            lambda sample: [sample["anchor_features"], 
                            sample["pos_features"],
                            sample["neg1_features"],
                            sample["neg2_features"],
                            sample["neg3_features"],
                            sample["neg4_features"]
                            ])

    '''
            lambda sample: ({"anchor": sample["anchor_features"],
                            "anchor_plyrid": sample["anchor"]["plyrid"],
                            "anchor_start" : sample["anchor"]["seg_start"],
                            "pos": sample["pos_features"],
                            "neg1": sample["neg1_features"],
                            "neg2": sample["neg2_features"],
                            "neg3": sample["neg3_features"],
                            "neg4": sample["neg4_features"]}))
    '''
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(batch_size)
    return dataset,n_data 





             

if __name__ == '__main__': 
    dataset, n_data = get_dataset(os.path.join(config.data_dir, 'damp_train_train.csv'), 'mono', 129,128,8)
    print (n_data)
    # , list(range(30)), 'mono2mix', 129, 128, 8) 

    # test the dataset
    iterator = dataset.make_one_shot_iterator()
    sample = iterator.get_next()


    with tf.Session() as sess:
        for i in range(1000):
            sample_value = sess.run(sample)
            print (len(sample_value))
            print (sample_value[0].shape)




