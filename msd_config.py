data_dir = '/home1/irteam/users/kylee/dev/mono2mixed-singer/data/msd_data'
audio_dir = '/home1/irteam/users/jongpil/data/msd/songs' 
mel_dir = '/home1/irteam/users/kylee/data/msd_svd_mel_128'

sr=22050
n_fft=1024
hop_length=512
n_mels=128
input_frame_len=129
num_singers=1000 # number of training singers
num_test_singers=500 # number of testing singers
num_neg_singers=4 # number of singers for negative sampling
num_pos_tracks=1 # number of tracks for positive sampling 
margin=0.1 # hinge loss margin 

total_mean=0.26567551 # mean of the input melspectrogram of the training data
total_std=0.2700282 # std of the input melspectrogram of the training data

batch_size=32
num_epochs=1000
num_parallel=20

