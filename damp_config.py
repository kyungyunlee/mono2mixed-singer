import os 
data_dir = '/home1/irteam/users/kylee/dev/mono2mixed-singer/data' # base data dir 
vocal_mel_dir = os.path.join(data_dir, 'damp_mel') # path to original DAMP data melspectrogram
vocal_audio_dir = os.path.join(data_dir, 'damp_audio') # path to original DAMP audio files 
mix_mel_dir = os.path.join(data_dir, 'damp_mashup_mix_snr_mel_v2') # path to DAMP mashup data melspectrogram
mix_audio_dir = os.path.join(data_dir,'damp_mashup_mix_snr_v2') # path to DAMP mashup data audio files 
bg_audio_dir = os.path.join(data_dir, 'musdb_accompaniment_combined')

num_singers=1000
num_pos_tracks=1
num_neg_artist=4
margin=0.1

sr=22050
n_fft=1024
hop_length=512
n_mels=128
input_frame_len=129

mix_total_mean = 0.22986141
mix_total_std = 0.27396998

vocal_total_mean=0.10185046
vocal_total_std=0.21093594

batch_size=32
num_epochs=1000
num_parallel=20
