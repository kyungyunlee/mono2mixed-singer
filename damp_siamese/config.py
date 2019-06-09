
data_dir = '/home1/irteam/users/kylee/dev/mono2mixed-singer/data/'
vocal_mp3_path = '/home1/irteam/users/kylee/data/audio/'
vocal_mel_path = '/home1/irteam/users/kylee/data/damp_mel/'
# vocal_mel_path = '/home1/irteam/users/kylee/data/damp_vocal_snr_lower3_mel/'
vocal_stft_path = '/home1/irteam/users/kylee/data/damp_stft/'

# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mix_mel/' # old 
# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_mel/' 
# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_mel/' 
mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_mel_v2/' 
mix_stft_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_stft_v2/' 

# snr 
# mix_mel_lower_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_lower6_el/' 
# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_lower6_el/' 
# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_lower3_mel/' 
# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_higher3_mel/' 

# mix_mel_path = '/home1/irteam/users/kylee/data/damp_mashup_mix_snr_higer6_mel/' 


bg_mp3_path = '/home1/irteam/users/kylee/data/musdb_accompaniment/'
bg_stft_path = '/home1/irteam/users/kylee/data/musdb_accompaniment_stft/'

num_singers=1000
num_pos_tracks=1
num_neg_artist=4 # try different number of negative artists
# num_hard_neg_artist=20
margin=0.1 #try different margins


sr=22050
n_fft=1024
hop=512
n_mels=128
input_frame_len=129

vocal_total_mean=0.10185046
vocal_total_std=0.21093594


mix_total_mean = 0.22986141 # new mashup
mix_total_std = 0.27396998 # new mashup



batch_size=32
num_epochs=1000
num_parallel=20


ss_total_mean=0.15362854
ss_total_std=0.20888159

