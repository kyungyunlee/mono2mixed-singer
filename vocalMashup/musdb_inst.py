''' Code for mixing tracks excluding the vocal tracks of MUSDB18 dataset
'''
import os 
import sys
import musdb 
from multiprocessing import Pool
import subprocess as sp 
import soundfile

savedir = 'musdb_accompaniment'
if not os.path.exists(savedir):
    os.mkdirs(savedir)

mus = musdb.DB(root="/mnt/bach4/kylee/speech-singingvoice/musdb")

def process(track):
    accompaniment = track.stems[1] + track.stems[2] + track.stems[3]
    print (accompaniment.shape, track.name)
    soundfile.write(os.path.join(savedir, track.name + '.wav'), accompaniment, 44100, format='WAV')

with Pool(8) as p:
    p.map(process, mus)



