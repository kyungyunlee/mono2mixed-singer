# mono2mixed-singer

**Code is being updated**   
Supplementary material for "Learning a Joint Embedding Space of Monophonic and Mixed Music Signals for Singing Voice" - Kyungyun Lee, Juhan Nam, ISMIR 2019 Delft, Netherlands   
[paper](http://archives.ismir.net/ismir2019/paper/000034.pdf)  
[audio examples](http://kyungyunlee.github.io/archives/ISMIR2019-Mono2Mixed)

I tried to provide as much code and data configuration as possible to make the paper reproduction easier. Everything is provided except the actual mashup audio files (examples in blog), and you can run the files under `vocalMashup` to create data.  


### Requirements
Build a container using the dockerfile in `Dockerfile` directory.   
If you want to train your own system, make sure you download the public datasets (DAMP, musdb18, Million Song Dataset) and set the path in the config files (`msd_config.py` and `damp_config.py`).   

### To just use the trained networks
This uses the CROSS model to extract feature embeddings for singing voice. 
```
'''
args
    model_path : path to trained model weights
    audio_path : path to directory containing audio files
    domain : either "mono" or "mixed", depending on the type of the audio files
'''
python embeddings.py --model_path damp_siamese/models/cross_pretrained.h5 --audio_path examples_dir --domain mixed
```

### Dataset
**MSD-singer**   
This is created from the [Million Song Dataset](http://millionsongdataset.com/) by running singing voice detection.    
MSD-singer repo is [here](https://github.com/kyungyunlee/MSD-singer).   

**DAMP and DAMP-Mash**    
Download [DAMP](http://ccrma.stanford.edu/damp/) and [musdb18](http://sigsep.github.io/datasets/musdb.html). Create your own mashup. Code is under `vocalMashup` directory.  

### Preprocessing audio features  
Input to the model is a 3-sec melspectrogram. I recommend precomputing the melspectrogram of all the audio data to speed up the training process if enough disk space is available (34G for DAMP mashup, 119G for MSD-Singer)     
Code is in `feature_extract.py`. Run this file to extract features.    


### Training and testing
Different versions (random subsets) of datasets can be made using `damp_datamaker.py` and `msd_datamaker.py`.   
Check different subdirectories for instructions. 

**DAMP**   
For MONO, MIX and CROSS models in the paper, main code is in `damp_siamese` and pre-training model code is in `damp_dcnn`.   

**MSD-singer**   
Also, for testing generalizability of the model trained with the DAMP mashup dataset, I also trained a model with the MSD dataset and code is in `msd_siamese` and `msd_dcnn`.    

**All the trained weights are also provided under each folder's `models/` directory.** 


### Cite
```
@inproceedings{lee2019learning,
    title={Learning a Joint Embedding Space of Monophonic and Mixed Music Signals for Singing Voice},
    author={Lee, Kyungyun and Nam, Juhan},
    booktitle={Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), Delft, Netherland},
    year={2019}
}
```
