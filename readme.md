# mono2mixed-singer
Supplementary material for "Learning a Joint Embedding Space of Monophonic and Mixed Music Signals for Singing Voice" - Kyungyun Lee, Juhan Nam, ISMIR 2019 Delft, Netherlands   
[paper]()  
[blog]()  

I tried to provide as much code and data configuration as possible to make the paper reproduction easier. Everything is provided except the actual mashup audio files (examples are given), and you can run the files under `mashup` to create data easily.  


### Requirements
```
conda env create -f conda-keras-env.yml
pip install -r requirements.txt
```

### Data setup 
I generated my own dataset from the publicly available datasets. Make sure you create all the dataset and set the path in the config files (`msd_config.py` and `damp_config.py`).  


**MSD-singer**   
This is derived from the [Million Song Dataset](http://millionsongdataset.com/).  
Details can be found [here](https://github.com/kyungyunlee/MSD-singer). 

**DAMP**    
Code for music mashup and instructions are under `mashup` directory.   
You have to create your own mashup to proceed further.   

**Preprocessing audio features**   
Input to the model is melspectrogram. I recommend precomputing the melspectrogram of all the audio data to speed up the training process if enough disk space is available (34G for DAMP mashup, 119G for MSD-Singer)     
Code is in `feature_extract.py`. Run the file to extract features.    


### Training and testing
Different versions (random subsets) of datasets can be made using `damp_datamaker.py` and `msd_datamaker.py`.   

**DAMP**   
For MONO, MIX and CROSS models in the paper, main code is in `damp_siamese` and pre-trained model code is in `damp_dcnn`.   

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
