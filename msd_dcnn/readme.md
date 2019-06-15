## Train, test singer classification model with Million song dataset 
Used MSD-singer data [here](https://github.com/kyungyunlee/MSD-singer). 
1000 singers for training, 500 unseen singers for testing.  


### To train
```
python train.py --model_name blahblah
```

### To test
```
python test.py --model_path 'models/blahblah.h5'
```
