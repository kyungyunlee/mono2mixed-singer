## Train and test singer identification on Million Song Dataset 
Uses MSD-singer data. The code used to create MSD-singer data is provided [here](https://github.com/kyungyunlee/MSD-singer).

### To train
```
# To train with pretrained classification model 
python train.py --model_name blahblah --pretrained_model '../msd_dcnn/models/classification_1000.h5'
```

### To test 
```
# To test siamese model that used a pretrained classification model 
python test.py --model_path 'models/msd_siamese_pretrained.h5' --pretrained
```

### Trained model 
`'models/msd_siamese_pretrained.h5'`

