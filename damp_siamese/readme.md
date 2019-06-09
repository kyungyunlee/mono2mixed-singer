### To train
```
# Train args
# --model_name : name of the model (for saving model)
# --model_type : 'mono', 'mix' or 'cross'
# to train with only monophonic tracks without using pretrained model
python train.py --model_name blahblah --model_type mono 

# to train cross model without using pretrained model
python train.py --model_name blahblah --model_type cross

# to train cross model with pretrained model
python train.py --model_name blahblah --model_type cross --pretrained 'path_to_model.h5' 
```

### To test 
Choose a testing scenario and a model to use
```
# Test args
# --model_path : path to trained model
# --model_type : 'mono', 'mix' or 'cross'
# --scenario : 'mono2mono', 'mix2mix' or 'mono2mix' 
# --pretrained : whether the model was trained with a pretrained model or not 

# Ex. to test cross model(trained with a pretrained model) for mono2mix scenario
python test.py --model_path 'path_to_cross_model.h5' --model_type cross --scenario mono2mix --pretrained

# if the model did not use a pretrained model
python test.py --model_path 'path_to_cross_model.h5' --model_type cross --scenario mono2mix 
```
