## Train and test MONO, MIX, CROSS models from the paper
Uses the DAMP original and mashup dataset  

### To train
Train args
* `--model_name` : name of the model (for saving model)
* `--model_type` : 'mono', 'mix' or 'cross'

``` 
# to train CROSS model with pretrained model
python train.py --model_name blahblah --model_type cross --pretrained 'path_to_model.h5' 

# to train CROSS model without using pretrained model
python train.py --model_name blahblah --model_type cross

# to train MONO model (with only monophonic tracks) without using pretrained model
python train.py --model_name blahblah --model_type mono 
```

### To test singer identification and query-by-singer
Test args
* `--model_path` : path to trained model
* `--model_type` : 'mono', 'mix' or 'cross'
* `--scenario` : 'mono2mono', 'mix2mix' or 'mono2mix' 
* `--pretrained` : whether the model was trained with a pretrained model or not 

```
# Ex. to test singer identification with CROSS model(trained with a pretrained model) for mono2mix scenario
python test_singer_id.py --model_path 'path_to_cross_model.h5' --model_type cross --scenario mono2mix --pretrained

python test_retrieval.py --model_path 'path_to_cross_model.h5' --model_type cross --scenario mono2mix --pretrained

# if the model did not use a pretrained model
python test.py --model_path 'path_to_cross_model.h5' --model_type cross --scenario mono2mix 
```
