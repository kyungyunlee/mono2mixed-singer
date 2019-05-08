# mono2mixed-singer

### Requirements
I created a conda environment and to replicate, run following
```
conda env create -f conda-keras-env.yml
pip install -r requirements.txt
```

### Mashup procedure
1. Preprocess damp and musdb18 (background) data by computing beat tracking and chromagram
```
python mashup_preprocess.py
```

2. Find a matching vocal and background tracks based on key, tempo
```
python mashup_find_match.py --set_type train --curr_batch 0 --total_batch 5
python mashup_find_match.py --set_type train --curr_batch 1 --total_batch 5
...
python mashup_find_match.py --set_type train --curr_batch 4 --total_batch 5

python mashup_find_match.py --set_type test --curr_batch 0 --total_batch 3
python mashup_find_match.py --set_type test --curr_batch 1 --total_batch 3
python mashup_find_match.py --set_type test --curr_batch 2 --total_batch 3
```

3. Mix vocal and background tracks after adjusting gain 

### Training procedure
1. make dataset (mashup, data split) 
2. preprocess audio and data
3. train network 

