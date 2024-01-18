# GRAPH-CONSTRAINED DIFFUSION FOR END-TO-END PATH PLANNING

## Data preparison
Make directories as below
```bash
sets_data/
  |--real/
    |--map/
    |--raw/
    |--trajectories/
  |--real2/
    |--map/
    |--raw/
    |--trajectories/
```
Then download original data from https://gaia.didichuxing.com/ and the GPS file under directory ``raw/``.
I have put a small sample file in raw/ for better understanding, simply delete the suffix "_small" to use it.
Then execute
```bash
python -m loader.preprocess.mm.process_all
```

## Model Training
Execute 
```
bash ./train.sh
```
The model will be found at ``sets_models/``
The evaluation results will be found at ``sets_res/``