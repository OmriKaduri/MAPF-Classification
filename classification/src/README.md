# MAPF Algorithm selection - documentation #

1. [ Solved MAPF problems data preparation ](#Dataset)
2. [ MAPF feature extraction ](#Feature)
3. [ Experiments configuration ](#Configuration)
4. [ Train models ](#Train)
5. [ Deeper Analysis ](#Deeper)

## Dataset
You can either use our solved MAPF problems dataset, or create your own custom dataset.

### Solved MAPF problems dataset ###
We solved more then 200,000 MAPF problems with 5 different MAPF solvers. 
Specifically, we solved the classical [grid-based MAPF benchmark](https://www.movingai.com/benchmarks/mapf.html) problems, 
and our suggested extension to it.
Both benchmarks data (with features for algorithm selection) can be found [here](https://www.movingai.com/benchmarks/mapf.html).
 
### Custom MAPF problems dataset ###
Read this section if you choose to create your own solved MAPF dataset (with different solvers/problems). 
In order to create new data, you need to choose/generate MAPF problems and solve them.
The data about the solved MAPF problems should contain the following attributes for each problem: 
1. Number of agents solved
2. Instance id
3. Map file
4. Scenario type (i.e. even/random)
3. [ALG] Runtime (i.e. CBS-H Runtime)

You are advised to use this repository to solve your MAPF problems. See the relevant [documentation](https://github.com/OmriKaduri/MAPF-Classification/blob/master/README.md#how-to-run-solve-mapf-problems) for using our solvers.
Yet, you can solve with any other implementation, and provide the attributes above.

#### Joining experiments ####
In a case you ran multiple experiements (might be the case if running differenta algorithms from different code bases),
you want to merge those experiment results to a single dataset.
Relevant code is found in `usage_example` function under `experiments.py`.

## Feature Extraction ##
Having collected solved MAPF problems attribute, now you will extract features 
for algorithm selection. Two types of features might be extracted, depending on the 
machine learning approach being used to takcle the algorithm selection task. 
We provide code to extract both MAPF-specific features (for our XGBoost models),
and casting MAPF problems to images (for CNN models). 

We note by `RAW_DATA_PATH` as the path to your `csv` file with attributes about solved MAPF instances.
Specifically, next parts assume the following columns to exists in your `csv` file:
1. GridName (i.e. `Berlin-1-256`)
2. InstanceId
3. problem_type (i.e. even/random)
4. [ALG] Runtime
5. NumOfAgents

### MAPF-specific features ###
The code for extracting MAPF-specific features is in `feature_extraction.py`.
You need to update `raw_data_path` to point to your `csv` file. Also, update `scen` and `maps` directories according
to their location on your machine.
Then, choose relevant features to be extracted from `config.yaml` file (under `features` array).
Also, uncomment the following lines:
```python
df = df.groupby('scen').progress_apply(feature_enhancement_per_group)
df.to_csv(raw_data_path.split('.csv')[0] + '-with-features.csv', index=False)
```
Finally, run:
```commandline
python feature_extraction.py
```

  
### MAPF as an image ###
The code for casting MAPF to an image features is in `VizMapfGraph.py`.
First, make sure to update `scen` and `maps` directories according
to their location on your machine. 
At the `main` function, update the path for the `read_csv` command with your `raw_data_path`.
Finally, run:
```commandline
 python VizMapfGraph.py
```

## Configuration ##
Before evaluating the different algorithms and conducting algorithm selection experiments,
you need to define in `config.yaml` the following properties:
1. `max_runtime` - Maximum running time for a MAPF algorithm to solve a problem (in milliseconds). We used 300,000 (5 minutes) throughout our experiments.
2. `data_split_method` - Following our definitions of algorithm selection setups, here you can se the relevant setup train/test split logic (i.e., in-map, in-maptype, etc.)
3. `group_splits` - How many cross-validation splits should be used. We performed 3-fold CV at our experiments.
4. `with_models` - Boolean attribute - True in case you want to train/evaluate algorithm selection models, False otherwise.
5. `unsolved_problems_only` - You might want to evaluate different subsets of your solved MAPF dataset. We provide options to select 
problems where only one algorithm solved (`OnlyOneWin`), problems where at least one algorithm failed to solve `AtLeastOneFail`, 
and problems where all algorithms successfully solved (`AllSuccess)`. Leaving this option with `All` would 
evaluate over all dataset.

## Train and evaluate algorithm selection models
We support both CNN and XGBoost models over four different algorithm selection tasks.

Training the different models have basically the same basic API. For example, training XGBoost regression model:

```python 
xgb_reg = XGBRegModel(runtime_cols, max_runtime, features, success_cols, type_suffix)
xgb_reg.train_cv(X_train,
                 n_splits=config['inner_splits'],
                 hyperopt_evals=config['hyperopt_evals'],
                 load=load,
                 models_dir='models/regression/{i}'.format(i=index),
                 exp_type=exp_name)
reg_preds = xgb_reg.predict(X_test, y_test)
```

You can see more examples of training XGBoost models at [main.py file](https://github.com/OmriKaduri/MAPF-Classification/blob/master/classification/src/main.py).
For CNN models, check [main-cnn.py file](https://github.com/OmriKaduri/MAPF-Classification/blob/master/classification/src/main-cnn.py).

*CNN Model* weights can be found. 

The trained models results save to `model-results.csv` file.
The trained models (weights/xgb files) saved to the relevant models directory.

## Deeper Analysis
For deeper analysis of the MAPF results, we refer to our `analysis` notebook under `notebooks` directory. 
