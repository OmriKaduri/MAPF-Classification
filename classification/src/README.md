# MAPF Algorithm selection - documentation #

This page will contain all neccessery documentation in order to train the existing models on new data
(or just use the trained models in order to predict on new data)

1. [ Generate new data ](#data)
2. [ Add labels to new data ](#labels)
3. [ Preprocess the data ](#preprocess)
4. [ Explore the data ](#exploration)
5. [ Train models ](#train)
6. [ Prediction and metrics ](#predict)

## Data
In order to create new data, you need to generate new MAPF problems and solve them.
There are various methods to achieve this goal, but after solving the problems the data should be a csv file with the following columns (features for every solved MAPF problem):
```csv
GridRows, GridColumns, NumOfAgents, NumOfObstacles, BranchingFactor, ObstacleDensity, AvgDistanceToGoal, MaxDistanceToGoal, MinDistanceToGoal, AvgStartDistances, AvgGoalDistances, PointsAtSPRatio, A*+OD+ID Runtime, MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime, Basic-CBS/(A*/SIC)+ID Runtime, ICTS 3E +ID Runtime , EPEA*+ID Runtime
```
*Clarification* - `PointsAtSPRatio` computed as the number of cells at the grid that exists in a shortest path of some agent divided by the total number of cells at the grid. All other features should be self-describing. Otherwise - contact me (Omri) :)

Ofcourse that using the repository MAPF project to solve the problems would generate the output at the desired format.
You can see how to solve MAPF problems using the repo's code [here](https://github.com/OmriKaduri/MAPF-Classification/blob/master/README.md#how-to-run-solve-mapf-problems)

### Joining experiments ###
In a case you're running multiple experiements (different algorithm?) on the same set of problems,
you will probably want to merge those experiment results to a single dataset.
You can look at the `usage_example` in `experiments.py` file.

## Labels
After generating the data, we need to add labels to it in order to train the supervised models.

The label for each row would be the fastest algorithm solved the mapf problem. In order to add the label, you should run:
```python
from src.preprocess import Preprocess
max_runtime = 300000
runtime_cols = ['EPEA*+ID Runtime',
                'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
                'ICTS 3E +ID Runtime',
                'A*+OD+ID Runtime',
                'Basic-CBS/(A*/SIC)+ID Runtime',
                'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime',
                'Y Runtime']
                
preprocess = Preprocess(max_runtime, runtime_cols)
results_file = 'AllData.csv'
labelled_results = preprocess.label_raw_results(results_file)
```

Now, the labelled results saved at `AllData-labelled.csv` file.

## Preprocess
After labelling the data, we need to do some basic preprocessing.

The preprocess basically does:
1. putting upper bound for algorithm runtime
2. droping all non-solved MAPF problems

```python
df = preprocess.load_labelled_results(labelled_results)
```

## Exploration
Currently the exploration contains:
1. Generating histograms of the running time for each algorithm
2. Generating histograms of the rankings for each algorithm
*rankings* are the relative ranking for each MAPF problem, i.e. at a given problem MA-CBS might be ranked 1, CBS-H 2, Basic-CBS 3, ICTS 4, EPEA* 5, A* 6.

```python
from src.mapf_eda import MapfEDA

mapf_eda = MapfEDA(df, runtime_cols)
mapf_eda.create_runtime_histograms()
mapf_eda.create_rankings_histograms()
```

The exploration output (images) will be saved by default to files under `src` folder, but the path be given as an argument.

## Train
It is time to train some models!

Before training, we need to take one very important step - splitting our data to train and test.

The pretrained models data already splitted and can be used as:
```python
X_train = pd.read_csv('data/splitted/train_features.csv')
y_train = pd.read_csv('data/splitted/train_labels.csv')
X_test = pd.read_csv('data/splitted/test_features.csv')
y_test = pd.read_csv('data/splitted/test_labels.csv')
```

But you can ofcourse just use scikit-learn to split your new data:
```python
X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.25)
```

It is recommended to save the output of the split before any training and save it (or just using the same random_state for the split)

**Now we are really ready to train!**

Currently the supported models are:
1. XGBoost classification models
2. XGBoost regression models
3. CNN VGG16-based models (classification and regression)

Training the different models have basically the same basic API. For example, training XGBoost regression model:

```python 
xgb_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_reg.train_cv() #xgb_reg.train() can be used if CV (Cross Validation) takes too long
```

You can see more examples of training models at the [main.py file](https://github.com/OmriKaduri/MAPF-Classification/blob/master/classification/src/main.py)

*CNN Model* weights can be found [here](https://drive.google.com/file/d/1GrSK-M8jY0ZLAFuCkm-T_ahS6O556cZO/view?usp=sharing) 


## Predict
Now you will examine your new trained model!

First, you'll want to get some baselines. Those can be achieved using:
```python
baselines = baselines(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
baselines.print_results()
```

Now we have a file named `model-results.csv` (file name can be given as an argument to `print_results`)
containing the baselines. Let's add our model results to the file!

```python
xgb_reg.print_results()
```

That's it! The trained model (XGBoost regression model) results save to the file. Have fun! :)
