from sklearn.model_selection import GroupShuffleSplit

from preprocess import Preprocess
from mapf_eda import MapfEDA
from models.baselines import Baselines
from models.cnn_clf_model import CNNClfModel
from models.cnn_reg_model import CNNRegModel
from models.cnn_coverage_model import CNNCoverageModel
from models.cnn_cost_sensitive_model import CNNCostSensitiveModel
# from models.nn_clf_model import NNClfModel
from experiments import map_type
# from models.torch_classifier import TorchClassifier
import pandas as pd
import numpy as np
import yaml
import random
import argparse

parser = argparse.ArgumentParser(description='Train and evaluate AS models')
parser.add_argument('--config', dest='config', type=str, nargs='?',
                    help='Config file to use. Default is `config.yaml`', default='config.yaml')

args = parser.parse_args()
config_file = args.config
print(f'Using config file: {config_file}')

with open(config_file, 'r') as stream:
    config = yaml.safe_load(stream)

features_cols = config['features']
cat_features_cols = config['cat_features']
use_cell_features = config['use_cell_features']
max_runtime = config['max_runtime']
algorithms = config['algorithms']
runtime_cols = [algorithm + ' Runtime' for algorithm in algorithms]
runtime_cols.append('Y Runtime')
success_cols = [algorithm + ' Success' for algorithm in algorithms]
split_method = config['data_split_method']
preprocess = Preprocess(max_runtime, runtime_cols, features_cols, cat_features_cols, success_cols, use_cell_features)
unsolved = config['unsolved_problems_only']
unsolved_txt = unsolved
with_cat_features = '/with-cat' if len(cat_features_cols) > 0 else ''
exp_name = split_method + '-' + str(max_runtime) + '-' + str(unsolved_txt) + '/' + '-vs-'.join(
    algorithms) + with_cat_features
with_models = config['with_models']
data_path = 'MovingAIData-labelled-with-features.csv'
# data_path = 'Alldata-labelled-custom-with-features.csv'
# data_path = 'AllData-labelled-with-features - without cell.csv'  # Even set only
# data_path = 'lazy-epea-icts-cbsh-sat-labelled-custom.csv'
# data_path = 'lazy-epea-icts-cbsh-sat-labelled-custom-with-features.csv'
# exp_name = 'custom/' + exp_name

# data_path = 'MovingAIAndCustomData-labelled.csv'
df, features_cols = preprocess.load_labelled_results(data_path,
                                                     drop_maps=['room-32-32-8'],
                                                     unsolved_filter=unsolved)
outer_splits = int(config['group_splits'])


# df['maptype'] = df.apply(lambda x: map_type(x['GridName']), axis=1)


def all_maps_by_types():
    cities = ['Berlin_1_256', 'Boston_0_256', 'Paris_1_256']
    games = ['brc202d', 'den312d', 'den520d', 'ht_chantry', 'ht_mansion_n', 'lak303d', 'lt_gallowstemplar_n', 'orz900d',
             'ost003d', 'w_woundedcoast']
    warehouses = ['warehouse-10-20-10-2-1', 'warehouse-10-20-10-2-2', 'warehouse-20-40-10-2-1',
                  'warehouse-20-40-10-2-2']
    empties = ['empty-8-8', 'empty-16-16', 'empty-32-32', 'empty-48-48']
    mazes = ['maze-128-128-1', 'maze-128-128-10', 'maze-128-128-2', 'maze-32-32-2', 'maze-32-32-4']
    rooms = ['room-32-32-4', 'room-64-64-16', 'room-64-64-8']
    randoms = ['random-32-32-10', 'random-32-32-20', 'random-64-64-10', 'random-64-64-20']
    return cities, games, warehouses, empties, mazes, rooms, randoms


def map_splits_in_type_generalization(df, n_splits=1, test_size=0.25, random_state=20):
    splits = []
    for split in range(n_splits):
        random.seed(random_state + split)
        map_types = all_maps_by_types()
        train_maps = []
        test_maps = []
        for maps in map_types:
            curr_train_maps = random.sample(maps,
                                            max(int((1 - test_size) * len(maps)),
                                                1))  # Choose at least one map from each type
            curr_test_maps = [map for map in maps if map not in curr_train_maps]
            train_maps.extend(curr_train_maps)
            test_maps.extend(curr_test_maps)
        tr_ind = df[df['GridName'].isin(train_maps)].index
        test_ind = df[df['GridName'].isin(test_maps)].index
        splits.append((tr_ind, test_ind))

    return splits


# assert ((df.groupby(['NumOfAgents', 'InstanceId', 'GridName', 'problem_type']).count().max() == 1).all())
assert (len(set(df['GridName'] == 32)))


def train_all_models(df, exp_name, outer_splits, with_plots=True, maptype='', problem_type='', load=True):
    if maptype != '':
        exp_name += '/' + maptype
    if problem_type != '':
        exp_name += '/' + problem_type

    print("Starting experiment: {e}".format(e=exp_name))
    mapf_eda = MapfEDA(df, runtime_cols)
    if with_plots:
        # mapf_eda.create_runtime_histograms()
        # mapf_eda.create_rankings_histograms()
        print("Plots!")
    # X_train = pd.read_csv('data/from-vpn/splitted/X_train.csv')
    # X_test = pd.read_csv('data/from-vpn/splitted/X_test.csv')
    # y_train = X_train['Y']
    # y_test = X_test['Y']

    offline_features_only = [f for f in features_cols if '0.' not in f]
    baselines = Baselines(runtime_cols, max_runtime, features_cols, success_cols, maptype + problem_type)

    cnn_coverage = CNNCoverageModel(runtime_cols, max_runtime, offline_features_only, success_cols,
                                    maptype + problem_type)

    cnn_cost_sensitive = CNNCostSensitiveModel(runtime_cols, max_runtime, offline_features_only, success_cols,
                                               maptype + problem_type)

    cnn_reg = CNNRegModel(runtime_cols, max_runtime, offline_features_only, success_cols, maptype + problem_type)

    cnn_clf = CNNClfModel(runtime_cols, max_runtime, offline_features_only, success_cols, maptype + problem_type)

    if split_method == 'in-maptype':
        splits = map_splits_in_type_generalization(df, n_splits=outer_splits, test_size=0.25, random_state=20)
    else:
        if split_method == 'in-map':
            groups = df['InstanceId']  # split by scenarios - some in test, all maps available in train
        elif split_method == 'between-maptypes':
            groups = df['maptype']  # split by matypes - some map types in train, some in test
        elif split_method == 'in-problem-type':  # Split by problem type (i.e. even vs random)
            groups = df['problem_type']
            outer_splits = min(outer_splits, len(set(df['problem_type'])))
        elif split_method == 'between-random-even-custom':
            problem_types = ['cross-sides', 'swap-sides', 'inside-out', 'outside-in', 'tight-to-tight', 'tight-to-wide']
            for problem_type in problem_types:
                df['problem_type'] = df['problem_type'].replace(problem_type, 'custom')
            groups = df['problem_type']
            outer_splits = 3

        elif split_method == 'no-split':  # Don't split
            groups = None
        else:
            raise NotImplementedError("Data split method wasn't defined")
        if groups is None:
            splits = [(list(range(len(df))), list(range(len(df))))]  # Both sets with all data
        else:
            gkf = GroupShuffleSplit(n_splits=outer_splits, test_size=0.25, random_state=20)
            splits = gkf.split(df, df['Y'], groups)

    for index, (tr_ind, test_ind) in enumerate(splits):
        print("Starting {i} outer fold out of {n}".format(i=index, n=outer_splits))

        X_train, X_test, y_train, y_test = df.iloc[tr_ind].copy(), df.iloc[test_ind].copy(), \
                                           df['Y'].iloc[tr_ind].copy(), df['Y'].iloc[test_ind].copy()
        if 'problem_type' in X_train.columns:
            print(set(X_train['problem_type']))
        # X_train.to_csv('../data/from-vpn/splitted/X_train.csv', index=False)
        # X_test.to_csv('../data/from-vpn/splitted/X_test.csv', index=False)

        baseline_preds = baselines.predict(X_train, X_test, y_test)
        for k, v in baseline_preds.items():
            mapf_eda.add_model_results(v, k)

        X_train_offline = X_train[X_train.columns.drop(list(X_train.filter(regex='0\.')))].copy()
        X_test_offline = X_test[X_test.columns.drop(list(X_test.filter(regex='0\.')))].copy()
        if with_models:
            cnn_clf.train_cv(X_train_offline, y_train, n_splits=1,
                             load=load,
                             models_dir='models/cnn-classification/{i}'.format(i=index),
                             exp_type=exp_name)
            clf_preds = cnn_clf.predict(X_test_offline, y_test)
            mapf_eda.add_model_results(clf_preds, 'P-Clf Runtime')
            X_test['P-Clf Runtime'] = clf_preds

            cnn_reg.train_cv(X_train_offline, y_train, n_splits=1,
                             load=load,
                             models_dir='models/cnn-regression/{i}'.format(i=index),
                             exp_type=exp_name)
            reg_preds = cnn_reg.predict(X_test_offline, y_test)
            mapf_eda.add_model_results(reg_preds, 'P-Reg Runtime')

            cnn_coverage.train_cv(X_train_offline, y_train, n_splits=1,
                                  load=load,
                                  models_dir='models/cnn-coverage/{i}'.format(i=index),
                                  exp_type=exp_name)
            cov_preds = cnn_coverage.predict(X_test_offline, y_test)
            mapf_eda.add_model_results(cov_preds, 'P-Cov Runtime')

            cnn_cost_sensitive.train_cv(X_train_offline, y_train, n_splits=1,
                                        load=load,
                                        models_dir='models/cnn-cost-sensitive/{i}'.format(i=index),
                                        exp_type=exp_name)
            cost_preds = cnn_cost_sensitive.predict(X_test_offline, y_test)
            X_test['P-Cost Runtime'] = cost_preds
            mapf_eda.add_model_results(cost_preds, 'P-Cost Runtime')

        if maptype != '':
            cactus_filename = maptype + '-cactus.jpg'
            stacked_filename = maptype + '-stacked_bar_rankings.jpg'
        else:
            cactus_filename = 'all-cactus.jpg'
            stacked_filename = 'stacked_bar_rankings.jpg'
        # mapf_eda.accumulate_cactus_data(X_test, fold_number=index, dir='plots/cactus', load=load,
        #                                 exp_type=exp_name, max_time=max_runtime, step=max_runtime // 60)
        # mapf_eda.create_stacked_rankings(df, filename=stacked_filename)

    # mapf_eda.plot_cactus_graph(exp_name=exp_name)
    if with_models:
        results = [baselines.print_results(exp_dir=exp_name + '/'),
                   cnn_reg.print_results(exp_dir=exp_name + '/', with_header=False),
                   cnn_clf.print_results(exp_dir=exp_name + '/', with_header=False),
                   cnn_coverage.print_results(exp_dir=exp_name + '/', with_header=False),
                   cnn_cost_sensitive.print_results(exp_dir=exp_name + '/', with_header=False),
                   # nn_clf.print_results(with_header=False)
                   ]
    else:
        results = baselines.print_results(with_header=True, exp_dir=exp_name + '/')
        # print(results.to_latex(index=False))
    if maptype != '':
        coverage_box_plot_filename = maptype + '-cov_boxplot.jpg'
    else:
        coverage_box_plot_filename = 'cov_boxplot.jpg'
    # mapf_eda.plot_coverage_box_plot(pd.concat(results), filename=coverage_box_plot_filename)
    return results


def train_predict_per_maptype(df, exp_name, outer_splits, load=True, problem_type=''):
    if 'maptype' not in df.columns:
        df['maptype'] = df.apply(lambda x: map_type(x['GridName']), axis=1)

    maptypes = df['maptype'].unique()
    for maptype in maptypes:
        print("Training model for maptype:", maptype)
        map_df = df[df['maptype'] == maptype]
        train_all_models(map_df, exp_name, outer_splits=outer_splits, with_plots=False, maptype=problem_type + maptype,
                         load=load)


def train_predict_per_problemtype(df, exp_name, outer_splits, load=True):
    if 'problem_type' not in df.columns:
        df['problem_type'] = 'even'
        print("Can't find problem type in columns. Assumed Even")
        # return

    problem_types = df['problem_type'].unique()
    results = []
    for problem_type in problem_types:
        print("Training model for problem type:", problem_type)
        map_df = df[df['problem_type'] == problem_type].reset_index()
        results.append(
            train_all_models(map_df, exp_name, outer_splits=outer_splits, with_plots=False, problem_type=problem_type,
                             load=load))


def train_predict_per_maptype_per_problem_type(df, exp_name, outer_splits, load=True):
    if 'problem_type' not in df.columns:
        print("Can't find problem type in columns")
        return

    problem_types = df['problem_type'].unique()
    results = []
    for problem_type in problem_types:
        print("Training model for problem type:", problem_type)
        map_df = df[df['problem_type'] == problem_type]
        results.append(
            train_predict_per_maptype(map_df, exp_name, outer_splits=outer_splits, load=load,
                                      problem_type=problem_type + '_'))


# train_predict_per_maptype_per_problem_type(df, exp_name=exp_name, outer_splits=outer_splits, load=False)
# train_predict_per_problemtype(df, exp_name=exp_name, outer_splits=outer_splits, load=False)
# train_predict_per_maptype(df, exp_name=exp_name,  outer_splits=outer_splits, load=False)
train_all_models(df, outer_splits=outer_splits, exp_name=exp_name, with_plots=False, load=False)
