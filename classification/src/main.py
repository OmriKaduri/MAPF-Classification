from sklearn.model_selection import GroupShuffleSplit

from preprocess import Preprocess
from mapf_eda import MapfEDA
from models.baselines import Baselines
from models.xgb_reg_model import XGBRegModel
from models.xgb_clf_model import XGBClfModel
from models.coverage_classifier_model import CoverageClassifier
from models.cost_sensitive_model import CostSensitiveClassifier
from experiments import map_type
# from models.torch_classifier import TorchClassifier
import pandas as pd
import numpy as np
import yaml
import random

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

features_cols = config['features']
cat_features_cols = config['cat_features']
max_runtime = config['max_runtime']
algorithms = config['algorithms']
runtime_cols = [algorithm + ' Runtime' for algorithm in algorithms]
runtime_cols.append('Y Runtime')
success_cols = [algorithm + ' Success' for algorithm in algorithms]
split_method = config['data_split_method']
preprocess = Preprocess(max_runtime, runtime_cols, features_cols, cat_features_cols, success_cols)
unsolved = config['unsolved_problems_only']
unsolved_txt = 'unsolved' if unsolved else 'all'
with_cat_features = '/with-cat' if len(cat_features_cols) > 0 else ''
exp_name = split_method + '-' + str(max_runtime) + '-' + str(unsolved_txt) + '/' + '-vs-'.join(
    algorithms) + with_cat_features
with_models = config['with_models']
# data_path = 'AllData-labelled-partial_problems_withoutSAT.csv'
data_path = 'AllData-labelled-with-features.csv'
# data_path = 'old-ICAPS-AllData-labelled.csv'
# data_path = 'AllData-labelled-with-features.csv'
# data_path = 'random-with-features.csv'
print("Starting experiment: {e}".format(e=exp_name))
df, features_cols = preprocess.load_labelled_results(data_path,
                                                     # drop_maps=['warehouse'],
                                                     unsolved_only=unsolved)
outer_splits = config['group_splits']


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


assert ((df.groupby(['NumOfAgents', 'InstanceId', 'GridName']).count().max() == 1).all())


def train_all_models(df, exp_name, with_plots=True, maptype=''):
    if maptype != '':
        exp_name += '/' + maptype

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
    baselines = Baselines(runtime_cols, max_runtime, features_cols, success_cols, maptype)

    xgb_reg = XGBRegModel(runtime_cols, max_runtime, offline_features_only, success_cols, maptype)
    # xgb_reg_online_selection = XGBRegModel(runtime_cols, max_runtime, features_cols, success_cols, maptype)
    # xgb_reg_online_selection.add_modelname_suffix(' Online selection')
    #
    xgb_clf = XGBClfModel(runtime_cols, max_runtime, offline_features_only, success_cols, maptype)
    # xgb_clf_online_selection = XGBClfModel(runtime_cols, max_runtime, features_cols, success_cols, maptype)
    # xgb_clf_online_selection.add_modelname_suffix(' Online selection')

    xgb_coverage = CoverageClassifier(runtime_cols, max_runtime, offline_features_only, success_cols, maptype)

    xgb_cost_sensitive = CostSensitiveClassifier(runtime_cols, max_runtime, offline_features_only, success_cols,
                                                 maptype)

    if split_method == 'in-maptype':
        splits = map_splits_in_type_generalization(df, n_splits=outer_splits, test_size=0.25, random_state=20)
    else:
        if split_method == 'in-map':
            groups = df['InstanceId']  # split by scenarios - some in test, all maps available in train
        elif split_method == 'between-maptypes':
            groups = df['maptype']  # split by matypes - some map types in train, some in test
        else:
            raise NotImplementedError("Data split method wasn't defined")
        gkf = GroupShuffleSplit(n_splits=outer_splits, test_size=0.25, random_state=20)
        splits = gkf.split(df, df['Y'], groups)

    for index, (tr_ind, test_ind) in enumerate(splits):
        print("Starting {i} outer fold out of {n}".format(i=index, n=outer_splits))

        X_train, X_test, y_train, y_test = df.iloc[tr_ind].copy(), df.iloc[test_ind].copy(), \
                                           df['Y'].iloc[tr_ind].copy(), df['Y'].iloc[test_ind].copy()

        X_train.to_csv('../data/from-vpn/splitted/X_train.csv', index=False)
        X_test.to_csv('../data/from-vpn/splitted/X_test.csv', index=False)

        baseline_preds = baselines.predict(X_train, X_test, y_test)
        for k, v in baseline_preds.items():
            mapf_eda.add_model_results(v, k)

        X_train_offline = X_train[X_train.columns.drop(list(X_train.filter(regex='0\.')))].copy()
        X_test_offline = X_test[X_test.columns.drop(list(X_test.filter(regex='0\.')))].copy()
        if with_models:
            xgb_coverage.train_cv(X_train_offline,
                                  n_splits=config['inner_splits'],
                                  hyperopt_evals=config['hyperopt_evals'],
                                  load=True,
                                  models_dir='models/coverage/{i}'.format(i=index),
                                  exp_type=exp_name)
            cov_preds = xgb_coverage.predict(X_test_offline, y_test)
            mapf_eda.add_model_results(cov_preds, 'P-Cov Runtime')

            xgb_reg.train_cv(X_train_offline,
                             n_splits=config['inner_splits'],
                             hyperopt_evals=config['hyperopt_evals'],
                             load=True,
                             models_dir='models/regression/{i}'.format(i=index),
                             exp_type=exp_name)
            reg_preds = xgb_reg.predict(X_test_offline, y_test)
            mapf_eda.add_model_results(reg_preds, 'P-Reg Runtime')

            # xgb_reg_online_selection.train_cv(X_train,
            #                                   model_suffix='-online-reg-model.xgb',
            #                                   n_splits=config['inner_splits'],
            #                                   hyperopt_evals=config['hyperopt_evals'],
            #                                   load=False)

            # xgb_reg_online_selection.predict(X_test, y_test, online_feature_extraction_time='0.9maxtime_1000calctime')
            # xgb_reg.plot_feature_importance()
            # X_test['P-Reg Runtime'] = reg_test_preds
            # mapf_eda.add_model_results(reg_test_preds, 'P-Reg Runtime')

            xgb_clf.train_cv(X_train_offline, X_train['Y_code'],
                             n_splits=config['inner_splits'],
                             hyperopt_evals=config['hyperopt_evals'],
                             load=True,
                             models_dir='models/classification/{i}'.format(i=index),
                             exp_type=exp_name)
            clf_preds = xgb_clf.predict(X_test_offline, y_test)
            X_test['P-Clf Runtime'] = clf_preds
            mapf_eda.add_model_results(clf_preds, 'P-Clf Runtime')

            xgb_cost_sensitive.train_cv(X_train_offline,
                                        n_splits=config['inner_splits'],
                                        hyperopt_evals=config['hyperopt_evals'],
                                        load=True,
                                        models_dir='models/cost-sensitive/{i}'.format(i=index),
                                        exp_type=exp_name)
            xgb_cost_sensitive.predict(X_test_offline, y_test)
            X_test['P-Cost Runtime'] = clf_preds
            mapf_eda.add_model_results(clf_preds, 'P-Cost Runtime')

            #
            # xgb_clf_online_selection.train_cv(X_train, y_train, model_suffix='-online-reg-model.xgb',
            #                                   n_splits=config['inner_splits'],
            #                                   hyperopt_evals=config['hyperopt_evals'],
            #                                   load=False)
            # xgb_clf_online_selection.predict(X_test, y_test, online_feature_extraction_time='0.9maxtime_60000calctime')
            # xgb_clf.plot_feature_importance()
            # X_test['P-Clf Runtime'] = clf_test_preds
            # mapf_eda.add_model_results(clf_test_preds, 'P-Clf Runtime')

        # mapf_eda.create_cumsum_histogram(X_test)
        if maptype != '':
            cactus_filename = maptype + '-cactus.jpg'
            stacked_filename = maptype + '-stacked_bar_rankings.jpg'
        else:
            cactus_filename = 'all-cactus.jpg'
            stacked_filename = 'stacked_bar_rankings.jpg'
        mapf_eda.accumulate_cactus_data(X_test, fold_number=index, dir='plots/cactus', load=True,
                                        exp_type=exp_name, max_time=max_runtime, step=max_runtime // 60)
        # mapf_eda.create_stacked_rankings(df, sorted_runtime_cols, filename=stacked_filename)

    mapf_eda.plot_cactus_graph(exp_name=exp_name)
    results = [baselines.print_results(),
               xgb_reg.print_results(with_header=False),
               xgb_clf.print_results(with_header=False),
               xgb_coverage.print_results(with_header=False),
               xgb_cost_sensitive.print_results(with_header=False)
               ]
    if maptype != '':
        coverage_box_plot_filename = maptype + '-cov_boxplot.jpg'
    else:
        coverage_box_plot_filename = 'cov_boxplot.jpg'
    # mapf_eda.plot_coverage_box_plot(pd.concat(results), filename=coverage_box_plot_filename)


def train_predict_per_maptype(df, exp_name):
    if 'maptype' not in df.columns:
        df['maptype'] = df.apply(lambda x: map_type(x['GridName']), axis=1)

    maptypes = df['maptype'].unique()
    for maptype in maptypes:
        print("Training model for maptype:", maptype)
        map_df = df[df['maptype'] == maptype]
        train_all_models(map_df, exp_name, with_plots=False, maptype=maptype)


# train_predict_per_maptype(df, exp_name=exp_name)
train_all_models(df, exp_name=exp_name, with_plots=False)
