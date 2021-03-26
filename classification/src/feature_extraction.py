import glob

import numpy as np
import pandas as pd
import yaml

from src.preprocess import Preprocess
from src.utils.mapfgraph import MapfGraph, grid_name_from_map
from src.utils.VizMapfGraph import VizMapfGraph
from pathlib import Path
from tqdm import tqdm
import itertools
from timeit import default_timer as timer

import os

base_dir = Path('../data/from-vpn')
maps_dir = base_dir / 'maps'
scen_suffix = 'custom'
scen_dir = base_dir / 'scen/scen-{s}'.format(s=scen_suffix)

raw_data_path = 'lazy-epea-icts-cbsh-sat-labelled-custom.csv'
default_features_path = 'Alldata-labelled-custom-with-features.csv'
# default_features_path = '../data/from-vpn/experiments/random/random-version2/all-with-features.csv'

with open("config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

features_cols = config['features'] + config['cat_features']
offline_features_only = [f for f in features_cols if '0.' not in f]

df = pd.read_csv(raw_data_path)
labelled_df = pd.read_csv(default_features_path)
labelled_df = labelled_df.dropna()
# labelled_df = pd.DataFrame(columns=labelled_df.columns)

if 'map' not in df.columns:
    df['map'] = df.GridName + '.map'
if 'scen' not in df.columns:
    if 'problem_type' in df.columns:
        df['scen'] = df.apply(
            lambda x: x['GridName'] + '-' + x['problem_type'] + '-*' + str(x['InstanceId']) + '.scen',
            axis=1)
    else:
        df['scen'] = df.apply(lambda x: x['GridName'] + '-' + scen_suffix + '-' + str(x['InstanceId']) + '.scen',
                              axis=1)

tqdm.pandas()


def problem_features(df, n_agents, instance_id, grid_name, problem_type=None):
    if 'problem_type' in df.columns:
        problem = df[(df.NumOfAgents == n_agents) & (df.InstanceId == instance_id) & (
                df.GridName == grid_name) & (df["problem_type"] == problem_type)]
    else:
        problem = df[(df.NumOfAgents == n_agents) & (df.InstanceId == instance_id) & (
                df.GridName == grid_name)]

    if len(problem) == 1:
        return problem.iloc[0][offline_features_only].to_dict()
    else:
        print("Problem wasn't found")
        return dict()


def feature_enhancement(row, graph):
    n_agents = int(row['NumOfAgents'])
    curr_scen = str(scen_dir / row['scen'])
    problem_type = None
    if 'problem_type' in row.keys():  # The case where the scen name might contain additional info
        curr_scen = glob.glob(curr_scen)[0]
        problem_type = row['problem_type']
    curr_instance_id = row['InstanceId']
    grid_name = grid_name_from_map(graph.map_filename)
    features = problem_features(labelled_df, n_agents, curr_instance_id, grid_name, problem_type)

    if features == dict():
        graph.load_agents_from_scen(curr_scen, n_agents, curr_instance_id)
        graph.feature_extraction(with_cell_features=False)
        features = graph.features
        # graph_filename = grid_name + '_' + str(curr_instance_id) + '_' + str(n_agents) + '.npz'
        # graph.draw_2d_mapf_representation(graph_filename)

    for feature, value in features.items():
        row[feature] = value

    return row


def feature_enhancement_per_group(group):
    curr_map = str(maps_dir / group['map'].iloc[0])
    if not os.path.isfile(curr_map):
        return group
    print(curr_map)
    graph = VizMapfGraph(map_filename=curr_map)
    graph.create_graph()
    return group.apply(lambda x: feature_enhancement(x, graph), axis=1)

# df = df.groupby('scen').progress_apply(feature_enhancement_per_group)
# df.to_csv(raw_data_path.split('.csv')[0] + '-with-features.csv', index=False)

#
# files = glob.iglob('../data/from-vpn/experiments/SAT/*.csv', recursive=True)
# for data_path in files:
#     print(data_path.split('.csv')[0] + '-with-features.csv')
#     df = pd.read_csv(data_path)
#
#     if 'map' not in df.columns:
#         df['map'] = df.GridName + '.map'
#     if 'scen' not in df.columns:
#         df['scen'] = df.apply(lambda x: x['GridName'] +  '-even-' + str(x['InstanceId']) + '.scen', axis=1)
#
#     df = df[~df.map.str.contains('warehouse')].groupby('scen').progress_apply(feature_enhancement_per_group)
#     df.to_csv(data_path.split('.csv')[0] + '-with-features.csv', index=False)
