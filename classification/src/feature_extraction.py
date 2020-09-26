import numpy as np
import pandas as pd
from src.preprocess import Preprocess
from src.utils.mapfgraph import MapfGraph
from pathlib import Path
from tqdm import tqdm
import itertools
from timeit import default_timer as timer

import os

base_dir = Path('../data/from-vpn')
maps_dir = base_dir / 'maps'
scen_dir = base_dir / 'scen/scen-even'

# raw_data_path = '../data/from-vpn/CBSH-C/cbsh-c.csv'
raw_data_path = 'AllData-labelled.csv'

df = pd.read_csv(raw_data_path)

if 'map' not in df.columns:
    df['map'] = df.GridName + '.map'
if 'scen' not in df.columns:
    df['scen'] = df.apply(lambda x: x['GridName'] + '-even-' + str(x['InstanceId']) + '.scen', axis=1)


def feature_enhancement(row, graph):
    n_agents = row['NumOfAgents']
    curr_scen = str(scen_dir / row['scen'])
    curr_instance_id = row['InstanceId']
    graph.load_agents_from_scen(curr_scen, n_agents, curr_instance_id)
    graph.feature_extraction()

    for feature, value in graph.features.items():
        row[feature] = value

    return row


def feature_enhancement_per_group(group):
    curr_map = str(maps_dir / group['map'].iloc[0])
    if not os.path.isfile(curr_map):
        return group
    graph = MapfGraph(curr_map)
    graph.create_graph()
    return group.apply(lambda x: feature_enhancement(x, graph), axis=1)


tqdm.pandas()
# groups = df[~df.map.str.contains('warehouse')].groupby('scen').groups
# rel_groups = dict(itertools.islice(groups.items(), 420, 500))
# res = {k: feature_enhancement_per_group(df.iloc[v]) for k, v in rel_groups.items()}
df = df[~df.map.str.contains('warehouse')].groupby('scen').progress_apply(feature_enhancement_per_group)
df.to_csv('AllData-with-features.csv', index=False)
