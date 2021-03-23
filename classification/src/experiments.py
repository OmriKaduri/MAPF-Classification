from collections import defaultdict

import yaml
import glob
import pandas as pd
import numpy as np
from pathlib import Path

from preprocess import Preprocess
from preprocess import map_type


def merge_dataframes(df1, df2):
    merged = df1.merge(df2, on=['GridName', 'InstanceId', 'NumOfAgents', 'problem_type'],
                       how='outer')
    cols = list(merged.columns)

    x_cols = [col for col in cols if '_x' in col and 'Unnamed' not in col]
    y_cols = [col for col in cols if '_y' in col and 'Unnamed' not in col]

    for x_col in x_cols:
        col = x_col.split('_')[0]
        y_col = col + '_y'
        merged[col] = merged[x_col].where(merged[x_col].notnull(), merged[y_col])

    # merged[x_cols] = merged[x_cols].combine_first(merged[y_cols].rename(columns=dict(zip(y_cols, x_cols))))

    # Dropping the _x suffix from _x cols
    # merged = merged.rename(
    #     columns=dict(zip(x_cols, [col.split('_')[0] for col in x_cols])))

    # Dropping non-relevant cols left from merging the dataframes
    merged = merged.drop(y_cols + x_cols, axis=1)
    return merged


class Experiment:
    def __init__(self):
        with open("config.yaml", 'r') as stream:
            self.config = yaml.safe_load(stream)
        self.experiments = defaultdict(list)
        self.df = pd.DataFrame()
        self.runtime_cols = []
        self.success_cols = []
        self.max_runtime = self.config['max_runtime']
        self.alg_runtime_mapping = {
            'EPEA*+ID Runtime': 'epea Runtime',
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime': 'macbs Runtime',
            'ICTS 3E +ID Runtime': 'icts Runtime',
            'ICTS 3E+ID Runtime': 'icts Runtime',
            'ICTS 3E Runtime': 'icts Runtime',
            'A*+OD+ID Runtime': 'astar Runtime',
            'Basic-CBS/(A*/SIC)+ID Runtime': 'cbs Runtime',
            'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime': 'cbsh Runtime',
            'CBSH-C Runtime': 'cbsh-c Runtime',
            'cbsh Runtime': 'cbsh-c Runtime',
            'LazyCBS Runtime': 'lazycbs Runtime',
            'IDCBS Runtime': 'idcbs Runtime',
            'SAT Runtime': 'sat Runtime'
        }
        self.alg_success_mapping = dict((Preprocess.runtime_to_success(k), Preprocess.runtime_to_success(v)) for k, v in
                                        self.alg_runtime_mapping.items())

    @staticmethod
    def load_experiment_by_type(experiment_key, exp):
        curr_exp = pd.read_csv(exp)
        # if 'CBSH-C' in experiment_key:
        #     curr_exp = curr_exp.dropna()
        return curr_exp

    @staticmethod
    def remove_group_failures(group, success_cols):
        # TODO: Fixme, check if there is a group failure using sum(1).min(), and then cut the group untill that argmin
        first_failure = group[success_cols].sum(1).argmin()
        if first_failure == 0:
            return group
        return group[:first_failure]

    def load(self):
        for exp in glob.iglob(self.config['experiments_path'] + '/*.csv', recursive=True):
            print(exp)
            # if not ('Version' in exp or 'CBSH' in exp or 'EPEA' in exp):  # CBSH-C and Version only
            #     print("Skipping on", exp)
            #     continue
            experiment_type = Path(Path(exp).parent).stem
            curr_exp = Experiment.load_experiment_by_type(experiment_type, exp)
            curr_exp = curr_exp[curr_exp.NumOfAgents > 1]  # Drop all not multi-agent problems
            curr_exp = curr_exp.sort_values(['GridName', 'InstanceId', 'NumOfAgents'])
            curr_exp = curr_exp.rename(columns=lambda x: ' '.join(x.split()))
            # Dropping rows where no solver succeeded to solve (if there is any)
            curr_success_cols = list(curr_exp.filter(regex="Success$"))
            curr_runtime_col = list(curr_exp.filter(like="Runtime"))
            if len(curr_success_cols) == 0:
                curr_exp[experiment_type.lower() + ' Success'] = (
                        curr_exp[curr_runtime_col] <= self.config['max_runtime']).astype(int)
                curr_success_cols.append(experiment_type.lower() + ' Success')

            if 'problem_type' not in curr_exp.columns:
                return "problem_type must be column in data"

            # Remove rows after first failure (if exist).
            # curr_exp = curr_exp.groupby(['InstanceId', 'GridName']).apply(
            #     lambda x: Experiment.remove_group_failures(x, curr_success_cols)).reset_index(drop=True)
            # curr_exp = curr_exp[curr_exp[curr_success_cols].sum(axis=1) > 0]

            self.experiments[experiment_type].append(curr_exp)

    def merge(self):
        merged_df = pd.DataFrame()

        for experiment_type in self.experiments.keys():
            tmp_experiments = pd.concat(self.experiments[experiment_type], ignore_index=True)
            if len(merged_df) != 0:
                merged_df = merge_dataframes(merged_df, tmp_experiments)
            else:
                merged_df = tmp_experiments

        self.success_cols = list(merged_df.filter(regex="Success$"))
        self.runtime_cols = list(merged_df.filter(like="Runtime"))
        merged_df.fillna('irrelevant', inplace=True)

        for runtime_col in self.runtime_cols:
            merged_df[runtime_col] = merged_df[runtime_col].apply(
                lambda x: self.config['max_runtime'] if x == 'irrelevant' else x)
        for success_col in self.success_cols:
            merged_df[success_col] = merged_df[success_col].apply(lambda x: 0 if x == 'irrelevant' else x)

        self.df = merged_df

    def add_target_variable(self):
        self.df = self.df.rename(columns=self.alg_runtime_mapping)
        self.df = self.df.rename(columns=self.alg_success_mapping)
        # self.runtime_cols = list(self.alg_runtime_mapping.values())
        # self.success_cols = list(self.alg_success_mapping.values())
        self.success_cols = list(self.df.filter(regex="Success$"))
        self.runtime_cols = list(self.df.filter(like="Runtime"))

        Y = self.df[self.runtime_cols].idxmin(axis=1)

        self.df['Y'] = Y
        self.df['Y Success'] = self.df.apply(lambda x: x[Preprocess.runtime_to_success(x['Y'])], axis=1)
        self.df['Y Runtime'] = self.df.apply(lambda x: x[x['Y']], axis=1)

    def add_computed_features(self):
        # self.df['GridSize'] = self.df['GridRows'] * self.df['GridColumns']
        # self.df['Sparsity'] = self.df.apply(lambda x: x['NumOfAgents'] / (x['GridSize'] - x['NumOfObstacles']), axis=1)
        self.df['maptype'] = self.df.apply(lambda x: map_type(x['GridName']), axis=1)

    def remove_redundant_columns(self):
        relev_cols = self.config['features'] + self.config['cat_features'] + self.runtime_cols + \
                     self.success_cols + ['GridSize', 'InstanceId', 'problem_type', 'Y', 'Y Runtime', 'Y Success']
        self.df = self.df[list(set(relev_cols))]

    def to_csv(self, filename):
        self.df.to_csv(filename, index=False)


def usage_example():
    mapf_experiemnt = Experiment()
    mapf_experiemnt.load()
    mapf_experiemnt.merge()
    mapf_experiemnt.add_target_variable()
    mapf_experiemnt.add_computed_features()
    mapf_experiemnt.remove_redundant_columns()
    mapf_experiemnt.to_csv('../data/from-vpn/experiments/customAndMovingAI/MovingAIAndCustomData-labelled.csv')


# usage_example()
