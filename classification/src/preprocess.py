import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self, max_runtime, runtime_cols, features_cols, cat_features_cols, max_sub_runtime=-1):
        self.max_runtime = max_runtime
        self.runtime_cols = runtime_cols
        self.features_cols = features_cols
        self.cat_features_cols = cat_features_cols
        self.only_alg_runtime_cols = self.runtime_cols.copy()
        self.only_alg_runtime_cols.remove('Y Runtime')
        self.max_sub_runtime = max_sub_runtime
        self.df = pd.DataFrame()
        self.solved_df = {}

    @staticmethod
    def runtime_to_success(col):
        splitted = col.split()
        splitted[-1] = "Success"
        return " ".join(splitted)

    def fix_measurement_errors_on_maxtime(self):
        for runtime_col in self.runtime_cols:
            self.df.loc[self.df[runtime_col] > self.max_runtime, runtime_col] = self.max_runtime

            # self.df[runtime_col] = self.df[runtime_col].where(
            # self.df[runtime_col] < self.max_runtime, self.max_runtime)
            success_col = Preprocess.runtime_to_success(runtime_col)
            self.df[success_col] = 1
            self.df.loc[self.df[runtime_col] == self.max_runtime, success_col] = 0

    @staticmethod
    def algorithm_statistics(df):
        print("Algorithm statistics for dataframe with", len(df), "mapf problems")
        count_df = df['Y'].value_counts().reset_index()
        count_df['BestPercentage'] = count_df.apply(lambda row: row['Y'] / len(df), axis=1)
        print(count_df)
        print("Choosing the best algorithm each time accuracy:", df['Y'].value_counts()[0] / len(df))

    @staticmethod
    def map_type(x):
        cities = ['Berlin', 'Boston', 'Paris']
        games = ['brc', 'den', 'ht_', 'lak', 'lt_', 'orz', 'ost', 'woundedcoast']
        # TODO: Add warehouse support
        if any(city in x for city in cities):
            return 'city'
        elif any(game in x for game in games):
            return 'game'
        elif 'maze' in x:
            return 'maze'
        elif 'room' in x:
            return 'room'
        elif 'random' in x:
            return 'random'
        elif 'empty' in x:
            return 'empty'
        else:
            print("OH SHIT!")
            return 'other'

    def load_labelled_results(self, csv_file, drop_maps=None):
        """
        load_raw_results read the csv given to it,
        fixing measurment errors (if existed, i.e. 300100 can't be a valid runtime),
        extracting the subset of at least partially solved problems and prints the statistics for the labels (algorithms)
        """

        self.df = pd.read_csv(csv_file)

        if drop_maps is not None:
            for map in drop_maps:
                self.drop_map(map)

        algorithms_not_in_configuration = set(self.df['Y'].unique()) - set(self.runtime_cols)
        if algorithms_not_in_configuration != set():
            self.drop_algorithms(algorithms_not_in_configuration)

        self.fix_measurement_errors_on_maxtime()

        for cat_feature in self.cat_features_cols:
            feature_dummies = pd.get_dummies(self.df[cat_feature])
            feature_dummies = feature_dummies.add_prefix(cat_feature + '_')
            self.df = pd.concat([self.df, feature_dummies], axis=1)
            self.features_cols.extend(feature_dummies.columns.values)

        self.solved_df = self.df[self.df['Y Runtime'] < self.max_runtime]  # Drop all rows no algorithm solved
        self.algorithm_statistics(self.df)
        self.algorithm_statistics(self.solved_df)

        print("Preprocessing done on dataset with {n} mapf problems".format(n=len(self.solved_df)))

        return self.solved_df, self.features_cols

    @staticmethod
    def balance_dataset_by_label(df):
        g = df.groupby('Y')
        balanced_df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        return balanced_df

    def drop_map(self, maptype):
        self.df = self.df[~self.df.GridName.str.contains(maptype)]

    def drop_algorithms(self, algorithms_not_in_configuration):
        self.df = self.df.drop(list(algorithms_not_in_configuration), axis=1)
        self.only_alg_runtime_cols = [alg for alg in self.only_alg_runtime_cols if
                                      alg not in algorithms_not_in_configuration]
        Y = self.df[self.only_alg_runtime_cols].idxmin(axis=1)
        self.df['Y'] = Y
        self.df['Y Success'] = self.df.apply(lambda x: x[Preprocess.runtime_to_success(x['Y'])], axis=1)
        self.df['Y Runtime'] = self.df.apply(lambda x: x[x['Y']], axis=1)
