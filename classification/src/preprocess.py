import pandas as pd


class Preprocess:
    def __init__(self, max_runtime, runtime_cols):
        self.max_runtime = max_runtime
        self.runtime_cols = runtime_cols
        self.features_cols = ['GridRows', 'GridColumns', 'NumOfObstacles',
                              'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal', 'AvgStartDistances',
                              'AvgGoalDistances',
                              'NumOfAgents', 'ObstacleDensity', 'PointsAtSPRatio', 'Sparsity',
                              'BranchingFactor']
        self.only_alg_runtime_cols = self.runtime_cols.copy()
        self.only_alg_runtime_cols.remove('Y Runtime')
        self.df = {}
        self.solved_df = {}

    @staticmethod
    def runtime_to_success(col):
        splitted = col.split()
        splitted[-1] = "Success"
        return " ".join(splitted)

    def fix_measurement_errors_on_maxtime(self):
        for runtime_col in self.runtime_cols:
            self.df[runtime_col] = self.df[runtime_col].where(self.df[runtime_col] < self.max_runtime, self.max_runtime)
            success_col = Preprocess.runtime_to_success(runtime_col)
            self.df[success_col] = self.df.apply(lambda x: self.df[runtime_col] < self.max_runtime, 0)

    @staticmethod
    def algorithm_statistics(df):
        print("Algorithm statistics for dataframe with", len(df), "mapf problems")
        count_df = df['Y'].value_counts().reset_index()
        count_df['BestPercentage'] = count_df.apply(lambda row: row['Y'] / len(df), axis=1)
        print(count_df)
        print("Choosing the best algorithm each time accuracy:", df['Y'].value_counts()[0] / len(df))

    def label_raw_results(self, csv_file):
        df = pd.read_csv(csv_file)
        relev_cols = self.only_alg_runtime_cols + self.features_cols + ['Y']
        Y = df[self.only_alg_runtime_cols].idxmin(axis=1)
        df['Y'] = Y
        df['Y Success'] = df.apply(lambda x: x[Preprocess.runtime_to_success(x['Y'])], axis=1)
        df['Y Runtime'] = df.apply(lambda x: x[x['Y']], axis=1)
        df['GridSize'] = df['GridRows'] * df['GridColumns']
        df['Sparsity'] = df.apply(lambda x: x['NumOfAgents'] / (x['GridSize'] - x['NumOfObstacles']), axis=1)
        self.features_cols.append('GridSize')
        self.features_cols.append('Sparsity')
        self.features_cols.append('Y')

        labelled_csv_file = csv_file.split('.csv')[0] + "-labelled.csv"
        df.to_csv(labelled_csv_file)

        return labelled_csv_file

    def load_labelled_results(self, csv_file):
        """
        load_raw_results read the csv given to it,
        fixing measurment errors (if existed, i.e. 300100 can't be a valid runtime),
        extracting the subset of at least partially solved problems and prints the statistics for the labels (algorithms)
        """
        self.df = pd.read_csv(csv_file)

        self.fix_measurement_errors_on_maxtime()
        self.solved_df = self.df[self.df['Y Runtime'] < self.max_runtime]  # Drop all rows no algorithm solved
        self.algorithm_statistics(self.df)
        self.algorithm_statistics(self.solved_df)

        return self.solved_df

    @staticmethod
    def balance_dataset_by_label(df):
        g = df.groupby('Y')
        balanced_df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))
        return balanced_df
