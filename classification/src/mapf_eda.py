import matplotlib.pyplot as plt
import operator
from src.preprocess import Preprocess


class MapfEDA:
    def __init__(self, df, runtime_cols):
        self.df = df
        self.runtime_cols = runtime_cols
        self.alg_runtime_cols = self.runtime_cols.copy()

    def create_runtime_histograms(self, histograms_filename='runtime_histograms.jpg'):
        fig, axes = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(10, 5))
        index = 0
        for runtime_col in self.runtime_cols:
            if 'P Runtime' in runtime_col or 'Y Runtime' in runtime_col:
                continue
            print(runtime_col)
            self.df[runtime_col].hist(ax=axes[index % 3][index % 2])
            axes[index % 3][index % 2].set_title(runtime_col)
            index += 1
        fig.savefig(histograms_filename, format="jpg")

    @staticmethod
    def places_for(row, alg, alg_runtime_cols):
        results = {}
        sorted_results = {}
        for col in alg_runtime_cols:
            if 'Y Runtime' in col:
                continue
            results[col] = row[col]

        results = sorted(results.items(), key=operator.itemgetter(1))
        for index, (curr_alg, result) in enumerate(results):
            if curr_alg == alg:
                return index + 1

        print("OH SHIT")

    @staticmethod
    def add_ranking_results(df, alg_runtime_cols):
        for alg in alg_runtime_cols:
            if 'Y Runtime' in alg:
                continue
            df[alg + '-results'] = df.apply(lambda x: MapfEDA.places_for(x, alg, alg_runtime_cols), axis=1)
        if 'P' in df:
            df['P Runtime-results'] = df.apply(lambda x: x[x['P'] + '-results'], axis=1)

        return df

    def create_rankings_histograms(self, histograms_filename='ranking_histograms.jpg'):
        ranked_df = MapfEDA.add_ranking_results(self.df, self.alg_runtime_cols)

        fig, axes = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(10, 5))
        index = 0

        for alg in self.runtime_cols:
            if 'Y Runtime' in alg:
                continue
            ranked_df[alg + '-results'].hist(ax=axes[index % 3][index % 2])
            axes[index % 3][index % 2].set_title(alg)
            print(alg, "avg place", ranked_df[alg + '-results'].mean(), ranked_df[alg + '-results'].std())

            index += 1
        fig.savefig(histograms_filename, format="jpg")

    def create_cumsum_histogram(self, df, predict_col='P', filename='cumsum_histogram.jpg'):
        predict_runtime_col = predict_col + ' Runtime'

        df[Preprocess.runtime_to_success(predict_runtime_col)] = df.apply(
            lambda x: x[Preprocess.runtime_to_success(x[predict_col])], axis=1)

        df[predict_runtime_col] = df.apply(lambda x: x[x[predict_col]], axis=1)

        runtime_per_algo = {}
        if predict_col not in df:
            print("ERROR - Didn't found predicted runtime at Dataframe.")
        cols = self.runtime_cols + ['P Runtime']
        for runtime in cols:
            substr_index = runtime.rfind(')')
            if (substr_index != -1):
                key = runtime[:substr_index + 1]
            else:
                key = runtime
            runtime_per_algo[key] = df[runtime].sum()
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.bar(*zip(*runtime_per_algo.items()))
        fig.savefig(filename, format="jpg")
