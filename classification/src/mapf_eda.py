import matplotlib.pyplot as plt
import operator
from src.preprocess import Preprocess
from src.metrics import coverage_score
import numpy as np
import seaborn as sns


class MapfEDA:

    def __init__(self, df, runtime_cols):
        self.df = df
        self.runtime_cols = runtime_cols
        self.alg_runtime_cols = self.runtime_cols.copy()
        self.conversions = {
            0: 'EPEA*+ID Runtime',
            1: 'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
            2: 'ICTS 3E +ID Runtime',
            3: 'A*+OD+ID Runtime',
            4: 'Basic-CBS/(A*/SIC)+ID Runtime',
            5: 'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime'
        }

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

    @staticmethod
    def folder_from_label(label):
        return {
            'EPEA*+ID Runtime': 'epea',
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime': 'ma-cbs',
            'ICTS 3E +ID Runtime': 'icts',
            'A*+OD+ID Runtime': 'astar',
            'Basic-CBS/(A*/SIC)+ID Runtime': 'basic-cbs',
            'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime': 'cbs-h',
            'Y Runtime': 'Oracle',
            'P Runtime': 'ML Model',
            'Random': 'Random'
        }[label]

    def plot_cactus_graph(self, X_test, test_preds, filename='cactus.jpg'):
        print("Plotting cactus graph to:",filename)
        # y-axis = coverage, x-axis=time, draw line for each algorithm/model/oracle
        coverages = {'Random': {}}
        random_preds = [self.conversions[x] for x in np.random.randint(0, 6, size=(len(X_test)))]
        fig, ax = plt.subplots(figsize=(10, 7))
        cols = self.runtime_cols + ['P Runtime']

        for i in range(1, 300000, 500):
            for runtime_col in cols:
                if 'P Runtime' in runtime_col:
                    elem = coverage_score(X_test, test_preds, i)
                else:
                    elem = coverage_score(X_test, [runtime_col] * len(X_test), i)
                if runtime_col in coverages:
                    coverages[runtime_col][i / 1000] = elem
                else:
                    coverages[runtime_col] = {}
                    coverages[runtime_col][i / 1000] = elem

            coverages['Random'][i / 1000] = coverage_score(X_test, random_preds, i)
        sorted_runtime_cols = sorted(cols)
        for runtime_col in sorted_runtime_cols:
            lists = sorted(coverages[runtime_col].items())
            x, y = zip(*lists)

            sns.lineplot(x, y)

        ax.legend([MapfEDA.folder_from_label(r) for r in sorted_runtime_cols])
        fig.savefig(filename, format="jpg")

        # plt.show()
