import matplotlib.pyplot as plt
import operator
from src.preprocess import Preprocess
from src.metrics import coverage_score
import numpy as np
import seaborn as sns


class MapfEDA:

    def __init__(self, df, runtime_cols):
        self.models = {}
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

    def create_stacked_rankings(self, X_test, filename='stacked_bar_rankings.jpg'):
        ranked_df = MapfEDA.add_ranking_results(X_test, self.alg_runtime_cols)
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_subplot(1, 1, 1)
        index = 0
        X = [1, 2, 3, 4, 5, 6]
        vals = [0, 0, 0, 0, 0, 0]
        prevals = [0, 0, 0, 0, 0, 0]
        for alg in self.runtime_cols:
            if 'Y Runtime' in alg or 'P Runtime' in alg:
                continue
            prevals = [a + b for a, b in zip(vals, prevals)]
            vals = ranked_df[alg + '-results'].value_counts().to_dict()
            vals = [v[1] for v in sorted(vals.items())]
            print(vals)
            if index == 0:
                ax.bar(X, vals)
            else:
                ax.bar(X, vals, bottom=prevals)
            index += 1
        ax.legend([MapfEDA.folder_from_label(alg) for alg in self.alg_runtime_cols])
        fig.savefig(filename, format="jpg")

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
            'P-Reg Runtime': 'Regression Model',
            'P-Clf Runtime': 'Classification Model',
            'Random': 'Random'
        }[label]

    @staticmethod
    def line_style_for_model(model):
        return {
            'EPEA*+ID Runtime': (0, (1, 1)),  # densely dotted
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime': 'dashed',
            'ICTS 3E +ID Runtime': (0, (1, 10)),  # loosely dotted
            'A*+OD+ID Runtime': 'dotted',
            'Basic-CBS/(A*/SIC)+ID Runtime': 'dashdot',
            'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime': (
                0, (5, 10)),
            'Y Runtime': 'solid',
            'P Runtime': 'solid',
            'P-Reg Runtime': 'solid',
            'P-Clf Runtime': 'solid',
        }[model]

    def plot_cactus_graph(self, X_test, filename='cactus.jpg', max_time=300000, step=500):
        print("Plotting cactus graph to:", filename)
        # y-axis = coverage, x-axis=time, draw line for each algorithm/model/oracle
        coverages = {'Random': {}}
        random_preds = [self.conversions[x] for x in np.random.randint(0, 6, size=(len(X_test)))]
        fig, ax = plt.subplots(figsize=(10, 7))
        cols = self.runtime_cols.copy()
        for col_name in self.models:
            if col_name not in self.runtime_cols:
                cols.append(col_name)

        for i in range(1, max_time, step):
            for runtime_col in cols:
                if runtime_col not in self.runtime_cols:  # It's a model results
                    elem = coverage_score(X_test, self.models[runtime_col], i)
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

            ax.plot(x, y, linestyle=MapfEDA.line_style_for_model(runtime_col))

        ax.legend([MapfEDA.folder_from_label(r) for r in sorted_runtime_cols], prop={'size': 16})
        fig.savefig(filename, format="jpg")

        # plt.show()

    def add_model_results(self, preds, col_name):
        self.models[col_name] = preds
