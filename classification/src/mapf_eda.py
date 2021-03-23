import os
import pickle

import joblib
import matplotlib.pyplot as plt
import operator
from preprocess import Preprocess
from metrics import coverage_score
import numpy as np
import math
from pathlib import Path
import seaborn as sns
import pandas as pd
from matplotlib.colors import ListedColormap
from joblib import dump, load


class MapfEDA:

    def __init__(self, df, runtime_cols, plots_dir='plots'):
        self.models = {}
        self.df = df
        self.runtime_cols = runtime_cols
        self.only_alg_runtime_cols = self.runtime_cols.copy()
        self.only_alg_runtime_cols.remove('Y Runtime')
        self.plots_dir = plots_dir
        self.conversions = dict(zip(np.arange(len(self.only_alg_runtime_cols)), iter(self.only_alg_runtime_cols)))
        self.coverages = {}

    def create_runtime_histograms(self, histograms_filename='runtime_histograms.jpg'):

        row_num, col_num = 3, math.ceil(len(self.only_alg_runtime_cols) / 3)
        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True, figsize=(10, 5))
        index = 0
        for runtime_col in self.runtime_cols:
            if 'P Runtime' in runtime_col or 'Y Runtime' in runtime_col:
                continue
            if col_num > 1:
                self.df[runtime_col].hist(ax=axes[index % row_num][index // row_num])
                axes[index % row_num][index // row_num].set_title(runtime_col)
            else:
                self.df[runtime_col].hist(ax=axes[index % row_num])
                axes[index % row_num].set_title(runtime_col)
            index += 1
        fig.savefig(Path(self.plots_dir) / histograms_filename, format="jpg")

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
        ranked_df = MapfEDA.add_ranking_results(self.df, self.only_alg_runtime_cols)
        row_num, col_num = 3, math.ceil(len(self.only_alg_runtime_cols) / 3)

        fig, axes = plt.subplots(row_num, col_num, sharey=True, sharex=True, figsize=(10, 5))
        index = 0

        for alg in self.runtime_cols:
            if 'Y Runtime' in alg:
                continue
            if col_num > 1:
                ranked_df[alg + '-results'].hist(ax=axes[index % row_num][index // row_num])
                axes[index % row_num][index // row_num].set_title(alg)
            else:
                ranked_df[alg + '-results'].hist(ax=axes[index % row_num])
                axes[index % row_num].set_title(alg)
            # print(alg, "avg place", ranked_df[alg + '-results'].mean(), ranked_df[alg + '-results'].std())

            index += 1
        fig.savefig(Path(self.plots_dir) / histograms_filename, format="jpg")

    def create_stacked_rankings(self, X_test, sorted_runtime_cols=None, filename='stacked_bar_rankings.jpg'):
        sns.set()
        ranked_df = MapfEDA.add_ranking_results(X_test, self.only_alg_runtime_cols)
        alg_winnings = pd.DataFrame()
        if not sorted_runtime_cols:
            sorted_runtime_cols = self.only_alg_runtime_cols
        if 'Y Runtime' in sorted_runtime_cols:
            sorted_runtime_cols.remove('Y Runtime')
        for alg in sorted_runtime_cols:
            print(alg)
            if 'Y Runtime' in alg:
                alg_rankgs = dict(
                    zip(np.arange(1, len(self.only_alg_runtime_cols) + 1), np.zeros(len(self.only_alg_runtime_cols))))
                alg_rankgs.update({'Solver': 'Oracle'})
            else:
                alg_rankgs = dict(ranked_df[alg + '-results'].value_counts())
                alg_rankgs.update({'Solver': MapfEDA.model_name_for_plot(alg)})
            alg_winnings = alg_winnings.append(alg_rankgs, ignore_index=True)

        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = colors[1:]
        alg_winnings = alg_winnings.set_index('Solver').T
        ax = alg_winnings.plot(kind='bar', stacked='True',
                               color=colors,
                               # colormap=ListedColormap(sns.color_palette("deep", len(self.only_alg_runtime_cols))),
                               legend=False)

        # linestyle = MapfEDA.line_style_for_model(runtime_col)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], [MapfEDA.model_name_for_plot(alg) for alg in sorted_runtime_cols[::-1]],
                  bbox_to_anchor=(1.05, 1), loc='upper left'
                  )
        fig = ax.get_figure()
        ax.set_ylabel('Problems')
        fig.savefig(Path(self.plots_dir) / filename, format="jpg", bbox_inches="tight")

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
        fig.savefig(Path(self.plots_dir) / filename, format="jpg")

    @staticmethod
    def folder_from_label(label):
        return {
            'epea Runtime': 'epea',
            'macbs Runtime': 'ma-cbs',
            'icts Runtime': 'icts',
            'astar Runtime': 'astar',
            'cbs Runtime': 'basic-cbs',
            'cbsh Runtime': 'cbs-h',
            'cbsh-c Runtime': 'cbsh-c',
            'sat Runtime': 'sat',
            'lazycbs Runtime': 'lazy-cbs',
            'Y Runtime': 'Oracle',
            'P Runtime': 'ML Model',
            'P-Reg Runtime': 'XGBoost Regression',
            'P-Clf Runtime': 'XGBoost Classification',
            'P-Cov Runtime': 'XGBoost Coverage',
            'P-Cost Runtime': 'XGBoost Cost-Sensitive',
            'P-CNNReg Runtime': 'CNN Regression',
            'P-CNNClf Runtime': 'CNN Classification',
            'P-Best-at-maptype': 'Best-at-maptype Baseline',
            'P-Best-at-grid': 'Best-at-grid Baseline',
            'Random': 'Random'
        }[label]

    @staticmethod
    def model_name_for_plot(label):
        return {
            'epea Runtime': 'EPEA*',
            'macbs Runtime': 'MA-CBS',
            'icts Runtime': 'ICTS',
            'astar Runtime': 'A*',
            'cbs Runtime': 'CBS',
            'cbsh Runtime': 'CBS-H',
            'cbsh-c Runtime': 'CBS-H',
            'sat Runtime': 'SAT',
            'lazycbs Runtime': 'LAZY-CBS',
            'Y Runtime': 'Oracle',
            'P Runtime': 'ML Model',
            'P-Reg Runtime': 'XGBoost Regression',
            'P-Clf Runtime': 'XGBoost Classification',
            'P-Cov Runtime': 'XGBoost Coverage',
            'P-Cost Runtime': 'XGBoost Cost-Sensitive',
            'P-CNNReg Runtime': 'CNN Regression',
            'P-CNNClf Runtime': 'CNN Classification',
            'P-Best-at-maptype': 'Best-at-maptype Baseline',
            'P-Best-at-grid': 'Best-at-grid Baseline',
            'Random': 'Random'
        }[label]

    @staticmethod
    def line_style_for_model(model):
        return {
            'epea Runtime': 'solid',  # densely dotted
            'macbs Runtime': 'solid',
            'icts Runtime': 'solid',  # loosely dotted
            'astar Runtime': 'solid',
            'cbs Runtime': 'solid',
            'cbsh Runtime': 'solid',
            'cbsh-c Runtime': 'solid',
            'sat Runtime': 'dashed',
            'lazycbs Runtime': 'dashed',
            'Random': 'dashed',
            'Y Runtime': 'solid',
            'P Runtime': 'solid',
            'P-Reg Runtime': 'solid',
            'P-Clf Runtime': 'dashed',
            'P-Cov Runtime': 'dashed',
            'P-Cost Runtime': 'dashed',
            'P-CNNClf Runtime': 'dashed',
            'P-CNNReg Runtime': 'solid',
            'P-Best-at-maptype': 'dashed',
            'P-Best-at-grid': 'dashed',
        }[model]

    @staticmethod
    def line_color_for_model(model):
        return {
            'epea Runtime': 'blue',  # densely dotted
            'macbs Runtime': 'red',
            'icts Runtime': 'green',  # loosely dotted
            'astar Runtime': 'gray',
            'cbs Runtime': 'gray',
            'cbsh Runtime': 'orange',
            'cbsh-c Runtime': 'orange',
            'sat': 'pink',
            'lazycbs Runtime': 'red',
            'Y Runtime': 'black',
            'P Runtime': 'solid',
            'P-Reg Runtime': '#E30022',
            'P-Clf Runtime': '#E30022',
            'P-Cov Runtime': 'purple',
            'P-Cost Runtime': 'orange',
            'P-CNNClf Runtime': 'purple',
            'P-CNNReg Runtime': 'purple',
        }[model]

    def accumulate_cactus_data(self, X_test, exp_type, dir='plots/cactus', fold_number=0, max_time=300000,
                               step=5000, load=False):
        self.coverages[fold_number] = {'Random': {}}
        cactus_dir = Path(dir) / exp_type
        cactus_dir.mkdir(parents=True, exist_ok=True)
        cactus_data_path = str(cactus_dir / str(fold_number))
        exclude_cols = []
        if load and os.path.exists(cactus_data_path):
            self.coverages[fold_number] = joblib.load(cactus_data_path)
            exclude_cols = list(self.coverages[fold_number].keys())

        random_preds = [self.conversions[x] for x in
                        np.random.randint(0, len(self.only_alg_runtime_cols), size=(len(X_test)))]
        cols = self.runtime_cols.copy()

        for col_name in self.models:
            if col_name not in self.runtime_cols:
                cols.append(col_name)
        cols = list(set(cols) - set(exclude_cols))
        if len(cols) == 0:
            return
        for i in range(1, max_time + 2, step):
            i = min(max_time, i)
            for runtime_col in cols:
                if runtime_col not in self.runtime_cols:  # It's a model results
                    elem = coverage_score(X_test, self.models[runtime_col], i)
                else:
                    elem = coverage_score(X_test, [runtime_col] * len(X_test), i)
                if runtime_col in self.coverages[fold_number]:
                    self.coverages[fold_number][runtime_col][i / 1000] = elem
                else:
                    self.coverages[fold_number][runtime_col] = {}
                    self.coverages[fold_number][runtime_col][i / 1000] = elem

            self.coverages[fold_number]['Random'][i / 1000] = coverage_score(X_test, random_preds, i)
        joblib.dump(self.coverages[fold_number], cactus_data_path)

    def plot_cactus_graph(self, exp_name):
        exp_name = exp_name + '/cactus.jpg'
        print("Plotting cactus graph to:", exp_name)
        # y-axis = coverage, x-axis=time, draw line for each algorithm/model/oracle

        # Compute Average and std for each column over all folds
        coverages_df = pd.DataFrame.from_dict({(i, j): self.coverages[i][j]
                                               for i in self.coverages.keys()
                                               for j in self.coverages[i].keys()},
                                              orient='index')
        mean_dict = coverages_df.groupby(level=1).mean().T.to_dict()
        std_dict = coverages_df.groupby(level=1).std().fillna(0).T.to_dict()

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.set()
        sorted_runtime_cols = list(mean_dict.keys())
        sorted_runtime_cols = sorted(sorted_runtime_cols, key=lambda x: list(mean_dict[x].values())[-1],
                                     reverse=True)
        for runtime_col in sorted_runtime_cols:
            lists = sorted(mean_dict[runtime_col].items())
            std_list = sorted(std_dict[runtime_col].items())
            _, s_y = zip(*std_list)
            x, y = zip(*lists)
            lower_std = np.array(y) - np.array(s_y)
            upper_std = np.array(y) + np.array(s_y)
            ax.plot(x, y, linestyle=MapfEDA.line_style_for_model(runtime_col),
                    # color=MapfEDA.line_color_for_model(runtime_col)
                    )
            ax.fill_between(x, lower_std, upper_std, alpha=0.35)

        ax.legend([MapfEDA.model_name_for_plot(r) for r in sorted_runtime_cols], prop={'size': 16})
        ax.set_ylabel('Coverage')
        ax.set_xlabel('Time (seconds)')
        fig.savefig(Path(self.plots_dir) / 'cactus' / exp_name, format="jpg")
        return sorted_runtime_cols

    def plot_coverage_box_plot(self, results, filename='cov_boxplot.jpg'):

        results = results[
            results.Model.str.contains(
                '|'.join(['Classification', 'Regression', 'Coverage', 'Best-at-grid', 'Random']))]
        results.Model = results.Model.str.replace("XGBoost", "")
        results.Model = results.Model.str.replace('baseline', "")
        axes = results.boxplot(column='Coverage', by='Model', return_type='axes')[0]
        # axes.set_title('Coverage box-plot')
        # plt.tight_layout()
        axes.get_figure().savefig(Path(self.plots_dir) / filename, format="jpg")

    @staticmethod
    def save_obj(obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_obj(name):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def add_model_results(self, preds, col_name):
        self.models[col_name] = preds
