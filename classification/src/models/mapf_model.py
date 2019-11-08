import numpy as np
from abc import abstractmethod


class MapfModel:
    def __init__(self, X_train, y_train, X_test, y_test,
                 runtime_cols, max_runtime, features_cols):
        self.runtime_cols = runtime_cols
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_runtime = max_runtime
        self.features_cols = features_cols
        self.alg_runtime_cols = self.runtime_cols.copy()
        self.only_alg_runtime_cols = runtime_cols.copy()
        self.only_alg_runtime_cols.remove('Y Runtime')
        self.train_samples_weight = X_train.apply(lambda x:
                                                  np.log10(np.std(x[self.only_alg_runtime_cols].values)), axis=1)

    conversions = {
        0: 'EPEA*+ID Runtime',
        1: 'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
        2: 'ICTS 3E +ID Runtime',
        3: 'A*+OD+ID Runtime',
        4: 'Basic-CBS/(A*/SIC)+ID Runtime',
        5: 'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime'
    }

    @abstractmethod
    def print_results(self, results_file='model-results.csv'):
        pass
