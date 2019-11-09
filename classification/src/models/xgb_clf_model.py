from sklearn.metrics import accuracy_score
from src.metrics import coverage_score, cumsum_score
from src.models.mapf_model import MapfModel
import xgboost as xgb
import csv
from src.preprocess import Preprocess
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats


class XGBClfModel(MapfModel):

    def __init__(self, *args):
        super(XGBClfModel, self).__init__(*args)
        self.xg_cls = {}
        self.trained = False
        self.balanced = False
        self.modelname = 'XGBoost Classification Model '

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    def train(self):
        if self.balanced:
            self.X_train = Preprocess.balance_dataset_by_label(self.X_train)
            self.y_train = self.X_train['Y']
        self.xg_cls = xgb.XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.01)
        self.xg_cls.fit(self.X_train[self.features_cols], self.y_train, sample_weight=self.train_samples_weight)
        self.trained = True

    def train_cv(self):
        if self.balanced:
            self.X_train = Preprocess.balance_dataset_by_label(self.X_train)
            self.y_train = self.X_train['Y']

        self.xg_cls = xgb.XGBClassifier(objective='multi:softmax')

        param_dist = {'n_estimators': stats.randint(100, 300),
                      'learning_rate': stats.uniform(0.01, 0.07),
                      'subsample': stats.uniform(0.3, 0.7),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': stats.uniform(0.5, 0.45),
                      'min_child_weight': [1, 2, 3],
                      "gamma": [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
                      "reg_alpha": [0, 0.5, 1],
                      }

        clf = RandomizedSearchCV(self.xg_cls,
                                 param_distributions=param_dist,
                                 cv=2,
                                 n_iter=1,
                                 scoring='accuracy',
                                 error_score=0,
                                 verbose=3,
                                 n_jobs=-1)
        clf.fit(self.X_train[self.features_cols], self.y_train, sample_weight=self.train_samples_weight)
        self.xg_cls = clf.best_estimator_
        self.trained = True

    def print_results(self, results_file='model-results.csv'):
        if not self.trained:
            print("ERROR! Can't print model results before training")
            return
        test_preds = self.xg_cls.predict(self.X_test[self.features_cols])
        self.X_test['P'] = test_preds

        model_acc = accuracy_score(self.y_test, test_preds)
        model_coverage = coverage_score(self.X_test, test_preds)
        model_cumsum = cumsum_score(self.X_test, test_preds)

        with open(results_file, 'a+', newline='') as csvfile:
            fieldnames = ['Model', 'Accuracy', 'Coverage', 'Cumsum(minutes)', 'Notes']
            res_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            res_writer.writerow({'Model': self.modelname,
                                 'Accuracy': "{0:.2%}".format(model_acc),
                                 'Coverage': "{0:.2%}".format(model_coverage),
                                 'Cumsum(minutes)': int(model_cumsum),
                                 'Notes': ''})

        return self.X_test, test_preds
