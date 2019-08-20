from sklearn.metrics import accuracy_score
from src.metrics import coverage_score, cumsum_score
from src.models.mapf_model import MapfModel
import xgboost as xgb
import csv
from src.preprocess import Preprocess


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
        self.xg_cls = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.01)
        self.xg_cls.fit(self.X_train[self.features_cols], self.y_train, sample_weight=self.train_samples_weight)
        self.trained = True

    def print_results(self, results_file='xgbmodel-results.csv'):
        if not self.trained:
            print("ERROR! Can't print model results before training")
            return
        test_preds = self.xg_cls.predict(self.X_test[self.features_cols])

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
