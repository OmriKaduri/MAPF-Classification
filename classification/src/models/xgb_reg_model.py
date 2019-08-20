from src.models.mapf_model import MapfModel
from src.preprocess import Preprocess
import numpy as np
from sklearn.metrics import accuracy_score
from src.metrics import coverage_score, cumsum_score
from sklearn.compose import TransformedTargetRegressor
import xgboost as xgb
import csv


class XGBRegModel(MapfModel):

    def __init__(self, *args):
        super(XGBRegModel, self).__init__(*args)
        self.xg_regs = []
        self.trained = False
        self.balanced = False
        self.xg_test_res = []
        self.xg_train_res = []
        self.modelname = 'Regression based classification'

    def balance_dataset(self):
        self.balanced = True
        self.modelname += ' - balanced'

    @staticmethod
    def func(x):
        return np.log1p(x)

    @staticmethod
    def inverse_func(x):
        return np.expm1(x)

    def train(self):
        if self.balanced:
            self.X_train = Preprocess.balance_dataset_by_label(self.X_train)
            self.y_train = self.X_train['Y']

        regs_data = []
        index = 0
        for runtime_col in self.runtime_cols:
            if 'P Runtime' in runtime_col or 'Y Runtime' in runtime_col:
                continue
            # print(index, ":", runtime_col)
            index += 1
            #     reg_X_train, reg_X_test, reg_y_train, reg_y_test = train_test_split(X_train[features_cols],
            #                         X_train[runtime_col], test_size=0.3)
            regs_data.append([self.X_train[self.features_cols], self.X_test[self.features_cols],
                              self.X_train[runtime_col], self.X_test[runtime_col]])
            xg_reg = xgb.XGBRegressor(objective='reg:linear', min_child_weight=0.5, colsample_bytree=0.8,
                                      learning_rate=0.1
                                      , max_depth=3, alpha=10,
                                      n_estimators=75)  # , colsample_bylevel= 0.5, colsample_bynode = 0.5)
            xg_reg_trans = TransformedTargetRegressor(regressor=xg_reg, func=XGBRegModel.func,
                                                      inverse_func=XGBRegModel.inverse_func)

            xg_reg_trans.fit(self.X_train[self.features_cols], self.X_train[runtime_col],
                             sample_weight=self.train_samples_weight)

            self.xg_regs.append(xg_reg_trans)
            self.trained = True

    def add_regression_as_features_to_data(self):
        features_with_reg_cols = self.features_cols.copy()
        for key, conversion in self.conversions.items():
            self.X_test['P ' + conversion] = self.xg_test_res[key]
            self.X_train['P ' + conversion] = self.xg_train_res[key]
            features_with_reg_cols.append('P ' + conversion)

        return self.X_train, self.X_test, features_with_reg_cols

    def print_results(self, results_file='model-results.csv'):
        if not self.trained:
            print("ERROR! Can't print model results before training")
            return

        for xg_reg in self.xg_regs:
            self.xg_test_res.append(np.array(xg_reg.predict(self.X_test[self.features_cols])))
            self.xg_train_res.append(np.array(xg_reg.predict(self.X_train[self.features_cols])))
        self.xg_test_res = np.array(self.xg_test_res)

        test_preds = [self.conversions[index] for index in self.xg_test_res.argmin(axis=0)]

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
                                 'Notes': 'This model is a super-model of 5 Regression models - one for each model'
                                          ' - and then argmin for each regression output gives the classification'})
