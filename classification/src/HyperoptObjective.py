from hyperopt import STATUS_OK
from sklearn.compose import TransformedTargetRegressor
from metrics import coverage_score
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import xgboost as xgb
import numpy as np


def func(x):
    return x
    # return np.log1p(x)


def inverse_func(x):
    return x
    # return np.expm1(x)


class HyperoptObjective(object):
    def __init__(self, x_train, y_train, x_test, y_test, model, const_params, fit_params, features_cols,
                 regressor=False, binary=False):
        self.evaluated_count = 0
        self.X_train = x_train
        self.y_train = y_train
        self.X_test = x_test
        self.y_test = y_test
        self.model = model
        self.constant_params = const_params
        self.fit_params = fit_params
        self.features_cols = features_cols
        self.regressor = regressor
        self.binary = binary

    def __call__(self, hyper_params):
        if self.regressor:
            xg_reg = xgb.XGBRegressor(objective='reg:squarederror', **hyper_params)
            curr_model = TransformedTargetRegressor(regressor=xg_reg, func=func,
                                                    inverse_func=inverse_func)
        else:
            curr_model = self.model(**hyper_params, **self.constant_params)

        best = curr_model.fit(X=self.X_train[self.features_cols], y=self.y_train, **self.fit_params)

        self.evaluated_count += 1

        test_preds = best.predict(self.X_test[self.features_cols])

        metric = 0
        if self.regressor:
            metric = np.sqrt(mean_squared_error(self.y_test, test_preds))
        elif self.binary:
            metric = -accuracy_score(self.y_test, test_preds)
        else:
            metric = -coverage_score(self.X_test, test_preds)

        return {
            'loss': metric,
            'status': STATUS_OK,
            'model': best
        }
        # NOTE: The negative sign is due to that fact that we optimize for coverage,
        # therefore we want to minimize the negative coverage (approach to -1)
