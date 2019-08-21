from src.preprocess import Preprocess
from src.mapf_eda import MapfEDA
from src.models.baselines import baselines
from src.models.xgb_reg_model import XGBRegModel
from src.models.xgb_clf_model import XGBClfModel
from src.models.cnn_reg_model import CNNRegModel
<<<<<<< HEAD
from sklearn.model_selection import train_test_split
=======
import pandas as pd
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28

features_cols = ['GridSize', 'NumOfObstacles',
                 'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal', 'AvgStartDistances',
                 'AvgGoalDistances',
<<<<<<< HEAD
                 'NumOfAgents', 'ObstacleDensity', 'PointsAtSPRatio', 'Sparsity']
# 'bridges']
=======
                 'NumOfAgents', 'ObstacleDensity', 'PointsAtSPRatio', 'Sparsity',
                 'bridges']
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
max_runtime = 300000
runtime_cols = ['EPEA*+ID Runtime',
                'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
                'ICTS 3E +ID Runtime',
                'A*+OD+ID Runtime',
                'Basic-CBS/(A*/SIC)+ID Runtime',
                'Y Runtime']

preprocess = Preprocess(max_runtime, runtime_cols)
<<<<<<< HEAD
results_file = 'data/nathan/nathan_AllData.csv'
labelled_results = preprocess.label_raw_results(results_file)
=======
labelled_results = preprocess.label_raw_results('AllData.csv')
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
df = preprocess.load_labelled_results(labelled_results)

mapf_eda = MapfEDA(df, runtime_cols)
mapf_eda.create_runtime_histograms()
mapf_eda.create_rankings_histograms()

<<<<<<< HEAD
# X_train = pd.read_csv('data/splitted/train_features.csv')
# y_train = pd.read_csv('data/splitted/train_labels.csv')
# X_test = pd.read_csv('data/splitted/test_features.csv')
# y_test = pd.read_csv('data/splitted/test_labels.csv')


X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.25)

baselines = baselines(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
baselines.print_results()

# cnn_reg = CNNRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
# cnn_reg.prepare_data()
# cnn_reg.build_model()
# # cnn_reg.train() # Training the cnn model would take ~10 minutes on a decent GPU (tested on K80)
# cnn_reg.load_weights()  # weights file can be transferred as an argument
# cnn_reg.print_results()

xgb_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_reg.train_cv()
xgb_reg_prediction = xgb_reg.print_results()
X_train, X_test, features_with_reg_cols = xgb_reg.add_regression_as_features_to_data()

mapf_eda.create_cumsum_histogram(xgb_reg_prediction, filename='xgb_reg_cumsum.jpg')

xgb_balanced_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_balanced_reg.balance_dataset()
xgb_balanced_reg.train_cv()
=======
X_train = pd.read_csv('data/splitted/train_features.csv')
y_train = pd.read_csv('data/splitted/train_labels.csv')
X_test = pd.read_csv('data/splitted/test_features.csv')
y_test = pd.read_csv('data/splitted/test_labels.csv')
#
# baselines = baselines(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
# baselines.print_results()

cnn_reg = CNNRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
cnn_reg.prepare_data()
cnn_reg.build_model()
# cnn_reg.train() # Training the cnn model would take ~10 minutes on a decent GPU (tested on K80)
cnn_reg.load_weights()  # weights file can be transferred as an argument
cnn_reg.print_results()

xgb_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_reg.train()
xgb_reg.print_results()
X_train, X_test, features_with_reg_cols = xgb_reg.add_regression_as_features_to_data()

xgb_balanced_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_balanced_reg.balance_dataset()
xgb_balanced_reg.train()
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
xgb_balanced_reg.print_results()
X_train, X_test, features_with_reg_cols = xgb_reg.add_regression_as_features_to_data()

xgb_balanced_clf = XGBClfModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
<<<<<<< HEAD
xgb_balanced_clf.train_cv()
=======
xgb_balanced_clf.train()
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
xgb_balanced_clf.print_results()

xgb_clf = XGBClfModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
xgb_clf.balance_dataset()
<<<<<<< HEAD
xgb_clf.train_cv()
xgb_clf.print_results()

mapf_eda.create_cumsum_histogram(X_test)
=======
xgb_clf.train()
xgb_clf.print_results()
>>>>>>> 4d8f07cd55d17bda7fb9aa0442a39a9056248d28
