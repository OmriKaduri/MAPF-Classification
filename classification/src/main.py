from src.preprocess import Preprocess
from src.mapf_eda import MapfEDA
from src.models.baselines import Baselines
from src.models.xgb_reg_model import XGBRegModel
from src.models.xgb_clf_model import XGBClfModel
from sklearn.model_selection import train_test_split
import pandas as pd

features_cols = ['GridRows', 'GridColumns', 'NumOfObstacles',
                 'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal', 'AvgStartDistances', 'AvgGoalDistances',
                 'NumOfAgents', 'ObstacleDensity', 'PointsAtSPRatio', 'Sparsity',
                 'BranchingFactor']
# 'bridges']

max_runtime = 300000
runtime_cols = ['A*+OD+ID Runtime',
                'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
                'Basic-CBS/(A*/SIC)+ID Runtime',
                'ICTS 3E +ID Runtime',
                'EPEA*+ID Runtime',
                'CBS/(A*/SIC) + BP + PC without smart tie breaking using Dynamic Lazy Open List with Heuristic MVC of Cardinal Conflict Graph Heuristic Runtime',
                'Y Runtime']

preprocess = Preprocess(max_runtime, runtime_cols)

results_file = '../data/from-vpn/AllData.csv'
# results_file = 'data/nathan/nathan_AllData.csv'
labelled_results = preprocess.label_raw_results(results_file)

df = preprocess.load_labelled_results(labelled_results)


def train_all_models(df):
    mapf_eda = MapfEDA(df, runtime_cols)
    mapf_eda.create_runtime_histograms()
    mapf_eda.create_rankings_histograms()

    # X_train = pd.read_csv('../data/from-vpn/splitted/-X_train.csv')
    # X_test = pd.read_csv('../data/from-vpn/splitted/-X_test.csv')
    # y_train = X_train['Y']
    # y_test = X_test['Y']

    X_train, X_test, y_train, y_test = train_test_split(df, df['Y'], test_size=0.25,
                                                        # stratify=df['GridName'].values
                                                        )
    X_train.to_csv('../data/from-vpn/splitted/X_train.csv', index=False)
    X_test.to_csv('../data/from-vpn/splitted/X_test.csv', index=False)

    baselines = Baselines(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    baselines.print_results()

    # cnn_reg = CNNRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    # cnn_reg.prepare_data()
    # cnn_reg.build_model()
    # # cnn_reg.train() # Training the cnn model would take ~10 minutes on a decent GPU (tested on K80)
    # cnn_reg.load_weights()  # weights file can be transferred as an argument
    # cnn_reg.print_results()
    #
    # xgb_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    # xgb_reg.train_cv()
    # xgb_reg_prediction = xgb_reg.print_results()
    # X_train, X_test, features_with_reg_cols = xgb_reg.add_regression_as_features_to_data()

    # mapf_eda.create_cumsum_histogram(xgb_reg_prediction, filename='xgb_reg_cumsum.jpg')

    # xgb_balanced_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    # xgb_balanced_reg.balance_dataset()
    # xgb_balanced_reg.train_cv()
    #
    # xgb_balanced_reg.print_results()
    # X_train, X_test, features_with_reg_cols = xgb_reg.add_regression_as_features_to_data()
    #
    # xgb_balanced_clf = XGBClfModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    # xgb_balanced_clf.balance_dataset()

    # xgb_balanced_clf.train_cv()
    # xgb_balanced_clf.print_results()

    xgb_clf = XGBClfModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
    xgb_clf.train_cv()
    X_test, test_preds = xgb_clf.print_results()
    mapf_eda.create_cumsum_histogram(X_test)
    mapf_eda.plot_cactus_graph(X_test, test_preds, filename='all-cactus.jpg')


def train_predict_per_maptype(df):
    maptypes = df['maptype'].unique()
    mapf_eda = MapfEDA(df, runtime_cols)
    for maptype in maptypes:
        print("Training model for maptype:", maptype)
        map_df = df[df['maptype'] == maptype]

        X_train = pd.read_csv('../data/from-vpn/splitted/' + maptype + '-X_train.csv')
        X_test = pd.read_csv('../data/from-vpn/splitted/' + maptype + '-X_test.csv')
        y_train = X_train['Y']
        y_test = X_test['Y']

        # X_train, X_test, y_train, y_test = train_test_split(map_df, map_df['Y'], test_size=0.25,
        #                                                     # stratify=df['GridName'].values
        #                                                     )
        # X_train.to_csv('../data/from-vpn/splitted/'+maptype + '-X_train.csv', index=False)
        # X_test.to_csv('../data/from-vpn/splitted/'+maptype + '-X_test.csv',index=False)

        baselines = Baselines(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
        baselines.print_results(notes='on map type ' + maptype)
        model = XGBClfModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
        model.add_modelname_suffix(maptype)
        model.train_cv()
        clf_X_test, clf_test_preds = model.print_results()
        mapf_eda.plot_cactus_graph(clf_X_test, clf_test_preds , filename=maptype+'-clf-cactus.jpg')

        xgb_reg = XGBRegModel(X_train, y_train, X_test, y_test, runtime_cols, max_runtime, features_cols)
        model.add_modelname_suffix(maptype)
        xgb_reg.train_cv()
        reg_X_test, reg_test_preds = xgb_reg.print_results()
        mapf_eda.plot_cactus_graph(reg_X_test, reg_test_preds , filename=maptype+'-reg-cactus.jpg')


train_predict_per_maptype(df)
# train_all_models(df)
