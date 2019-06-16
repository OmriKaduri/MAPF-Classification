import time
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import  SVC
from DataSetsGetter import get_data
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import csv
from sklearn.model_selection import GridSearchCV

import numpy as np

# This class is the main class

csv_file = 'results.csv'

# DataSets
Data_Sets = {'data': 'data\\10-100grid2-6agents0-20obstacle_ratio.csv'}



# This function is the main function
def crossvalidation():
    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['dataset', 'method' ,'accuracy', 'AUC', 'Precision', 'Recall', 'train_time', 'classify_runtime']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data_set in Data_Sets.keys():
            # get the data
            X,Y = get_data[data_set](Data_Sets[data_set])
            # base learn classification.
            main(X , Y , data_set)


def main(X, Y, dataName):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    # cross validation 10 fold
    svm_clf = SVC(gamma='auto')
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    ann_clf = MLPClassifier(solver='sgd', activation='tanh' ,alpha= 0.00001, learning_rate='constant' ,hidden_layer_sizes=(100,), random_state=1)
    rf_clf = RandomForestClassifier(bootstrap = 'true' , min_samples_split = 2 , min_samples_leaf=10 , max_depth=100 , max_features='auto' ,n_estimators=100)
    ada = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(min_samples_split=0.5, criterion='entropy', max_depth=2), n_estimators=100, algorithm='SAMME')

    # create clf
    candidates = [rf_clf, svm_clf, knn_clf, ann_clf ,ada ]
    candidates_names = ['rf_clf', 'svm', 'ann', 'knn', 'ada', 'rf']

    for i in range(len(candidates)):
        with open(csv_file, 'a', newline='') as csvfile:
            fieldnames = ['dataset', 'method', 'accuracy', 'AUC', 'Precision', 'Recall', 'train_time',
                          'classify_runtime']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            clf = candidates[i]

            fold = StratifiedKFold(n_splits=5)
            acc = 0
            auc = 0
            train_time = 0
            pred_time = 0
            precision = 0
            recall = 0
            start_time = time.time()
            for train_index, test_index in fold.split(X, Y):

                xtrain, xtest = X[train_index], X[test_index]
                ytrain, ytest = Y[train_index], Y[test_index]
                # fit
                clf.fit(xtrain, ytrain)
                train_time += time.time() - start_time
                start_time = time.time()
                # predict
                prediction = clf.predict(xtest)
                pred_time += time.time() - start_time
                # calculate auc and acc
                acc += metrics.accuracy_score(ytest, prediction)
                auc += roc_auc_score_multiclass(ytest, prediction, average='macro')
                precision += precision_score(ytest, prediction, average='macro')
                recall += recall_score(ytest , prediction, average='macro')

            train_time = train_time/5
            pred_time = pred_time/5
            acc = acc/5
            auc = auc/5
            recall = recall/5
            precision = precision/5
            writer.writerow({
                             'dataset': dataName, 'method': candidates_names[i],
                             'accuracy': str(acc), 'AUC': str(auc), 'Recall' : str(recall) , 'Precision' : str(precision),
                             'train_time': str(train_time), 'classify_runtime': str(pred_time)})


def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

      #creating a set of all the unique classes using the actual class list
      unique_class = set(actual_class)
      num_classes = len(unique_class)
      avg_auc = 0;
      roc_auc_dict = {}
      for per_class in unique_class:
        #creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
        avg_auc += roc_auc

      return  avg_auc/num_classes



def hyperParamtersTuning(X, Y, dataName):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(Y, pd.DataFrame):
        Y = Y.values
    # cross validation 10 fold
    # mlp = MLPClassifier(max_iter=100)
    # parameter_space = {
    #     'hidden_layer_sizes': [(50,), (100,) ,(5,2)],
    #     'activation': ['tanh', 'relu'],
    #     'solver': ['sgd', 'adam'],
    #     'alpha': [0.0001, 0.05 , 0.00001],
    #     'learning_rate': ['constant', 'adaptive'],
    # }
    # Number of trees in random forest
    n_estimators = [5, 20 , 50, 100, 200]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [100 , 50 ,20 , 10]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [10, 8, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    parameter_space = {'max_depth' : max_depth , 'max_features' : max_features}
    rf_clf = RandomForestClassifier(bootstrap = 'true' , min_samples_split = 2 , min_samples_leaf=10 , max_depth=100 , max_features='auto' ,n_estimators=100)
    clf = GridSearchCV(rf_clf, parameter_space, n_jobs=-1, cv=3)
    clf.fit(X,Y)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))



crossvalidation()
