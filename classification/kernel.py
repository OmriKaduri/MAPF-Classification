import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import time
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../data/saftey_efficay_myopiaTrain.csv')
df_test = pd.read_csv('../data/saftey_efficay_myopiaTest.csv')

fignum = 1
#30451
TRAIN_RECORDS = 30451
TEST_RECORDS = 12452
VALUES_PREC_THRESHOLD = 0.6
VALUES_1_PREC_THRESHOLD = 0.5
df_train = df_train[:TRAIN_RECORDS]
df_test = df_test[:TEST_RECORDS]
#------------tries
ones = df_train[df_train.Class == 1]
zeroes = df_train[df_train.Class == 0]
# print(len(ones[ones.T_L_Laser_Type == 'EX500']) / len(df_train[df_train.T_L_Laser_Type == 'EX500']))
# print(len(ones[ones.T_L_Laser_Type == 'Alegreto']) / len(df_train[df_train.T_L_Laser_Type == 'Alegreto']))
# print(len(ones[ones.T_L_Laser_Type == 'LSX-NEW']) / len(df_train[df_train.T_L_Laser_Type == 'LSX-NEW']))
# print(len(ones[ones.T_L_Laser_Type == '41']) / len(df_train[df_train.T_L_Laser_Type == '41']))
# print(len(df_train[df_train.T_L_Laser_Type == '41']))
# print(pd.Series(df_train['T_L_Laser_Type'], name='A').unique())
ones_lasiks = ones[ones.T_L_Treatment_Type == 'Lasik']
zeroes_lasiks = zeroes[zeroes.T_L_Treatment_Type == 'Lasik']

# -----------------
dum_cols = ['D_L_Sex', 'D_L_Eye' , 'Pre_L_Contact_Lens', 'T_L_Laser_Type', 'T_L_Treatment_Type', 'T_L_Cust._Ablation',
        'T_L_Micro', 'T_L_Head', 'T_L_Therapeutic_Cont._L.', 'T_L_Epith._Rep.']
insignificant_cols = ['D_L_Dominant_Eye', 'T_L_Year']
for col in df_train.columns:
    filt_1 = df_train[df_train.Class == 1][col]
    filt_1 = filt_1.dropna()
    filt_all = df_train[col]
    filt_all = filt_all.dropna()
    if len(filt_all) < len(df_train) * (1 - VALUES_PREC_THRESHOLD):
        insignificant_cols.append(col)
        if col in dum_cols:
            dum_cols.remove(col)
    else:
        if len(filt_1) < len(df_train[df_train.Class == 1]) * (1 - VALUES_1_PREC_THRESHOLD):
            insignificant_cols.append(col)
            if col in dum_cols:
                dum_cols.remove(col)
df_train = df_train.drop(insignificant_cols, axis=1)
df_test = df_test.drop(insignificant_cols, axis=1)
train_dummies = []
for col in dum_cols:
    train_dummies.append(pd.get_dummies(df_train[col]))
test_dummies = []
for col in dum_cols:
    test_dummies.append(pd.get_dummies(df_test[col]))
all_train_dummies = pd.concat(train_dummies, axis=1)
all_test_dummies = pd.concat(test_dummies, axis=1)
df_train = pd.concat((df_train, all_train_dummies), axis=1)
df_train = df_train.drop(dum_cols, axis=1)
df_test = pd.concat((df_test, all_test_dummies), axis=1)
df_test = df_test.drop(dum_cols, axis=1)
def fillAvg(minval, avgval, maxval):
    if (math.isnan(avgval)) and (not math.isnan(minval)) and (not math.isnan(maxval)):
        return (minval + maxval) / 2
    return avgval
def fillMax(minval, avgval, maxval):
    if (not math.isnan(avgval)) and (not math.isnan(minval)) and (math.isnan(maxval)):
        return minval + (avgval - minval) * 2
    return maxval
def fillMin(minval, avgval, maxval):
    if (not math.isnan(avgval)) and (math.isnan(minval)) and (not math.isnan(maxval)):
        return maxval - (maxval - avgval) * 2
    return minval
def calc_PRE_L_K(df):
    if ('Pre_L_Average_K' not in insignificant_cols) and ('Pre_L_K_Minimum' not in insignificant_cols) and ('Pre_L_K_Maximum' not in insignificant_cols):
        df['Pre_L_Average_K_new'] = df.apply(lambda x: fillAvg(x['Pre_L_K_Minimum'], x['Pre_L_Average_K'], x['Pre_L_K_Maximum']), axis=1)
        df['Pre_L_K_Maximum_new'] = df.apply(lambda x: fillMax(x['Pre_L_K_Minimum'], x['Pre_L_Average_K'], x['Pre_L_K_Maximum']), axis=1)
        df['Pre_L_K_Minimum_new'] = df.apply(lambda x: fillMin(x['Pre_L_K_Minimum'], x['Pre_L_Average_K'], x['Pre_L_K_Maximum']), axis=1)
        df.drop(['Pre_L_Average_K', 'Pre_L_K_Minimum', 'Pre_L_K_Maximum'], axis=1)
        df.rename(index=str, columns={'Pre_L_Average_K_new': 'Pre_L_Average_K', 'Pre_L_K_Maximum_new': 'Pre_L_K_Maximum', 'Pre_L_K_Minimum_new': 'Pre_L_K_Minimum'})
#calc_PRE_L_K(df_train)
#calc_PRE_L_K(df_test)
df_train = df_train.apply(lambda x: x.fillna(x.mean()), axis=0)
df_test = df_test.apply(lambda x: x.fillna(x.mean()), axis=0)
X = df_train.copy()



X = X.drop(['Class'], axis=1)
X_test = df_test.copy()
for col in X_test.columns:
    if col not in X.columns:
        X_test = X_test.drop(col, axis=1)
for col in X.columns:
    if col not in X_test.columns:
        X = X.drop(col, axis=1)
X = X.values
Y = df_train['Class'].values
Y_test = df_test.copy()
X_test = X_test.values
Y_test = Y_test.values
#------------------------SVM

"""start = time.time()
clf = OneVsRestClassifier(SVC(kernel='poly',gamma='auto', probability=True, degree=3 , cache_size=7000))
clf.fit(X, Y)
end = time.time()
print ("Single SVC", end - start, clf.score(X,Y))
proba = clf.predict_proba(X_test)"""

"""n_estimators = 10
start = time.time()
clf = OneVsRestClassifier(BaggingClassifier(SVC(kernel='poly', gamma='auto',probability=True , degree=3), max_samples=1.0 / n_estimators, n_estimators=n_estimators))
clf.fit(X, Y)
end = time.time()
print ("Bagging SVC", end - start, clf.score(X,Y))
proba = clf.predict_proba(X_test)
df_dict = {}
df_dict['id'] = [i for i in range(1, len(proba) + 1)]
df_dict['class'] = [y for x, y in proba]
print(df_dict)
df = pd.DataFrame.from_dict(df_dict)
df.to_csv('results3.csv', index=False)"""
"""
start = time.time()
clf = RandomForestClassifier(min_samples_leaf=20)
clf.fit(X, Y)
end = time.time()
print ("Random Forest", end - start, clf.score(X_test,Y_test))
proba = clf.predict_proba(X_test)
df_dict = {}
df_dict['id'] = [i for i in range(1, len(proba) + 1)]
df_dict['class'] = [y for x, y in proba]
print(df_dict)
df = pd.DataFrame.from_dict(df_dict)
df.to_csv('results1.csv', index=False)"""