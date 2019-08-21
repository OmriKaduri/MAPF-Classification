

from sklearn.preprocessing import OneHotEncoder


import pandas as pd
from sklearn.utils import shuffle


# This class assist with cleaning the data and arrange him.

def data(dataSet):
    df = pd.read_csv(dataSet)
    features_cols = ['NumOfAgents', 'NumOfObstacles', 'BranchingFactor', 'ObstacleDensity',
            'AvgDistanceToGoal', 'MaxDistanceToGoal', 'MinDistanceToGoal', 'AvgStartDistances', 'AvgGoalDistances',
            'PointsAtSPRatio']

    X = df[features_cols]
    Y = df['Y']
    Y = Y.replace (['Basic-CBS/(A*/SIC)+ID Runtime',
            'Basic-CBS/(A*/SIC) choosing cardinal conflicts using lookahead Runtime',
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime',
            'Basic-CBS/(A*+OD/SIC) choosing the first conflict in CBS nodes Runtime',
            'MA-CBS-Local-10/(single:A*/SIC multi:A*+OD/SIC) choosing the first conflict in CBS nodes Runtime',
            'MA-CBS-Global-10/(A*+OD/SIC) choosing the first conflict in CBS nodes Runtime','A*+OD+ID Runtime',
            'MA-CBS-Local-10/(single:A*/SIC multi:EPEA*/SIC) choosing the first conflict in CBS nodes Runtime'] , [1,2,3,4,5,6,7,8])

   # print( Y.value_counts())
    return X , Y


get_data = {'data': data}

