import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def preproc_randomforest(dataset):
    """Feature selection technique: Random forest"""
    # Features for training
    X = np.array(dataset)[:, :-1]

    # Labels
    y = np.array(dataset)[:, -1]

    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X, y.astype('int'))

    features_to_select = sel.get_support(indices=True)
    print('Features to select: ', features_to_select)

    new_X = X[:, features_to_select]
    print(X.shape, new_X.shape)

    return new_X, y
