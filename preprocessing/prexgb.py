import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def preproc_xgboost(dataset, n_features):
    """Feature selection technique: XGBoost"""
    # Features for training
    X = np.array(dataset)[:, :-1]

    # Labels
    y = np.array(dataset)[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y)

     # Training with best gamma
    regressor = xgb.XGBRegressor(
        n_estimators=100,
        gamma=1.5,
        max_depth=n_features
    )

    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('XGBoost mean squared error: ', mse)

    feature_importance = regressor.feature_importances_
    print('Feature importance: ', feature_importance)

    features_to_select = feature_importance.argsort()[-n_features:][::-1]
    print('Features to select: ', features_to_select)

    new_X = X[:, features_to_select]
    print(X.shape, new_X.shape)

    return new_X, y
