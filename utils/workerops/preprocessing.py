import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA


def preprocessing(dataset_path, manip_type, params):
    """Preprocess data with specified data manipulation before model training.
    
    Keyword arguments:
    dataset_path -- path to manipulated data set.
    manip_type -- type of data manipulation to use.
    params -- user specified parameters to pass to manipulation."""
    dataset = pd.read_csv(dataset_path, header=None)

    if manip_type == "xgboost":
        feat, label = _preproc_xgb(dataset, int(params["n_features"]))
        return feat, label

    elif manip_type == "randomforest":
        feat, label = _preproc_randomforest(dataset)
        return feat, label

    elif manip_type == "pca":
        feat, label = _preproc_pca(dataset, int(params["n_features"]))
        return feat, label

    elif manip_type == "candlestick":
        feat, label = _preproc_candlestick(dataset, int(params["time_interval"]))
        return feat, label

    elif manip_type is None:
        feat, label = _preproc_none(dataset)
        return feat, label

    else:
        raise ValueError("Invalid data manipulation type {}. Please specify valid data manipulation.".format(manip_type))


def _preproc_xgb(dataset, n_features):
    """Feature selection technique: XGBoost
    
    Keyword arguments:
    dataset -- the pandas DataFrame to perform XGBoost feature selection on."""
    # Features for training
    features = np.array(dataset)[:, :-1]

    # Labels
    labels = np.array(dataset)[:, -1]

    feature_train, feature_test, labels_train, labels_test = train_test_split(features, labels)

     # Training with best gamma
    regressor = xgb.XGBRegressor(
        n_estimators=100,
        gamma=1.5,
        max_depth=n_features
    )

    regressor.fit(feature_train, labels_train)
    # IMPORTANT: These blocks will most likely be crucial for future logging
    # labels_pred = regressor.predict(feature_test)
    # mse = mean_squared_error(labels_test, labels_pred)
    # print('XGBoost mean squared error: ', mse)

    feature_importance = regressor.feature_importances_
    # print('Feature importance: ', feature_importance)

    features_to_select = feature_importance.argsort()[-n_features:][::-1]
    # print('Features to select: ', features_to_select)

    new_features = features[:, features_to_select]

    return new_features, labels


def _preproc_randomforest(dataset):
    """Feature selection technique: Random Forest
    
    Keyword arguments:
    dataset -- the pandas DataFrame to perform Random Forest feature selection on."""
    # Features for training
    features = np.array(dataset)[:, :-1]

    # Labels
    labels = np.array(dataset)[:, -1]

    sel = SelectFromModel(RandomForestRegressor(n_estimators=100))
    sel.fit(features, labels.astype('int'))

    features_to_select = sel.get_support(indices=True)
    # print('Features to select: ', features_to_select)

    new_features = features[:, features_to_select]

    return new_features, labels


def _preproc_pca(dataset, n_features):
    """Dimensionality reduction technique: Principle Compoenent Analysis
    
    Keyword arguments:
    dataset -- the pandas DataFrame to perform Principle Component Analysis dimensionality reduction on.
    n_features -- """
    # Features for training
    features = np.array(dataset)[:, :-1]

    # Labels
    labels = np.array(dataset)[:, -1]

    new_features = PCA(n_components=n_features).fit_transform(features)

    return new_features, labels


def _preproc_candlestick(dataset, time_interval):
    """Trend extraction technique: Candlesticks
    
    Keyword arguments:
    dataset -- the pandas DataFrame to perform candlestick trend extraction on.
    time_interval -- """
    # Features for model training
    features = np.array(dataset)[:, :-1]

    # Labels
    labels = np.array(dataset)[:, -1]

    new_features = np.zeros((int(features.shape[0] / time_interval), int(features.shape[1] * 4)))
    new_labels = np.zeros((int(labels.shape[0] / time_interval),))
    for feature_ind in range(features.shape[1]):
        new_feature_ind = feature_ind * 4
        new_row_ind = 0
        for row_ind in range(0, features.shape[0] - time_interval, time_interval):
            # Find the 'open' value
            open_value = features[row_ind, feature_ind]
            new_features[new_row_ind, new_feature_ind] = open_value

            # Find the 'close' value
            end_ind = int(row_ind + (time_interval-1))
            close_value = features[end_ind, feature_ind]
            new_features[new_row_ind, new_feature_ind + 1] = close_value

            # Find the 'high' value
            high_value = np.max(features[row_ind:end_ind, feature_ind])
            new_features[new_row_ind, new_feature_ind + 2] = high_value

            # Find the 'low' value
            low_value = np.min(features[row_ind:end_ind, feature_ind])
            new_features[new_row_ind, new_feature_ind + 3] = low_value

            # Save the label -- to change probably
            new_labels[new_row_ind] = labels[row_ind]

            # Update row index for new_features
            new_row_ind += 1

    return new_features, new_labels


def _preproc_none(dataset):
    """Edge-case for if user does not desire to use any data manipulation technique.
    
    Keyword arguments:
    dataset -- pandas DataFrame to split into features and labels."""
    # Features for model training
    features = np.array(dataset)[:, :-1]

    # Labels
    labels = np.array(dataset)[:, -1]

    return features, labels
