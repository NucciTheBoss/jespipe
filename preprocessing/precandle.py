import numpy as np


def preproc_candlesticks(dataset, time_interval):
    """Trend extraction technique: Candlesticks"""
    # Features for training
    X = np.array(dataset)[:, :-1]

    # Labels
    y = np.array(dataset)[:, -1]

    new_X = np.zeros((int(X.shape[0] / time_interval), int(X.shape[1] * 4)))
    new_y = np.zeros((int(y.shape[0] / time_interval),))
    for feature_ind in range(X.shape[1]):
        new_feature_ind = feature_ind * 4
        new_row_ind = 0
        for row_ind in range(0, X.shape[0] - time_interval, time_interval):
            # Find the 'open' value
            open_value = X[row_ind, feature_ind]
            new_X[new_row_ind, new_feature_ind] = open_value

            # Find the 'close' value
            end_ind = int(row_ind + (time_interval-1))
            close_value = X[end_ind, feature_ind]
            new_X[new_row_ind, new_feature_ind + 1] = close_value

            # Find the 'high' value
            high_value = np.max(X[row_ind:end_ind, feature_ind])
            new_X[new_row_ind, new_feature_ind + 2] = high_value

            # Find the 'low' value
            low_value = np.min(X[row_ind:end_ind, feature_ind])
            new_X[new_row_ind, new_feature_ind + 3] = low_value

            # Save the label -- to change probably
            new_y[new_row_ind] = y[row_ind]

            # Update row index for new_X
            new_row_ind += 1
    
    print(new_X.shape, new_y.shape)

    return new_X, new_y