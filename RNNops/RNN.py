import pandas as pd
import numpy as np
import os
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import to_categorical


class RNN_model:
    """Recurrent neural network model."""
    def __init__(self, X_train, y_train, X_test, y_test, feature_count, file_name,
                 sequence_length, learning_rate, model=None):
        """Initialize RNN."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.feature_count = feature_count
        self.file_name = file_name

        if model:
            self.model = load_model(model)
        else:
            self.model = self.build_model(self.file_name)

    def build_model(self, fn):
        """Build the RNN model."""
        model = Sequential()
        for x in range(5):
            model.add(LSTM(self.feature_count,
                           input_shape=self.X_train.shape[1:],
                           return_sequences=True))
            model.add(Dropout(0.1))
        model.add(LSTM(self.y_train.shape[1], return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(units=self.y_train.shape[1]))
        model.add(Activation('softmax'))
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])

        # Write model summary to class
        with open(fn+'-model-summary.txt','w') as fh:
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        # Return the created model
        return model


def load_data(data, seq_len, feature_count, fn):
    """Loading/splitting the data into the training and testing data."""
    result = np.zeros((len(data) - seq_len, seq_len, feature_count+1))
    # Sequence lengths remain together
    # (i.e, 6 consecutive candles stay together at all times if seq_len=6)
    for index in range(len(data) - seq_len):
        result[index] = data[index: index + seq_len]

    np.random.seed(2020)  # Shuffling with for reproducable results
    np.random.shuffle(result)  # In-place shuffling for saving space

    row = len(result) * 0.85  # Amount of data to train on
    train = result[:int(row), :]
    x_train = train[:, :, :-1]
    y_train = to_categorical(train[:, -1][:, -1])  # one-hot encoding
    x_test = result[int(row):, :, :-1]
    y_test = to_categorical(result[int(row):, -1][:, -1])

    x_train = np.reshape(x_train,
                         (x_train.shape[0], x_train.shape[1], feature_count))
    x_test = np.reshape(x_test,
                        (x_test.shape[0], x_test.shape[1], feature_count))
    
    # Saving the parameters
    parameters = {'dataset': fn,
                  'feature_count': feature_count,
                  'sequence_length': seq_len,
                  'x_train shape': str(x_train.shape), 
                  'x_test shape': str(x_test.shape),
                  'y_train shape': str(y_train.shape),
                  'y_test': str(y_test.shape)}
    save_params = pd.DataFrame(parameters, index=[0])
    if os.path.isfile('loaded-data-shapes.csv'):
        header_value = False
    else:
        header_value = True
    save_params.to_csv('loaded-data-shapes.csv', mode='a', header=header_value)
    
    return [x_train, y_train, x_test, y_test]


def evaluate_model(y_test, predictions):
    """Evaluate the accuracy on the test data."""
    pred_argmax = np.argmax(predictions, axis=1)
    y_test_argmax = np.argmax(y_test, axis=1)
    accuracy = np.sum(pred_argmax == y_test_argmax) / len(y_test)
    return accuracy


def __helper_unravel(data):
    """Converting from 3D to 2D for sklearn models."""
    new_data = np.zeros((data.shape[0], data.shape[1]*data.shape[2]))
    for i in range(data.shape[0]):
        new_data[i] = np.ndarray.flatten(data[i])
    return new_data


# TODO: Change global values to be passed in as a dictionary!
def execute_RNN(X_train, y_train, X_test, y_test, fn, globals):
    """Training and testing the RNN model."""
    rnn_model = RNN_model(X_train, y_train, X_test, y_test, X_train.shape[1], fn)
    rnn_model.model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        verbose=VERBOSE)

    # Evaluate the model
    predictions = rnn_model.model.predict(X_test)
    accuracy = evaluate_model(y_test, predictions)
    print("Accuracy on benign test samples: {}%".format(accuracy * 100))
    
    # Saving the parameters
    parameters = {'dataset': fn,
                  'batch_size': BATCH_SIZE, 
                  'epochs': EPOCHS,
                  'validation_split': VALIDATION_SPLIT,
                  'learning_rate': LEARNING_RATE,
                  'sequence_length': SEQUENCE_LENGTH,
                  'accuracy': accuracy}
    save_params = pd.DataFrame(parameters, index=[0])
    if os.path.isfile('model-parameters.csv'):
        header_value = False
    else:
        header_value = True
    save_params.to_csv('model-parameters.csv', mode='a', header=header_value)
    rnn_model.model.save(fn+'-model.h5')