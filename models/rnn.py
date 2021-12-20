import pandas as pd
import numpy as np

from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow import convert_to_tensor

from tensorflow.keras.utils import plot_model


def create_model(X, nmb_of_labels, optimizer='SGD', loss='categorical_crossentropy'):

    model = Sequential()
    model.add(LSTM(64, input_shape=X.shape[1:], dropout=0.3, recurrent_dropout=0.3, recurrent_initializer='glorot_uniform', activation='tanh', return_sequences=True))
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3, recurrent_initializer='glorot_uniform', activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(nmb_of_labels, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def fit_rnn_model(X, y, optimizer='adam', loss='categorical_crossentropy', epochs=40, batch_size=64):
    model = create_model(optimizer, loss)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    y_train_cat = to_categorical(y_train, len(set(y)))
    y_test_cat = to_categorical(y_test, len(set(y)))
    history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), batch_size=batch_size,
                        epochs=epochs)

    model.summary()
    plot_model(model, to_file='rnn_model.png', show_shapes=True, show_layer_names=True)

    return model


def evaluate_rnn_model_params(X, labels):
    nmb_of_labels = len(set(labels))
    #X = X.reshape(X.shape[0], X.shape[1], 1)

    model = KerasClassifier(build_fn=create_model, X=X, nmb_of_labels=nmb_of_labels)

    param_grid = {
        'epochs': [20, 40, 60, 80],
        'batch_size': [16, 32, 64, 128],
        'optimizer': ['rmsprop', 'adam', 'SGD'],
        'loss': ['mse', 'categorical_crossentropy']
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=StratifiedKFold(n_splits=5, random_state=1410, shuffle=True),
                        n_jobs=-1, return_train_score=True)
    grid_result = grid.fit(X, labels)

    df = pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False)
    file = open("params_sorted_by_mean_rnn_model.txt", "a")
    file.write(df.to_string())
    file.write("\n")
    file.close()
    

def predict_single_instance(model, instance):
    prediction = model.predict(np.array([instance]))

    return [float(prc) for prc in prediction[0]]