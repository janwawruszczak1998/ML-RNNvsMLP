import pandas as pd
import numpy as np

from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model


def create_model(nmb_of_features, nmb_of_labels, optimizer='SGD', loss='categorical_crossentropy'):
    model = Sequential()
    model.add(Embedding(1000, 32))
    model.add(
        LSTM(64, dropout=0.3, recurrent_dropout=0.3, recurrent_initializer='glorot_uniform', return_sequences=True))
    model.add(LSTM(32, dropout=0.3, recurrent_dropout=0.3, recurrent_initializer='glorot_uniform'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(nmb_of_labels, activation='softmax'))
    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    model.summary()

    return model


def fit_rnn_model(X, labels, optimizer='adam', loss='categorical_crossentropy', epochs=40, batch_size=64):
    model = create_model(optimizer, loss)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, stratify=labels)
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), batch_size=batch_size,
                        epochs=epochs)

    model.summary()

    return model


def evaluate_rnn_model_params(X: pd.DataFrame, labels: pd.DataFrame):
    nmb_of_features = X.shape[1]
    nmb_of_labels = len(set(labels))

    model = KerasClassifier(build_fn=create_model, nmb_of_features=nmb_of_features, nmb_of_labels=nmb_of_labels)

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

    print(pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False))


def predict_single_instance(model, instance):
    prediction = model.predict(np.array([instance]))

    return [float(prc) for prc in prediction[0]]