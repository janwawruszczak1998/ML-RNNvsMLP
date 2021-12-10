import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model


def create_model(nmb_of_features, nmb_of_labels, optimizer='adam', loss='categorical_crossentropy', dropout_rate=0.25):
    model = Sequential()
    model.add(Dense(nmb_of_features, activation='tanh'))
    model.add(Dense(512, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(nmb_of_labels, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])


    return model


def fit_mlp_model(X, y):
    nmb_of_features = X.shape[1]
    nmb_of_labels = len(set(y))

    model = create_model(nmb_of_features=nmb_of_features, nmb_of_labels=nmb_of_labels, optimizer='rmsprop', loss='categorical_crossentropy', dropout_rate=0.2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=32, epochs=80)

    model.summary()
    plot_model(model, to_file='mlp_model.png', show_shapes=True, show_layer_names=True)

    return model


def evaluate_mlp_model_params(X, y):
    nmb_of_features = X.shape[1]
    nmb_of_labels = len(set(y))

    model = KerasClassifier(build_fn=create_model, nmb_of_features=nmb_of_features, nmb_of_labels=nmb_of_labels)

    param_grid = {
        'epochs': [20, 40, 60, 80],
        'batch_size': [16, 32, 64, 128],
        'dropout_rate': [0.2, 0.25, 0.3],
        'optimizer': ['rmsprop', 'adam', 'SGD'],
        'loss': ['mse', 'categorical_crossentropy']
    }


    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        cv=StratifiedKFold(n_splits=5, random_state=1410, shuffle=True),
                        n_jobs=-1, return_train_score=True)
    grid_result = grid.fit(X, y)

    df = pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False)
    file = open("params_sorted_by_mean_mlp_model.txt", "a")
    file.write(df.to_string())
    file.write("\n")
    file.close()


def predict_single_instance(model, instance):
    prediction = model.predict(np.array([instance]))

    return [float(prc) for prc in prediction[0]]
