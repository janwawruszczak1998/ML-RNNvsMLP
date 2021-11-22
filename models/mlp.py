import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.utils import plot_model


def create_model(nmb_of_features, optimizer='adam', loss='categorical_crossentropy', dropout_rate=0.25):
    model = Sequential()
    model.add(Dense(nmb_of_features, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])

    return model


def fit_mlp_model(X_tfidf_feat, labels, nmb_of_features=4, optimizer='adam', loss='categorical_crossentropy', epochs=20,
                  batch_size=16, dropout_rate=0.25):
    model = create_model(nmb_of_features, optimizer, loss, dropout_rate)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf_feat, labels, test_size=0.2, stratify=labels)
    y_train_cat = to_categorical(y_train, 2)
    y_test_cat = to_categorical(y_test, 2)
    history = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), batch_size=batch_size,
                        epochs=epochs)

    model.summary()
    plot_model(model, to_file='mlp_model.png', show_shapes=True, show_layer_names=True)

    return model


def evaluate_mlp_model_params(X_tfidf_feat: pd.DataFrame, labels: pd.DataFrame):
    nmb_of_features = X_tfidf_feat.shape[1]

    model = KerasClassifier(build_fn=create_model, nmb_of_features=nmb_of_features)

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
    grid_result = grid.fit(X_tfidf_feat, labels)

    print(pd.DataFrame(grid_result.cv_results_).sort_values('mean_test_score', ascending=False))


def predict_single_instance(model, instance):
    prediction = model.predict(np.array([instance]))

    return [float(prc) for prc in prediction[0]]
