from numpy.lib.npyio import load
from tensorflow.keras.datasets.fashion_mnist import load_data
from tensorflow.keras.utils import to_categorical

from models.mlp import evaluate_mlp_model_params, fit_mlp_model
from models.rnn import evaluate_rnn_model_params, fit_rnn_model

import numpy as np
import pandas as pd


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_data()

    X_train = X_train/255.0


    #file = open("params_sorted_by_mean_mlp_model.txt", "a")
    #file.write("\ndatasets/mnist\n")
    #file.close()
    #evaluate_mlp_model_params(X_train, y_train)
    
    file = open("params_sorted_by_mean_rnn_model.txt", "a")
    file.write("datasets/mnist\n")
    file.close()
    evaluate_rnn_model_params(X_train, y_train)
