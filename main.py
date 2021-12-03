import numpy as np
import pandas as pd
import models.mlp
from features.preprocessing import preprocess, datasets

from models.mlp import evaluate_mlp_model_params, fit_mlp_model
from models.rnn import evaluate_rnn_model_params, fit_rnn_model

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

if __name__ == "__main__":
    ### PREPROCESSING uncomment only if features need to be selected
    # preprocess()

    ### EVALUATION OF PARAMS <- we are here
    for data_id, dataset in enumerate(datasets):
        data = np.load("datasets/%s_selected.npy" % (dataset))
        X = data[:, :-1]
        y = data[:, -1].astype(int)
    
        ### Uncomment only when there is a need to do something about MLP model
        # Evaluate params
        # file = open("params_sorted_by_mean_mlp_model.txt", "a")
        # file.write("\ndatasets/%s_selected\n" % (dataset))
        # file.close()
        # evaluate_mlp_model_params(X, y)
        # Create
        # mlp_model = fit_mlp_model(X, y)
        # Save
        # mlp_model.save('models/mlp_%s_model.h5' % (dataset))
        # Load
        # mlp_model = load_model('models/mlp_model.h5')
    
        ### Uncomment only when there is a need to do something about RNN model
        # Evaluate params
        file = open("params_sorted_by_mean_rnn_model.txt", "a")
        file.write("datasets/%s_selected\n" % (dataset))
        file.close()
        evaluate_rnn_model_params(X, y)
        # Create
        # rnn_model = fit_rnn_model(X, y)
        # Save
        # rnn_model.save('models/rnn_%s_model.h5' % (dataset))
        # Load
        # rnn_model = load_model('models/rnn_%s_model.h5' % (dataset))

# remember to use venv!
# remebert to create dir datasets/ and fill it before dealing with code!
# /usr/bin/python3 -m pip install --upgrade pi
# pip install numpy
# pip install tensorflow
# pip install sklearn
# pip install pandas
