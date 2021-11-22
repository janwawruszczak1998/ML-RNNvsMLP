import pandas as pd
import numpy as np


import models.mlp

from models.mlp  import evaluate_mlp_model_params, fit_mlp_model


# datasets in csv
datasets = []



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

for data_id, dataset in enumerate(datasets):
    df = pd.read_csv("datasets/%s.csv" % (dataset), delimiter=",")
    df = df.dropna()
    df = df.reset_index(drop=True)

    X = df[:, :-1]
    y = df[:, -1].astype(int)

    venerable_features = chi2test(X, y) # names of venerable features
    create_selected_features(dataset, venerable_features) # save file .csv with only venerable features


    # Evaluate params <- we are here
    # evaluate_mlp_model_params(X, y) # <- prints sorted list of params by quality
    # Create
    # mlp_model = fit_mlp_model(X_tfidf_feat, df['label'])
    # Save -- after training
    # mlp_model.save('models/mlp_model.h5')
    # Load -- after saving
    # mlp_model = load_model('models/mlp_model.h5')



# remember to use venv!
# /usr/bin/python3 -m pip install --upgrade pi
# pip install numpy
# pip install tensorflow
# pip install sklearn
# pip install pandas