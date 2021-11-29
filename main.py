import pandas as pd
import numpy as np

from features import chi2test, features_creation
import models.mlp

from models.mlp  import evaluate_mlp_model_params, fit_mlp_model


# datasets in csv
datasets = ['1_appendicitis', '2_vehicle', '4_spectfheart', '5_spambase', '6_optdigits', '7_coil2000', '9_semeion']



pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

for data_id, dataset in enumerate(datasets):
    df = pd.read_csv("datasets/%s.csv" % (dataset), delimiter=",", dtype='float32')
    df = df.dropna()
    df = df.reset_index(drop=True)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    valuable_features_boolean = chi2test.chi2test(X, y) # True, if feature is valuable
    valuable_features = []
    for val in range(len(valuable_features_boolean)):
        if(valuable_features_boolean[val] == True):
            valuable_features.append(val)
    
    
    
    features_creation.create_selected_features(dataset, valuable_features) # save file .csv with only valueable features


    # Evaluate params <- we are here
    # evaluate_mlp_model_params(X, y) # <- prints sorted list of params by quality
    # Create
    # mlp_model = fit_mlp_model(X_tfidf_feat, df['label'])
    # Save -- after training
    # mlp_model.save('models/mlp_model.h5')
    # Load -- after saving
    # mlp_model = load_model('models/mlp_model.h5')



# remember to use venv!
# remebert to create dir datasets/ and fill it before dealing with code!
# /usr/bin/python3 -m pip install --upgrade pi
# pip install numpy
# pip install tensorflow
# pip install sklearn
# pip install pandas