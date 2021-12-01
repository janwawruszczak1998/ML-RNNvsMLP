import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from features.Chi2Selector import Chi2Selector
from sklearn.compose import make_column_selector as selector

# datasets in csv
datasets = ['1_appendicitis', '2_vehicle', '3_texture', '4_spectfheart', '5_spambase', '6_optdigits', '7_coil2000',
            '8_musk', '9_semeion']
DISCRETE_DTYPES = {'int', 'int_', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
CONTINUOUS_DTYPES = {'float', 'float_', 'float16', 'float32', 'float64'}


def preprocess():
    for data_id, dataset in enumerate(datasets):
        df = pd.read_csv("datasets/%s.csv" % (dataset), delimiter=",", header=None)

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1].astype(int)

        disc_transformer = Pipeline(steps=[
            ('selection', Chi2Selector(alpha=0.05)),
            ('standarization', StandardScaler())
        ])

        cont_transformer = Pipeline(steps=[
            ('standarization', StandardScaler()),
            ('pca', PCA(n_components=0.9, svd_solver='full'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('discrete', disc_transformer, selector(dtype_include=DISCRETE_DTYPES)),
                ('continuous', cont_transformer, selector(dtype_include=CONTINUOUS_DTYPES))
            ]
        )

        X_preprocessed = preprocessor.fit_transform(X, y)
        data_preprocessed = np.hstack((X_preprocessed, y.to_numpy()[:, np.newaxis]))
        print("Old shape: {old}, new shape: {new}".format(old=df.shape, new=data_preprocessed.shape))
        np.save('datasets/{}_selected.npy'.format(dataset), data_preprocessed)  # save preprocessed data to .npy file
