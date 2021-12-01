import pandas as pd

def create_selected_features(dataset, cols):
    selected_features = pd.read_csv('datasets/%s.csv' % (dataset),
                                    usecols=cols)
    df = pd.DataFrame(selected_features)
    df.to_csv('datasets/%s_selected.csv' % (dataset))