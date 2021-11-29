import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest


def chi2test(X, y):

    chi2vals, pvals = chi2(X, y)
    valuable_features = 0
    alpha = 0.05
    for pval in pvals:
        if pval <= alpha:
            valuable_features = valuable_features + 1

    
    chosen_features = SelectKBest(chi2, k=valuable_features).fit(X, y).get_support()
    print("Valuable features: ", valuable_features, "of all: ", X.shape[1])

    print(chosen_features)
    return chosen_features