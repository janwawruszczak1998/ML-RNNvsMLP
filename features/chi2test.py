import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest


def chi2test(X, y):

    chi2vals, pvals = chi2(X, y)
    venerable_features = 0
    alpha = 0.05
    for pval in pvals:
        if pval <= alpha:
            venerable_features = venerable_features + 1

    
    chosen_features = SelectKBest(chi2, k=venerable_features).fit(X_chi2, y_chi2).get_support()
    print("Venerable features: " + venerable_features)
    print(chosen_features)
    return chosen_features