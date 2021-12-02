from sklearn.feature_selection import chi2, SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin


class Chi2Selector(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.valuable_features = 0

    def fit(self, X, y=None):
        chi2vals, pvals = chi2(X, y)
        for pval in pvals:
            if pval <= self.alpha:
                self.valuable_features = self.valuable_features + 1
        return self

    def transform(self, X, y=None):
        self.X_best = SelectKBest(chi2, k=self.valuable_features).fit_transform(X, y)
        return self.X_best

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
