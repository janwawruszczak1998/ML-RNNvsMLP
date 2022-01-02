import numpy as np
from scipy.stats import shapiro, wilcoxon

MLP_scores = np.array([88.7, 81.0, 75.3, 82.8, 94.3, 98.0, 94.0, 98.7, 90.4])
RNN_scores = np.array([89.7, 65.0, 72.7, 84.3, 87.5, 97.3, 94.0, 94.3, 30.3])

alpha = 0.05
diff = MLP_scores - RNN_scores
stat, pval = shapiro(diff)
print(pval <= alpha)

stat, pval = wilcoxon(diff)
print(pval)
print(pval <= alpha)
