import numpy as np
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, RepeatedKFold

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
dataframe = read_csv(url, header=None)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]

model = Ridge(alpha=1.0)

cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=42)

mae_scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
average_mae = -np.mean(mae_scores)

print(f'Average MAE: {average_mae:.3f}')
