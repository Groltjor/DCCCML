from sklearn.model_selection import GridSearchCV
from typing import Any
import pandas as pd

def best_params(
    model : Any,
    params : dict,
    x_train_data: pd.DataFrame | pd.Series | list,
    y_train_data: pd.DataFrame | pd.Series | list,
    scoring_mode : str = 'balanced_accuracy',
    folds : int = 5
    ) -> GridSearchCV:
    """
    """

    X = x_train_data.copy()
    y = y_train_data.copy()

    clf = GridSearchCV(
        model,
        params,
        cv = folds,
        scoring = scoring_mode,
        n_jobs = -1
        )
    clf.fit(X, y)


    return clf