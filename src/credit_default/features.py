from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

TARGET = 'default payment next month'

@dataclass
class FeatureGroups:
    categorical_cols : list[str]
    numeric_cols : list[str]
    bill_cols : list[str]
    pay_amount_cols : list[str]
    payment_status_cols : list[str]

def get_feature_groups() -> FeatureGroups:
    """
    Define column groups used across preprocessing and modeling.
    """

    payment_status_cols = [f'PAY_{i}' for i in range(0, 7)]
    payment_status_cols.remove('PAY_1') ## No existe en el dataset
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amount_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

    categorical_cols = [
        'SEX',
        'EDUCATION',
        'MARRIAGE_CLEAN',
        *payment_status_cols,
    ]

    numeric_cols = [
        'LIMIT_BAL',
        'AGE',
        *bill_cols,
        *pay_amount_cols,
    ]

    return FeatureGroups(
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        bill_cols=bill_cols,
        pay_amount_cols=pay_amount_cols,
        payment_status_cols=payment_status_cols,
    )

def add_credit_behaviour_features(df: pd.DataFrame) -> pd.DataFrame:

    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_amount_cols = [f'PAY_AMT{i}' for i in range(1, 7)]

    df = df.copy()

    df["BILL_AMT_mean"] = df[bill_cols].mean(axis=1)
    df["BILL_AMT_max"] = df[bill_cols].max(axis=1)
    df["BILL_AMT_std"] = df[bill_cols].std(axis=1)

    df["PAY_AMT_mean"] = df[pay_amount_cols].mean(axis=1)
    df["PAY_AMT_max"] = df[pay_amount_cols].max(axis=1)
    df["PAY_AMT_std"] = df[pay_amount_cols].std(axis=1)

    df['debt_to_limit'] = np.where(
        df['LIMIT_BAL'] > 0,
        df['BILL_AMT_mean'] / df['LIMIT_BAL'],
        0
    )

    df['payment_to_debt'] = np.where(
        df['BILL_AMT_mean'] > 0,
        df['PAY_AMT_mean'] / df['BILL_AMT_mean'],
        0
    )

    columnas_anadidas = ['BILL_AMT_mean', 'BILL_AMT_max', 'BILL_AMT_std', 
                          'PAY_AMT_mean', 'PAY_AMT_max', 'PAY_AMT_std', 'debt_to_limit', 
                          'payment_to_debt']

    return df, columnas_anadidas

def clean_credit_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic cleaning decisions based on the data audit.

    Decisions:
    - Remove records with EDUCATION == 0 because this category is undocumented.
    - Map MARRIAGE == 0 to category 3, interpreted as 'others'
    - DROP ID
    - Keep PAY_* such as -2, because they are frequent and may contain signal.
    """

    df = df.copy()

    df = df[df['EDUCATION']!= 0].copy()

    df['MARRIAGE_CLEAN'] = df['MARRIAGE'].replace({0: 3})

    df = df.drop(columns = ['ID', 'MARRIAGE'])

    return df

def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split cleaned dataframe into features and target.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    return X, y

def check_for_vif(df: pd.DataFrame, numeric_cols: list[str]):
    """ 
    """

    vif_df = df[numeric_cols].copy()

    X = vif_df.copy()
    X_const = add_constant(X)

    vif_data = pd.DataFrame()
    vif_data['variable']  = X_const.columns
    vif_data['VIF'] = [
        variance_inflation_factor(X_const.values, i)
        for i in range(X_const.shape[1])
    ]

    vif_data = (
        vif_data[vif_data['variable'] != 'constant']
        .sort_values('VIF', ascending = False)
    )

    return vif_data



    

