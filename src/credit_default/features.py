from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

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

    payment_status_cols = [f'PAY_{i}' for i in range(1, 7)]
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