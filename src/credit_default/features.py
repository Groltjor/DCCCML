from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
import os
import json

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

def add_credit_behaviour_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:

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
        vif_data[vif_data['variable'] != 'const']
        .sort_values('VIF', ascending = False)
    )

    return vif_data


def split_pos_neg_features(df_preliminar: pd.DataFrame, numeric_cols: list[str]) -> tuple[list[str], list[str]]:
    """
    Receives a dataframe with final features to evaluare
    and checks on the numeric ones those whom are negative to prevent
    problems with np.log1p
    """

    df_work = df_preliminar[numeric_cols].copy()

    has_negative = df_work.lt(0).any()

    negative_cols = has_negative[has_negative].index.tolist()
    positive_cols = has_negative[~has_negative].index.tolist()

    return negative_cols, positive_cols

def skew_checker(df_preliminar: pd.DataFrame, candidates: list[str]) -> tuple[list[str], list[str]]:
    """
    Revisa que las columnas esten o no con skew bajo 2+ se marcan skewd, 
    nos retornara para auotmatizar la selección de skewed
    """
    df_work = df_preliminar[candidates].copy()

    is_skew = df_work.skew().gt(2)

    skewed_yes = is_skew[is_skew].index.tolist()
    skewed_no = is_skew[~is_skew].index.tolist()

    return skewed_no, skewed_yes

def save_checkpoint(
    componentes_test : list[pd.DataFrame],
    columnas_log1p : list[str],
    columnas_numericas : list[str],
    columnas_categoricas: list[str],
    columnas_yeo : list[str],
    nombre_checkpoint : str,
    ruta_de_proyecto : Path
) -> bool:
    """
    Esta es una función de alamcenamiento donde predefinidamente se alamcena dentro de los Checkpoins
    De momento no soporta validation
    """
    saving_df_route = ruta_de_proyecto / 'checkpoints' /nombre_checkpoint
    print('Creando ruta en ', saving_df_route)
    os.makedirs(saving_df_route, exist_ok = True)
    print('Ruta Creada')

    lista_almacenamiento = ['X_train', 'X_test', 'y_train', 'y_test']

    for i in range(0, len(lista_almacenamiento)):

        df = componentes_test[i]
        print('Almacenando ', lista_almacenamiento[i])
        parquet_name = os.path.join(saving_df_route, f'{lista_almacenamiento[i]}.parquet')

        if isinstance(df, pd.Series):
            df = df.to_frame(name = TARGET)
        df.to_parquet(parquet_name)

    save_route_file = os.path.join(saving_df_route, 'metadata.json')
    print('Almacenando el archivo en: ', save_route_file)

    contenido = {
        'nombre_df' : nombre_checkpoint,
        'columnas_log1p' : columnas_log1p,
        'columnas_numericas' : columnas_numericas,
        'columnas_categoricas' : columnas_categoricas,
        'columnas_yeo' : columnas_yeo,
    }

    with open(save_route_file, 'w') as file:
        json.dump(contenido, file, indent = 2)

    return

def load_checkpoint(checkpoint_PATH : Path) -> tuple( list[pd.DataFrame], json ):

    lista_almacenamiento = ['X_train', 'X_test', 'y_train', 'y_test']
    json_route = os.path.join(checkpoint_PATH, 'metadata.json')
    arreglo_frames = []
    
    with open(json_route, 'r') as jsonfile:
        json_loaded = json.load(jsonfile)
    
    for parquet in lista_almacenamiento:
        parquet_route = os.path.join(checkpoint_PATH, f'{parquet}.parquet')
        print(f'Cargando el archivo {parquet_route}')
        parquet_df = pd.read_parquet(parquet_route)

        if parquet in ['y_train', 'y_test']:
            print(f'Aplanando {parquet}')
            parquet_df = parquet_df.squeeze()

        arreglo_frames.append(parquet_df)

    return arreglo_frames, json_loaded