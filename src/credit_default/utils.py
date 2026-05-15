from __future__ import annotations

import joblib
import os
from pathlib import Path
import json
import pandas as pd


def save_model_joblib(ruta_almacenado: Path, modelo : any, model_name : str) -> tuple(str, list[any]):
    """
    Es para almacenar modelos en joblib
    """

    os.makedirs(ruta_almacenado, exist_ok = True)

    file_name = os.path.join(ruta_almacenado, f'{model_name}.joblib')

    joblib.dump(modelo, file_name)

    return

def get_models_and_routes(PROJECT_ROOT : Path | str )-> dict:

    print(f'Preparando modelos ...', PROJECT_ROOT)

    models_route = PROJECT_ROOT / 'models'
    model_list = []
    nombre_modelos = []

    for ruta in models_route.glob('**/*/*.joblib'):

        print(f'Leyendo la ruta {ruta} \n')
        model = joblib.load(ruta)

        model_list.append(model)
        nombre_mod = str(ruta).split('/')[-1:][0]
        nombre_modelos.append(nombre_mod)

        print(f'Modelo cargado {nombre_mod}. \n')
    

    print( '-'*5, 'Todos los modelos han sido cargados', '-'*5)
    return (model_list, nombre_modelos)

def load_checkpoint(checkpoint_PATH : Path) -> tuple[list[pd.DataFrame | pd.Series], dict[str, Any] ]:
    """
    Retorna un arreglo de frames y un json que contiene metadata, No tranforma
    parquets.
    """
    ## Nota, debemos cambiarla hacia utils

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
    
    print('-'*5, 'Todos los checkpoints cargados', '-'*5)

    return arreglo_frames, json_loaded

    