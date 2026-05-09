from __future__ import annotations

import joblib
import os
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent


def save_model_joblib(ruta_almacenado: Path, modelo : any, model_name : str) -> tuple(str, list[any]):
    """
    Es para almacenar modelos en joblib
    """

    os.makedirs(ruta_almacenado, exist_ok = True)

    file_name = os.path.join(ruta_almacenado, f'{model_name}.joblib')

    joblib.dump(modelo, file_name)

def get_models_and_routes()-> dict:

    print(f'Preparando modelos ...', PROJECT_ROOT)

    models_route = PROJECT_ROOT / 'models'
    model_list = []
    nombre_modelos = []

    for ruta in models_route.glob('**/*/*.joblib'):

        print(f'Leyendo la ruta {ruta}')
        model = joblib.load(ruta)

        model_list.append(model)
        nombre_mod = str(ruta).split('/')[-1:][0]
        nombre_modelos.append(nombre_mod)

        print(f'Modelo cargado {nombre_mod}.')
    

    return (model_list, nombre_modelos)