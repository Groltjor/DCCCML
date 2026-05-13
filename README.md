# Credit Default Case Study

Proyecto de práctica de machine learning para predecir default de clientes de tarjeta de crédito usando el dataset `default_of_credit_card_clients`.

Este repo busca generar Notebooks documentando el proceso de EDA, construcción de un modelo base dummy, moverse hacia Feature Engineering, construcción de pipelines de modelado, dividir etrenamiento y evaluación.

El objetivo es mantener un proyecto de ML documentado, legible y reproducible para el equipo evitando una entrega única en un notebook.

## Estructura

```text
DefaultUCICredit/
├── Data/
│   └── default_of_credit_card_clients.xls
├── Notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_credit_default_case_study.ipynb
│   └── archive/
│       └── first_book.ipynb
├── src/
│   └── credit_default/
│       ├── data.py
│       ├── features.py
│       ├── evaluation.py
│       └── modeling/
│           ├── pipelines.py
│           └── train.py
├── models/
├── reports/
└── requirements.txt
```

## /src/credit_default/modeling/train.py

La versión actual de /modeling/train entrena un modelo XGBoost usando un pipeline de `scikit-learn` y entregando en joblib, esto esta bajo revisión, recomiendo realizar entrenamientos desde el Notebook 03 o integrando nuevos pipelines de entrenamiento en 'src/credit_default/evaluation.py'. Es una vieja implementación a si que de momento es incompatible con los Notebooks cuyas son las versiones finales.

Actualmente trabajando en la evaluación de modelos.

## La mejor forma de usar el proyecto.

En este momento, podrás explorar el pipeline de entrenamiento hasta almacenamiento. Esto se encuentra /Notebooks.

## Estado del Proyecto

Esta es una versión 0 del proyecto. Todavía está en proceso de limpieza y mejora.
