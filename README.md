# Credit Default Case Study

Proyecto de práctica de machine learning para predecir default de clientes de tarjeta de crédito usando el dataset `default_of_credit_card_clients`.

La idea de esta versión inicial es ordenar un notebook exploratorio en una estructura limpia para GitHub, separando carga de datos, limpieza, features, pipelines de modelado y evaluación.

## Objetivo

Construir un modelo de clasificación binaria para estimar si un cliente caerá en default el siguiente mes.

El foco principal no es solo maximizar accuracy, sino revisar métricas más útiles para un problema de riesgo crediticio, especialmente:

- Recall de la clase default
- Precision
- F1 score
- ROC AUC
- PR AUC
- Matriz de confusión
- Evaluación por threshold

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

## Modelo Actual

La versión actual entrena un modelo XGBoost usando un pipeline de `scikit-learn`.
También esta versión lleva al usuario hasta el entrenamiento de pipelines y su almacenamiento en joblib.
Actualmente trabajando en la evaluación de modelos.

## La mejor forma de usar el proyecto.

En este momento, podrás explorar el pipeline de entrenamiento hasta almacenamiento. Esto se encuentra /Notebooks.

## Cómo correr el entrenamiento

Esta bajo implementación, de momento almacena un XGBoost.

## Estado del Proyecto

Esta es una versión 0 del proyecto. Todavía está en proceso de limpieza y mejora.

Pendientes posibles:

- Documentar mejor las decisiones del notebook original.
- Limpiar dependencias no usadas en `requirements.txt`. xd cambiar nombre
- Convertir el proyecto en paquete instalable para evitar depender de rutas locales.

## Nota

Este proyecto es principalmente de práctica y aprendizaje. Las decisiones de negocio, costos de errores y thresholds son ilustrativos.
