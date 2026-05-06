# Credit Default Case Study

Proyecto de práctica de machine learning para predecir default de clientes de tarjeta de crédito usando el dataset `default_of_credit_card_clients`.

La idea de esta versión inicial es ordenar un notebook exploratorio en una estructura más limpia para GitHub, separando carga de datos, limpieza, features, pipelines de modelado y evaluación.

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

El threshold final usado por ahora es:

```text
0.30
```

Este threshold se eligió como punto inicial para balancear la detección de defaults contra el costo de falsos positivos.

## Cómo correr el entrenamiento

Desde la carpeta `src`:

```bash
python3.11 -m credit_default.modeling.train
```

Esto genera:

- Modelo entrenado en `models/`
- Threshold elegido en `models/threshold.json`
- Métricas finales en `reports/tables/final_model_metrics.json`
- Reporte de thresholds en `reports/tables/threshold_report.csv`

## Estado del Proyecto

Esta es una versión 0 del proyecto. Todavía está en proceso de limpieza y mejora.

Pendientes posibles:

- Documentar mejor las decisiones del notebook original.
- Migrar más feature engineering desde el notebook exploratorio.
- Agregar un README más completo con resultados y conclusiones.
- Limpiar dependencias no usadas en `requirements.txt`.
- Agregar `.gitignore`.
- Convertir el proyecto en paquete instalable para evitar depender de rutas locales.

## Nota

Este proyecto es principalmente de práctica y aprendizaje. Las decisiones de negocio, costos de errores y thresholds son ilustrativos.
