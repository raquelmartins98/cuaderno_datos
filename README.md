# Data Analyst — Sesiones de mayo de 2026

Este paquete contiene notebooks Jupyter ampliados para sesiones de 3 horas.

## Estructura

- `notebooks/`: notebooks por bloque.
- `data/`: datasets sintéticos reproducibles para las prácticas.
- `docs/`: guía rápida y propuesta de uso docente.
- `requirements.txt`: librerías necesarias.
- `environment.yml`: entorno conda recomendado.

## Bloques incluidos

1. Python, pandas y análisis descriptivo.
2. Regresión y comparación de modelos.
3. Clasificación, matriz de confusión y ROC.
4. Clustering, silhouette y PCA.
5. Series temporales y forecasting.

## Uso recomendado

```bash
conda env create -f environment.yml
conda activate data-analyst-mayo-2026
jupyter lab
```

También puede usarse con:

```bash
pip install -r requirements.txt
jupyter lab
```

## Metodología

Cada notebook está diseñado para una sesión de 3 horas:
- explicación conceptual,
- demostración guiada,
- práctica individual,
- cierre con conclusiones.