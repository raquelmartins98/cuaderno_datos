import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

DATA_DIR = "C:/Users/Cruz/Downloads/cuaderno_datos/data"
df = pd.read_csv(f"{DATA_DIR}/clientes_clasificacion.csv")

print("="*60)
print("BLOQUE III - CLASIFICACION ML")
print("="*60)

print("\n1. DATASET")
print("-"*60)
print(f"Shape: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nDistribucion target:")
print(df["abandono"].value_counts())
print(f"\nDesbalance: {df['abandono'].value_counts()[0] / df['abandono'].value_counts()[1]:.1f}:1")

X = df.drop(columns=["abandono"])
y = df["abandono"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=42)

print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

print("\n" + "="*60)
print("2. MODELO: LOGISTIC REGRESSION")
print("="*60)

logit = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
logit.fit(X_train, y_train)
pred_logit = logit.predict(X_test)

print(classification_report(y_test, pred_logit, digits=3))

print("\n" + "="*60)
print("3. MODELO: RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

print(classification_report(y_test, pred_rf, digits=3))

print("MATRIZ DE CONFUSION - Random Forest:")
print("-"*40)
cm = confusion_matrix(y_test, pred_rf)
print(f"                 Predicho")
print(f"              No aband  Abandono")
print(f"Real No aband:  {cm[0,0]:4d}     {cm[0,1]:4d}")
print(f"Real Abandono:    {cm[1,0]:4d}     {cm[1,1]:4d}")

print("\n" + "="*60)
print("4. COMPARACION DE MODELOS")
print("="*60)

def metricas(y_true, y_pred, modelo):
    return {
        "modelo": modelo,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }

resultados = pd.DataFrame([
    metricas(y_test, pred_logit, "Logistic Regression"),
    metricas(y_test, pred_rf, "Random Forest")
])
print(resultados.to_string(index=False))

print("\n" + "="*60)
print("5. CONCLUSIONES")
print("="*60)
print("""
- Logistic Regression: modelo lineal, interpretable, buen baseline
- Random Forest: mejor handling de desbalance con class_weight="balanced"
- Para dataset desbalanceado, Random Forest puede ser mejor opcion
- La matriz de confusion permite ver tipos de error

RECOMENDACIONES:
- Usar class_weight="balanced" para datasets desbalanceados
- Considerar precision vs recall segun caso de negocio
- F1 es buena metrica cuando hay desbalance
""")