import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

df = pd.read_csv("C:/Users/Cruz/Downloads/cuaderno_datos/data/clientes_abandono_mayo_2026.csv")

print("="*60)
print("BLOQUE III - EJERCICIO INTEGRADOR")
print("="*60)

print("\n1. DISTRIBUCION DE CLASES")
print("-"*60)
print("No abandono (0):", (df["abandono"]==0).sum())
print("Abandono (1):", (df["abandono"]==1).sum())
print("Ratio desbalance:", round((df["abandono"]==0).sum() / (df["abandono"]==1).sum(), 1), ":1")

print("\n2. PREPARACION DE DATOS")
print("-"*60)
target = "abandono"
features_num = ["edad", "ingresos", "compras_12m", "visitas_web", "reclamaciones", "antiguedad_meses", "ticket_medio"]
features_cat = ["segmento"]
X = df[features_num + features_cat]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, features_num),
    ("cat", categorical_transformer, features_cat)
])

print("\n3. COMPARACION DE MODELOS")
print("-"*60)

def evaluar(nombre, modelo):
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    proba = modelo.predict_proba(X_test)[:, 1]
    return {
        "modelo": nombre,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba)
    }, pred, proba

logit = Pipeline(steps=[("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
tree = Pipeline(steps=[("preprocessor", preprocessor), ("model", DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced"))])
rf = Pipeline(steps=[("preprocessor", preprocessor), ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"))])

res_logit, pred_logit, proba_logit = evaluar("Logistic Regression", logit)
res_tree, pred_tree, proba_tree = evaluar("Decision Tree", tree)
res_rf, pred_rf, proba_rf = evaluar("Random Forest", rf)

resultados = pd.DataFrame([res_logit, res_tree, res_rf]).sort_values("recall", ascending=False)
print(resultados.to_string())

print("\n" + "="*60)
print("4. EJERCICIO: AJUSTE DE UMBRAL")
print("="*60)
print("\n*** Escenario: Empresa quiere detectar el MAXIMO numero de abandonos ***\n")

proba = proba_logit
for umbral in [0.3, 0.4, 0.5]:
    pred_umbral = (proba >= umbral).astype(int)
    fp = ((pred_umbral==1) & (y_test==0)).sum()
    fn = ((pred_umbral==0) & (y_test==1)).sum()
    print(f"Umbral {umbral}: Precision={precision_score(y_test, pred_umbral, zero_division=0):.2f} | Recall={recall_score(y_test, pred_umbral):.2f} | FP={fp} FN={fn}")

umbral = 0.3
pred_opt = (proba >= umbral).astype(int)

print("\n" + "="*60)
print("5. MATRIZ DE CONFUSION ( umbral =", umbral, ")")
print("="*60)
cm = confusion_matrix(y_test, pred_opt)
print(f"                  Predicho")
print(f"               No Abandono  Abandono")
print(f"Real No Abandono:   {cm[0,0]:4d}       {cm[0,1]:4d}")
print(f"Real Abandono:         {cm[1,0]:4d}       {cm[1,1]:4d}")
print(f"\nTP={cm[1,1]} TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]}")

print("\n" + "-"*60)
print("CLASSIFICATION REPORT")
print("-"*60)
print(classification_report(y_test, pred_opt, target_names=["No Abandono", "Abandono"]))

print("="*60)
print("CONCLUSIONES EJECUTIVAS")
print("="*60)
print("1. Mejor modelo por recall:", resultados.iloc[0]["modelo"], "- Recall:", round(resultados.iloc[0]["recall"], 2))
print("2. Umbral seleccionado: 0.3 (mayor recall)")
print("3. Error mas costoso: FALSO NEGATIVO (cliente perdido = ingresos perdidos)")
print("4. Recomendacion: Usar modelo con recall alto, umbral bajo para capturar mas abandonos")