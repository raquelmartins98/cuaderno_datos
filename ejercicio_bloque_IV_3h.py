import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DATA_DIR = "C:/Users/Cruz/Downloads/cuaderno_datos/data"
df = pd.read_csv(f"{DATA_DIR}/segmentacion_clientes_mayo_2026.csv")

print("="*60)
print("BLOQUE IV - CLUSTERING (3h) - SOLUCION COMPLETA")
print("="*60)

features = ["ingresos", "compras_12m", "ticket_medio", "visitas_web", "dias_desde_ultima_compra", "reclamaciones"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n1. SELECCION DE K (SILHOUETTE)")
print("-"*60)

for k in [2, 3, 4, 5]:
    modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = modelo.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"k={k}: silhouette={sil:.3f}")

modelo_final = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = modelo_final.fit_predict(X_scaled)

print("\n2. PERFILES FINALES")
print("="*60)

perfil = df.groupby("cluster")[features].mean().round(1)
print(perfil)

print("\n" + "="*60)
print("3. NOMBRES DE NEGOCIO Y ACCIONES")
print("="*60)

nombres = {
    0: "CLIENTES INACTIVOS",
    1: "VIP/PREMIUM",
    2: "PROBLEMATICOS",
    3: "REGULARES"
}

acciones = {
    0: "Camana de reactivacion - oferta especial",
    1: "Programa VIP - beneficios exclusivos",
    2: "Atencion al cliente - resolver reclamaciones",
    3: "Fidelizacion - upselling"
}

for c in range(4):
    cluster_data = df[df["cluster"] == c]
    print(f"\nCluster {c}: {nombres[c]}")
    print(f"  N={len(cluster_data)}")
    print(f"  Caracteristicas:")
    print(f"    - Ingresos medios: {cluster_data['ingresos'].mean():.0f}")
    print(f"    - Compras 12m: {cluster_data['compras_12m'].mean():.1f}")
    print(f"    - Ticket medio: {cluster_data['ticket_medio'].mean():.0f}")
    print(f"    - Visitas web: {cluster_data['visitas_web'].mean():.1f}")
    print(f"    - Dias sin compra: {cluster_data['dias_desde_ultima_compra'].mean():.0f}")
    print(f"    - Reclamaciones: {cluster_data['reclamaciones'].mean():.1f}")
    print(f"  ACCION: {acciones[c]}")

print("\n" + "="*60)
print("4. TABLA RESUMEN")
print("="*60)

print("""
| Cluster | Nombre         | N    | Ingresos | Compras | Ticket | Accion             |
|--------|---------------|------|---------|--------|--------|------------------|
| 0      | INACTIVOS     | 177 | 24,221  | 3      | 40     | Reactivacion      |
| 1      | VIP/PREMIUM   | 73  | 66,196  | 20     | 182    | VIP             |
| 2      | PROBLEMATICOS | 56  | 66,007  | 19     | 182    | Atencion al cliente |
| 3      | REGULARES     | 144 | 43,155  | 11     | 94     | Fidelizacion     |
""")

print("\n5. PCA")
print("-"*60)

pca = PCA(n_components=2)
componentes = pca.fit_transform(X_scaled)
print(f"Varianza explicada: {pca.explained_variance_ratio_.sum():.1%}")

print("\n" + "="*60)
print("CONCLUSIONES")
print("="*60)
print("""
- 4 clusters con perfiles diferenciables
- Silhouette: 0.459 (aceptable)
- PCA explica 88.5% de varianza en 2D

Cada cluster tiene accion comercial especifica:
- Cluster 0 (Inactivos): Reactivacion urgente
- Cluster 1 (VIP): Beneficios premium
- Cluster 2 (Problematicos): Resolver quejas
- Cluster 3 (Regulares): Mejorar valor
""")