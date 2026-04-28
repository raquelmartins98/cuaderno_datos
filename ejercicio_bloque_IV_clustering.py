import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

DATA_DIR = "C:/Users/Cruz/Downloads/cuaderno_datos/data"
df = pd.read_csv(f"{DATA_DIR}/clientes_clustering.csv")

print("="*60)
print("BLOQUE IV - CLUSTERING ML")
print("="*60)

print("\n1. DATASET")
print("-"*60)
print(f"Shape: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
print(f"\nPrimeras 5 filas:")
print(df.head())

print("\n" + "="*60)
print("2. ESCALADO (StandardScaler)")
print("="*60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print("Datos estandarizados (media~0, std~1)")

print("\n" + "="*60)
print("3. K-MEANS - SELECCION DE K POR SILHOUETTE")
print("="*60)

scores = []
for k in range(2, 8):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    scores.append({"k": k, "silhouette": sil})
    print(f"k={k}: silhouette={sil:.3f}")

scores_df = pd.DataFrame(scores)
k_optimo = int(scores_df.sort_values("silhouette", ascending=False).iloc[0]["k"])
print(f"\nMejor k: {k_optimo} (silhouette={scores_df['silhouette'].max():.3f})")

print("\n" + "="*60)
print("4. PERFILADO DE CLUSTERS")
print("="*60)

km_final = KMeans(n_clusters=k_optimo, n_init="auto", random_state=42)
df_clustered = df.copy()
df_clustered["cluster"] = km_final.fit_predict(X_scaled)

print(f"\nDistribucion de clusters:")
print(df_clustered["cluster"].value_counts().sort_index())

print(f"\nMedias por cluster:")
perfiles = df_clustered.groupby("cluster").mean().round(2)
print(perfiles)

print("\n" + "="*60)
print("5. PERFIL DE CADA CLUSTER")
print("="*60)

for c in range(k_optimo):
    cluster_data = df_clustered[df_clustered["cluster"] == c]
    print(f"\nCluster {c} (n={len(cluster_data)}):")
    for col in df.columns:
        media = cluster_data[col].mean()
        print(f"  {col}: {media:.2f}")

print("\n" + "="*60)
print("6. PCA PARA VISUALIZACION")
print("="*60)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
print(f"Varianza explicada: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
print(f"Total: {pca.explained_variance_ratio_.sum():.1%}")

print("\n" + "="*60)
print("7. CONCLUSIONES")
print("="*60)
print(f"""
- Dataset: {df.shape[0]} clientes, {df.shape[1]} features
- Mejor numero de clusters: {k_optimo}
- Silhouette score: {scores_df['silhouette'].max():.3f}

INTERPRETACION:
- Clustering permite segmentar clientes sin labels
- Cada cluster representa un perfil diferente
- PCA reduce dimensiones para visualizacion

RECOMENDACIONES:
- Usar silhouette para validar k
- Analizar perfiles para negocio
- Considerar accionable por segmento
""")