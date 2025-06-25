import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Carregar dados com rótulo original
df = pd.read_csv("/Users/anacarolinamachado/iA/IA/Lista 7/Iris_preprocessed.csv")

# 2. Executar KMeans exatamente como em kmeans.py
X = df.select_dtypes(include=[np.number]).values
X = MinMaxScaler().fit_transform(X)  # reescala 

kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
clusters = kmeans.fit_predict(X)
df["cluster"] = clusters

# Mapear cada cluster para a espécie dominante
dominant_species = (
    df.groupby("cluster")["class"]
      .agg(lambda x: x.value_counts().idxmax())
      .to_dict()
)

# Identificar instâncias mal agrupadas
df["pred_species"] = df["cluster"].map(dominant_species)
df["misclassified"] = df["class"] != df["pred_species"]

mis_df = df[df["misclassified"]].copy()

# Substitui a função da ace_tools por um print
print("Instâncias mal agrupadas:")
print(mis_df.head(10))

print(f"Total de instâncias: {len(df)}")
print(f"Instâncias fora do cluster correto: {mis_df.shape[0]}")

# Plot: duas primeiras features normalizadas
plt.figure()
for c in sorted(df["cluster"].unique()):
    idx = df["cluster"] == c
    plt.scatter(X[idx, 0], X[idx, 1], label=f"Cluster {c}", s=40)

# sobrepor mal classificados
plt.scatter(
    X[df["misclassified"], 0],
    X[df["misclassified"], 1],
    marker="x",
    s=80,
    linewidths=1.5,
    label="Mal agrupado"
)

plt.xlabel("Feature 1 (esc. 0–1)")
plt.ylabel("Feature 2 (esc. 0–1)")
plt.title("KMeans (k=3) – pontos mal agrupados marcados com X")
plt.legend()
plt.tight_layout()
plt.show()
