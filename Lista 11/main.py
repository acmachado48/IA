import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1. LEITURA DOS DADOS
# =========================

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# =========================
# 2. ANÁLISE EXPLORATÓRIA
# =========================
print("📊 Estatísticas Descritivas:\n", train_df.describe(include='all'))
print("\n🔍 Valores nulos (treinamento):\n", train_df.isnull().sum())
print("\n🔍 Valores nulos (teste):\n", test_df.isnull().sum())

plt.figure(figsize=(10, 8))
sns.heatmap(train_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.show()

sns.countplot(x='Survived', data=train_df)
plt.title("Distribuição de Sobreviventes")
plt.show()

# =========================
# 3. PRÉ-PROCESSAMENTO
# =========================
# Preencher nulos
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Engenharia de atributos
for df in [train_df, test_df]:
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = (df["FamilySize"] == 0).astype(int)
    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(['Lady', 'Countess', 'Capt', 'Col',
                                       'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})
    df["Title"] = df["Title"].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}).fillna(0)

# Conversão categórica
sex_map = {"male": 0, "female": 1}
embarked_map = {"S": 0, "C": 1, "Q": 2}
for df in [train_df, test_df]:
    df["Sex"] = df["Sex"].map(sex_map)
    df["Embarked"] = df["Embarked"].map(embarked_map)

# Features
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked",
            "FamilySize", "IsAlone", "Title"]

# Normalização
scaler = StandardScaler()
train_df[["Age", "Fare", "FamilySize"]] = scaler.fit_transform(train_df[["Age", "Fare", "FamilySize"]])
test_df[["Age", "Fare", "FamilySize"]] = scaler.transform(test_df[["Age", "Fare", "FamilySize"]])

# =========================
# 4. MODELAGEM - RANDOM FOREST
# =========================
X = train_df[features]
y = train_df["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

print("=== Random Forest ===")
print(classification_report(y_val, y_pred_rf, digits=3))

# =========================
# 5. MODELAGEM - REDE NEURAL SIMPLES
# =========================
# Rede neural customizada (sem sklearn)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# Conversão para numpy arrays
X_train_np = X_train.to_numpy()
y_train_np = y_train.to_numpy().reshape(-1, 1)

# Inicialização
n_input = X_train_np.shape[1]
n_hidden = 3
n_output = 1
np.random.seed(42)

w1 = np.random.uniform(-1, 1, (n_input, n_hidden))
b1 = np.zeros((1, n_hidden))

w2 = np.random.uniform(-1, 1, (n_hidden, n_output))
b2 = np.zeros((1, n_output))

epochs = 1000
lr = 0.1

for epoch in range(epochs):
    # Forward
    z1 = np.dot(X_train_np, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Erro
    error = y_train_np - a2

    # Backprop
    d_a2 = error * sigmoid_deriv(a2)
    d_a1 = d_a2.dot(w2.T) * sigmoid_deriv(a1)

    # Atualização
    w2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2, axis=0, keepdims=True) * lr
    w1 += X_train_np.T.dot(d_a1) * lr
    b1 += np.sum(d_a1, axis=0, keepdims=True) * lr

    if epoch % 100 == 0:
        print(f"Época {epoch} - Erro médio: {np.mean(np.abs(error)):.4f}")

# Avaliação na validação
X_val_np = X_val.to_numpy()
z1_val = np.dot(X_val_np, w1) + b1
a1_val = sigmoid(z1_val)
z2_val = np.dot(a1_val, w2) + b2
a2_val = sigmoid(z2_val)
pred_nn = (a2_val > 0.5).astype(int)

print("\n=== Rede Neural ===")
print(classification_report(y_val, pred_nn, digits=3))

# =========================
# 6. SUBMISSÃO
# =========================
X_test = test_df[features]
y_pred_final = rf.predict(X_test)  # escolha: rf ou rede neural

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": y_pred_final
})

submission.to_csv("submission.csv", index=False)
print("✅ Arquivo 'submission.csv' gerado com sucesso!")

# =========================
# 7. GRID SEARCH RANDOM FOREST
# =========================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

print("\n🌟 Melhor combinação de hiperparâmetros:")
print(grid_search.best_params_)

# Avaliação final
y_val_pred = best_rf.predict(X_val)
print("\n📊 Relatório de classificação (Random Forest ajustado):")
print(classification_report(y_val, y_val_pred))

# ===============================
# 8. IMPORTÂNCIA DAS VARIÁVEIS
# ===============================
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.title("📈 Importância das Variáveis")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45)
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Usar os mesmos dados normalizados
X_cluster = train_df[features]

from sklearn.metrics import silhouette_score

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    score = silhouette_score(X_cluster, labels)
    print(f"k={k}, silhouette score={score:.4f}")

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
train_df["Cluster"] = kmeans.fit_predict(X_cluster)

# Redução de dimensionalidade para visualização
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X_cluster)

# Visualizar os clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1],
                hue=train_df["Cluster"], palette="Set1", s=60)
plt.title("Agrupamento de Passageiros com K-Means (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.legend(title="Cluster")
plt.show()

# Análise descritiva dos clusters
cluster_profiles = train_df.groupby("Cluster")[["Survived", "Sex", "Pclass", "Age"]].mean()
print("\n📊 Perfil médio por cluster:\n", cluster_profiles)

# Exemplo: Distribuição de sobreviventes por cluster
sns.countplot(x='Cluster', hue='Survived', data=train_df)
plt.title("Distribuição de Sobreviventes por Cluster")
plt.show()

from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Selecionar atributos categóricos relevantes
assoc_df = train_df.copy()
assoc_df["Sex"] = assoc_df["Sex"].map({0: "Male", 1: "Female"})
assoc_df["Pclass"] = assoc_df["Pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})
assoc_df["Survived"] = assoc_df["Survived"].map({0: "Did not Survive", 1: "Survived"})

# Atributos para análise de associação
assoc_data = assoc_df[["Sex", "Pclass", "Embarked", "Title", "Survived"]].astype(str).values.tolist()

# Codificar transações
te = TransactionEncoder()
te_data = te.fit(assoc_data).transform(assoc_data)
df_trans = pd.DataFrame(te_data, columns=te.columns_)

# Aplicar Apriori
frequent_itemsets = apriori(df_trans, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Ordenar por lift
rules = rules.sort_values(by="lift", ascending=False)
print("\n📋 Top 3 regras extraídas:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(3))

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_components = tsne.fit_transform(X_cluster)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=tsne_components[:, 0], y=tsne_components[:, 1],
                hue=train_df["Cluster"], palette="Set2", s=60)
plt.title("Clusters com t-SNE")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Cluster")
plt.show()
