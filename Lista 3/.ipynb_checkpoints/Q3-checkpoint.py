# Importação das bibliotecas necessárias
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from skopt import BayesSearchCV


# Muda para o diretório onde o script está localizado
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Leitura dos Dados
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 2. Pré-processamento dos Dados
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Age"].fillna(test_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)

# Conversão de variáveis categóricas
train_df["Sex"] = train_df["Sex"].map({"male": 0, "female": 1})
test_df["Sex"] = test_df["Sex"].map({"male": 0, "female": 1})
embarked_mapping = {"S": 0, "C": 1, "Q": 2}
train_df["Embarked"] = train_df["Embarked"].map(embarked_mapping)
test_df["Embarked"] = test_df["Embarked"].map(embarked_mapping)

# Seleção de variáveis
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X = train_df[features]
y = train_df["Survived"]

# Divisão do conjunto de treino para validação interna
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Função para treinar, avaliar e plotar a árvore de decisão
def train_and_evaluate(criterion):
    print(f"\n🔹 Treinando Árvore de Decisão com criterion='{criterion}' 🔹")

    # Treinamento do modelo
    clf = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    # Avaliação do modelo
    y_val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    cm = confusion_matrix(y_val, y_val_pred)

    print(f"Acurácia na validação: {acc:.4f}")
    print("Matriz de Confusão:\n", cm)

    # Visualização da árvore
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=features, class_names=["Não Sobreviveu", "Sobreviveu"], filled=True)
    plt.title(f"Árvore de Decisão - Titanic (criterion='{criterion}')")
    plt.show()

    return clf

# Definição dos hiperparâmetros para GridSearchCV
param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 5, 10]
}

# Inicializando o GridSearch
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_grid = grid_search.best_estimator_
y_pred_grid = best_grid.predict(X_val)

# Resultados
print("🔹 Melhor combinação GridSearchCV:", grid_search.best_params_)
print("Acurácia na validação:", accuracy_score(y_val, y_pred_grid))


from scipy.stats import randint

# Definição dos hiperparâmetros para RandomizedSearchCV
param_dist = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 20)
}

# Inicializando o RandomizedSearch
random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_distributions=param_dist, n_iter=20, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_random = random_search.best_estimator_
y_pred_random = best_random.predict(X_val)

# Resultados
print("🔹 Melhor combinação RandomizedSearchCV:", random_search.best_params_)
print("Acurácia na validação:", accuracy_score(y_val, y_pred_random))


# Definição dos hiperparâmetros para BayesSearchCV
param_bayes = {
    "criterion": ["gini", "entropy"],
    "max_depth": (3, 10),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20)
}

# Inicializando o BayesSearch
bayes_search = BayesSearchCV(DecisionTreeClassifier(random_state=42), param_bayes, n_iter=20, cv=5, random_state=42, n_jobs=-1)
bayes_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_bayes = bayes_search.best_estimator_
y_pred_bayes = best_bayes.predict(X_val)

# Resultados
print("🔹 Melhor combinação BayesSearchCV:", bayes_search.best_params_)
print("Acurácia na validação:", accuracy_score(y_val, y_pred_bayes))
