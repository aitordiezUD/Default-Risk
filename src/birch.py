import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import optuna
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA


df = pd.read_parquet('../data/processed/selected_features_df.parquet')
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

print(f"Shape of the dataframe: {df.shape}")

pca = PCA(n_components=2)  # Adjust this number to reduce dimensionality
df_scaled_reduced = pca.fit_transform(df_scaled)

def objective(trial):
    # Sugerir valores para los hiperparámetros de BIRCH
    threshold = trial.suggest_float('threshold', 0.01, 1.0)  # Umbral para la construcción del árbol
    # n_clusters = trial.suggest_int('n_clusters', 2, 10)  # Número de clusters
    # branching_factor = trial.suggest_int('branching_factor', 10, 100)  # Número de hijos por nodo

    # Crear el modelo BIRCH con los hiperparámetros sugeridos
    model = Birch(threshold=threshold, n_clusters=2)

    # df_sparse = csr_matrix(df_scaled.get())
    # model.fit(df_sparse)
    # Entrenar el modelo
    model.fit(df_scaled_reduced)
    labels = model.labels_

    # Calcular el Silhouette Score para evaluar la calidad del clustering
    if len(np.unique(labels)) > 1:
        score = silhouette_score(df_scaled_reduced, labels)
        return score
    else:
        return -1

study = optuna.create_study(direction='maximize', study_name='birch_tuning')
study.optimize(objective, n_trials=5)