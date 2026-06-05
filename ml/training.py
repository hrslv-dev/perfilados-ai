'''
---------------------------------------------------------------------------------------------------------------------------------------
Imports realizados: 
- Pandas -> manipulação de dados tabulares (DataFrames)
- numpy -> operações númericas de baixo nível 
- joblib -> serialização eficiente de objetos python grandes ( o que é serialização? É o processo de tradução de estruturas de dados ou 
estado de objeto em um formato que possa ser armazenado em arquivo ou buffer de memória)
- json -> salvar metadados em arquivos legíveis ao olho humano 
- Path -> manipulação de caminhos de forma portável
---------------------------------------------------------------------------------------------------------------------------------------
'''

import pandas as pd  # type: ignore
import numpy as np
import joblib  # type: ignore
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings

'''
---------------------------------------------------------------------------------------------------------------------------------------
RESPONSABILIDADES DESTE ARQUIVO:
Carregar o dataset -> validar -> treinar o RandomForest -> avaliar com rigor -> salvar o modelo treinado para uso pelo predictor
em tempo real

COMO EXECUTAR: 
    python ml/training.py 
    Gera arquivos: 
        ml/random_forest.pkl -> modelo serializado
        ml/label_encoder.pkl -> codificador de classes (string -> inteiro)
        ml/metadata.json -> métricas de avaliação + ordem das features

Scikit-learn: 
    É a biblioteca usada para implementar algoritmos de machine learning classicos como RandomForest
    
---------------------------------------------------------------------------------------------------------------------------------------
'''

warnings.filterwarnings("ignore")

FEATURE_ORDER = [
    "circularity",
    "aspect_ratio",
    "holes",
    "area",
    "is_hollow",
    "material",
]
DATASET_PATH = Path(__file__).parent / "perfis_dataset.csv"
MODEL_DIR = Path(__file__).parent

# Carregamento e validação do dataset para treinamento do modelo de RandomForest


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {DATASET_PATH}\n")

    df = pd.read_csv(DATASET_PATH)

    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Features ausentes no CSV: {missing}")
    print(f"Dataset carregado: {len(df)} amostras")

    print(f"Distribuição:\n{df['label'].value_counts().to_string()}\n")
    x = df[FEATURE_ORDER]
    y = df["label"]
    return x, y


# Treinamento


def train():
    x, y = load_dataset()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split estratificado : mantém proporção de classes em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        x, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        # 200 árvores -> equilíbrio de performance/custo
        min_samples_leaf=2,
        # Cada folha pecisa de >= 2 amostras
        max_features="sqrt",
        # Em cada split avalia sqrt(n_features) features
        class_weight="balanced",
        # Pondera classes inversamente à frequência
        oob_score=True,
        # avalia amostras não usadas no bootstrap
        random_state=42,
        n_jobs=-1,
    )
