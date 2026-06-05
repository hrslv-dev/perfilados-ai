"""
---------------------------------------------------------------------------------------------------------------------------------------
Imports realizados:
- Pandas -> manipulação de dados tabulares (DataFrames)
- numpy -> operações númericas de baixo nível
- joblib -> serialização eficiente de objetos python grandes ( o que é serialização? É o processo de tradução de estruturas de dados ou
estado de objeto em um formato que possa ser armazenado em arquivo ou buffer de memória)
- json -> salvar metadados em arquivos legíveis ao olho humano
- Path -> manipulação de caminhos de forma portável
---------------------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd  # type: ignore
import numpy as np
import joblib  # type: ignore
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings

"""
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
    Ela implementa dezenas de algorítmos com uma interface unificada com: 
    .fit(x,y) para treinar e .predict(x) para predizer. 
---------------------------------------------------------------------------------------------------------------------------------------
"""
warnings.filterwarnings("ignore")

"""
---------------------------------------------------------------------------------------------------------------------------------------
CONSNTES NO PROJETO: 
FEATURE_ORDER -> Lista que define a ordem de quais colunas do dataset são features e em que ordem entram no modelo
A lista DEVE SER IDENTICA entre training.py e model.py

Por que? 
Random Forest aprende internamente que: 
 - coluna 0 -> circularity 
 - coluna 1 -> aspect_ratio 
 - coluna 2 -> holes 

Se na hora da predição essas features forem invertidas, o modelo usará o valor errado e consequentemente preverá errado
---------------------------------------------------------------------------------------------------------------------------------------
"""

FEATURE_ORDER = [
    "circularity",  # Quão próximo de um círculo (0 a 1)
    "aspect_ratio",  # largura/ altura do bounding rectangle
    "holes",  # número de buracos internos detectados
    "area",  # área do contorno em pixels quadrados
    "is_hollow",  # valor binário: 1 se holes > 0
    "material",  # código do material em valor inteiro -> tabela escalar de materiais
    """
    Material | Código
    carbono  | 0
    alumínio | 1 
    inox     | 2
    ...
    
    """,
]
DATASET_PATH = Path(__file__).parent / "perfis_dataset.csv"
MODEL_DIR = Path(__file__).parent


"""
---------------------------------------------------------------------------------------------------------------------------------------
CARREGAMENTO E VALIDAÇÃO DO DATASET 
---------------------------------------------------------------------------------------------------------------------------------------
"""


def load_dataset():
    # Exceção caso o arquivo do dataset não seja encontrado
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {DATASET_PATH}\n")
    df = pd.read_csv(DATASET_PATH)

    # Validação da presença das features
    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Features ausentes no CSV: {missing}")
    if "label" not in df.columns: 
        raise ValueError(f"Erro, Coluna 'label' não foi encontrada no dataset")
    
    print(f"Dataset carregado: {len(df)} amostras\n")
    print(f"Total de Features : {len(FEATURE_ORDER)}\n")
    print()
    print(f"Distribuição de classes: ")
    for cls, cnt in df["label"].value_counts().items(): 
        pct = cnt/ len(df) * 100
        bar = "█" * int(pct / 3)
        print(f" {cls:35s} {cnt:4d} ({pct:.0f}%) {bar}")
    print()
    print(" Estatística das features: ")
    print(df[FEATURE_ORDER].describe().round(3).to_string())
    print()
            
    X = df[FEATURE_ORDER]
    y = df["label"]
    return X, y

'''
---------------------------------------------------------------------------------------------------------------------------------------
TREINAMENTO: 
Mas o que é o RandomForest? 
    A ideia dele é a seguinte, ao em vez de treinar apenas uma árvore de decisão grande, que tende a ficar "viciada nos 
    dados de treino, treinar MUITAS árvores menores e independentes, e combinar suas respostas 

- Árvore de decisão - 
    Esta aprende perguntas binárias sore as features. Cada pergunta é escolhida automaticamente pelo algoritmo usado
    a critério 
---------------------------------------------------------------------------------------------------------------------------------------

'''
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
