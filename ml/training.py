import pandas as pd # type: ignore
import numpy as np 
import joblib # type: ignore
import json 
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

FEATURE_ORDER = ["circularity","aspect_ratio", "holes", "area", "is_hollow", "material"]
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
    return x,y 

# Treinamento

def train():
    le = LabelEncoder
    y_enc = le.fit_transform(y)

    # Split estratificado : mantém proporção de clases em treino e testes 
    #  
