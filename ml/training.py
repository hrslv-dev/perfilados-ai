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
    x, y = load_dataset()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split estratificado : mantém proporção de classes em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(x, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
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
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    model_path = MODEL_DIR / "random_forest.joblib"
   
       # Avaliação
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
    cv_f1  = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_weighted")
    auc    = roc_auc_score(y_test, y_proba[:, 1]) if len(le.classes_) == 2 else None

    print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print(f"OOB Score (estimativa sem test set) : {clf.oob_score_:.4f}")
    print(f"Cross-val Accuracy                  : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
    print(f"Cross-val F1-Score                  : {cv_f1.mean():.4f}")
    if auc:
        print(f"ROC-AUC                             : {auc:.4f}")

    print("\n=== IMPORTÂNCIA DAS FEATURES ===")
    fi = pd.Series(clf.feature_importances_, index=FEATURE_ORDER).sort_values(ascending=False)
    for feat, imp in fi.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:15s} {bar} {imp:.4f}")

    # Salva artefatos
    joblib.dump(clf, MODEL_DIR / "random_forest.pkl")
    joblib.dump(le,  MODEL_DIR / "label_encoder.pkl")

    metadata = {
        "feature_order":     FEATURE_ORDER,
        "classes":           list(le.classes_),
        "oob_score":         round(clf.oob_score_, 4),
        "cv_accuracy":       round(cv_acc.mean(), 4),
        "cv_accuracy_std":   round(cv_acc.std(), 4),
        "cv_f1":             round(cv_f1.mean(), 4),
        "roc_auc":           round(auc, 4) if auc else None,
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "n_estimators":      200,
        "dataset_source":    "synthetic",  # mude para "real" quando coletar dados reais
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Modelo salvo em {MODEL_DIR}/random_forest.pkl")
    print(f"✓ Metadados salvos em {MODEL_DIR}/metadata.json")

    return clf, le


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, y = load_dataset()
    clf, le = train(X, y)
