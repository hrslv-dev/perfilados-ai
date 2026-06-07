"""
training_v2.py
════════════════════════════════════════════════════════════════════════════════
Pipeline de treinamento para 4 classes de perfis metálicos.
Atualizado em relação ao training.py original:
  ✓ Suporta N classes (não apenas binário)
  ✓ ROC-AUC multiclasse com estratégia One-vs-Rest (OvR)
  ✓ Confusion matrix expandida para 4×4
  ✓ Dataset: perfis_dataset_v2.csv (4 classes, 600 amostras)

EXECUTE:
    python ml/training_v2.py

GERA:
    ml/random_forest_v2.pkl
    ml/label_encoder_v2.pkl
    ml/metadata_v2.json
════════════════════════════════════════════════════════════════════════════════

━━━ POR QUE RANDOM FOREST PARA 4 CLASSES? ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

O Random Forest lida com classificação multiclasse de forma NATIVA.
Internamente, cada árvore de decisão já decide entre N classes em cada folha.
Não é preciso treinar N modelos separados (como SVM One-vs-Rest).

━━━ O QUE MUDA DE 2 PARA 4 CLASSES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. predict_proba retorna shape (n_amostras, 4) em vez de (n_amostras, 2)
2. ROC-AUC precisa de multi_class="ovr" (One-vs-Rest):
     Para cada classe, calcula AUC de "essa classe vs todas as outras"
     e faz a média. Valor esperado: 0.95–1.00 para classes bem separadas.
3. Confusion matrix é 4×4 — mostra ENTRE QUAIS CLASSES os erros acontecem.
     Erros esperados: apenas entre oco e maciço da mesma forma (quando
     o furo não foi detectado), nunca entre redondo e quadrado.

━━━ INTERPRETAÇÃO DA CONFUSION MATRIX ESPERADA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                         PREDITO →
                  Q_MAC  Q_OC  R_MAC  R_OC
REAL ↓  Q_MAC  [  ✓      →x    0      0  ]  ← pode confundir com Q_OC
        Q_OC   [  x←      ✓    0      0  ]  ← pode confundir com Q_MAC
        R_MAC  [  0       0     ✓     →x ]  ← pode confundir com R_OC
        R_OC   [  0       0     x←    ✓  ]  ← pode confundir com R_MAC

Erros NUNCA esperados: redondo ↔ quadrado (circularity os separa fortemente)
Erros POSSÍVEIS:       oco ↔ maciço da mesma forma (quando holes=0 por falha)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


FEATURE_ORDER = [
    "circularity",
    "aspect_ratio",
    "holes",
    "area",
    "is_hollow",
    "material",
]

DATASET_PATH = Path(__file__).parent / "perfis_dataset_v2.csv"
MODEL_DIR = Path(__file__).parent


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERRO] Dataset não encontrado: {DATASET_PATH}\n"
            "Execute primeiro: python ml/gerar_dataset_v2.py"
        )
    df = pd.read_csv(DATASET_PATH)
    missing = set(FEATURE_ORDER) - set(df.columns)
    if missing:
        raise ValueError(f"Features ausentes: {missing}")

    print("=" * 65)
    print("  DATASET")
    print("=" * 65)
    print(f"  Total : {len(df)} amostras")
    print(f"  Classes: {sorted(df['label'].unique())}")
    print()
    for cls, cnt in df["label"].value_counts().sort_index().items():
        bar = "█" * int(cnt / 5)
        print(f"  {cls:35s} {cnt:4d}  {bar}")
    print()
    return df[FEATURE_ORDER], df["label"]


def train(X, y):

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_cls = len(le.classes_)

    print("  Classes mapeadas:")
    for i, cls in enumerate(le.classes_):
        print(f"    {i} → {cls}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    print(f"  Treino: {len(X_train)}  |  Teste: {len(X_test)}\n")

    clf = RandomForestClassifier(
        n_estimators=200,
        # 200 árvores: cada uma vota em uma das 4 classes.
        # A classe com mais votos vence.
        min_samples_leaf=2,
        # Previne overfitting em folhas com amostra única.
        max_features="sqrt",
        # sqrt(6) ≈ 2.4 features por split.
        # Garante diversidade — sem isso, circularity e is_hollow
        # dominam todos os splits de todas as árvores.
        class_weight="balanced",
        # Para 4 classes, se no futuro ficarem desbalanceadas
        # (ex: mais tubos ocos que maciços no estoque real),
        # o modelo ainda aprende todas as classes igualmente.
        oob_score=True,
        # OOB estima generalização sem consumir dados de teste.
        # Para 4 classes, ainda é válido (voto de maioria nos OOB samples).
        random_state=42,
        n_jobs=-1,
    )

    print("  Treinando... ", end="", flush=True)
    clf.fit(X_train, y_train)
    print("✓\n")

    # ── Avaliação ────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    # y_proba: shape (120, 4) → 120 amostras de teste × 4 classes
    # Cada linha soma 1.0 (é uma distribuição de probabilidade)

    print("=" * 65)
    print("  CLASSIFICATION REPORT")
    print("=" * 65)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # ── Confusion Matrix 4×4 ─────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    labels = [c[:12] for c in le.classes_]  # abreviado para caber

    print("  Confusion Matrix (linha=real, coluna=predito):")
    header = "  {:12s} | ".format("") + " | ".join(f"{l:12s}" for l in labels)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, row in enumerate(cm):
        cells = " | ".join(
            f"{'[ '+str(v)+' ]':12s}" if j == i else f"{'  '+str(v)+'  ':12s}"
            for j, v in enumerate(row)
        )
        print(f"  {labels[i]:12s} | {cells}")
    print()

    # ── OOB Score ────────────────────────────────────────────────────────────
    print(f"  OOB Score (estimativa interna)   : {clf.oob_score_:.4f}")

    # ── Cross-Validation 5-Fold ───────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X, y_enc, cv=cv, scoring="f1_weighted")
    print(
        f"  Cross-val Accuracy (5-fold)      : {cv_acc.mean():.4f} ± {cv_acc.std():.4f}"
    )
    print(
        f"  Cross-val F1 weighted (5-fold)   : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}"
    )

    # ── ROC-AUC MULTICLASSE ───────────────────────────────────────────────────
    # Para mais de 2 classes, roc_auc_score com multi_class="ovr":
    #
    # One-vs-Rest (OvR):
    #   Para cada classe i, calcula AUC(classe i vs todas as outras).
    #   Resultado: 4 valores de AUC, depois calcula a média (average="macro").
    #
    # Por que não multi_class="ovo" (One-vs-One)?
    #   → OvR é mais comum e interpretável para este contexto.
    #   → OvO é mais robusto com classes muito desbalanceadas,
    #     mas para 150 amostras por classe, OvR é preferível.
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"  ROC-AUC (OvR, macro avg)         : {auc:.4f}")

    # ── Importância das Features ──────────────────────────────────────────────
    fi = pd.Series(clf.feature_importances_, index=FEATURE_ORDER).sort_values(
        ascending=False
    )
    print()
    print("  Importância das Features (Mean Decrease in Gini):")
    for feat, imp in fi.items():
        bar = "█" * int(imp * 50)
        note = ""
        if feat == "circularity":
            note = "  ← separa ROUND vs SQUARE"
        elif feat in ("is_hollow", "holes"):
            note = "  ← separa OC O vs MACIÇO"
        elif feat == "area":
            note = "  ← reforça oco vs maciço"
        elif feat == "aspect_ratio":
            note = "  ← separa quadrado vs retangular/chato"
        print(f"    {feat:15s}  {bar:<50s}  {imp:.4f}{note}")

    # ── Análise dos Erros ─────────────────────────────────────────────────────
    print()
    print("  Análise dos erros de classificação:")
    y_pred_str = le.inverse_transform(y_pred)
    y_test_str = le.inverse_transform(y_test)
    errors = [(t, p) for t, p in zip(y_test_str, y_pred_str) if t != p]
    if errors:
        from collections import Counter

        error_counts = Counter(errors)
        print(f"  Total de erros: {len(errors)} de {len(y_test)}")
        for (real, pred), cnt in error_counts.most_common():
            print(f"    Real: {real:35s} → Predito: {pred}  ({cnt}×)")
        print()
        print("  Interpretação:")
        print("  Erros entre oco↔maciço da MESMA forma são fisicamente")
        print("  justificados: ocorrem quando holes=0 por falha de detecção.")
        print("  Solução: garantir que a câmera capture a ponta do tubo")
        print("  com boa iluminação difusa.")
    else:
        print("  Nenhum erro no conjunto de teste.")

    # ── Salvar artefatos ──────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_DIR / "random_forest_v2.pkl")
    joblib.dump(le, MODEL_DIR / "label_encoder_v2.pkl")

    metadata = {
        "dataset": "perfis_dataset_v2.csv",
        "feature_order": FEATURE_ORDER,
        "classes": list(le.classes_),
        "n_classes": n_cls,
        "n_estimators": clf.n_estimators,
        "oob_score": round(float(clf.oob_score_), 4),
        "cv_accuracy_mean": round(float(cv_acc.mean()), 4),
        "cv_accuracy_std": round(float(cv_acc.std()), 4),
        "cv_f1_mean": round(float(cv_f1.mean()), 4),
        "roc_auc_ovr": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "dataset_source": "synthetic_v2",
        "notes": (
            "4 classes: redondo/quadrado × oco/maciço. "
            "Retreinar com dados reais após validação em fábrica."
        ),
    }
    with open(MODEL_DIR / "metadata_v2.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 65)
    print("  ARTEFATOS SALVOS")
    print("=" * 65)
    for fname in ["random_forest_v2.pkl", "label_encoder_v2.pkl", "metadata_v2.json"]:
        fpath = MODEL_DIR / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"  ✓  {fname:30s}  ({size/1024:.1f} KB)")
    print()

    return clf, le


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TREINAMENTO V2 — 4 CLASSES — MECALD / SENAI           ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    X, y = load_dataset()
    clf, le = train(X, y)
    print("Treinamento V2 concluído.")
