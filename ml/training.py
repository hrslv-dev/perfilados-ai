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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

    # Imprime na tela como as classes estão distribuidas no dataset
    # Vizualização de como o dataset está estruturado

    for cls, cnt in df["label"].value_counts().items():
        pct = cnt / len(df) * 100
        bar = "█" * int(pct / 3)
        print(f" {cls:35s} {cnt:4d} ({pct:.0f}%) {bar}")
    print()
    print(" Estatística das features: ")
    print(df[FEATURE_ORDER].describe().round(3).to_string())
    print()

    X = df[FEATURE_ORDER]
    y = df["label"]
    return X, y


"""
---------------------------------------------------------------------------------------------------------------------------------------
TREINAMENTO: 
Mas o que é o RandomForest? 
    A ideia dele é a seguinte, ao em vez de treinar apenas uma árvore de decisão grande, que tende a ficar "viciada nos 
    dados de treino, treinar MUITAS árvores menores e independentes, e combinar suas respostas 

- Árvore de decisão - 
    Esta aprende perguntas binárias sore as features. Cada pergunta é escolhida automaticamente pelo algoritmo usado
    a critério 
---------------------------------------------------------------------------------------------------------------------------------------
"""


def train(X: pd.DataFrame, y: pd.Series) -> tuple:
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(" Mapeamento de classes ")

    # Encode de labels
    # RandomForestClassifier (internamente) trabalha com inteiros, ou seja cada classificação é um valor inteiro númerico
    # O LabelEncoder aprende esse mapeamento de classes com .fit_transform(y)
    # Por que salvar encoder junto ao modelo? Se eu treinar em ordem diferente amanhã (outra versão do dataset), o modelo vai
    # encontrar os labels errados.

    # O que significa esse loop for?
    for i, cls in enumerate(le.classes_):
        print(f"   {i} -> {cls}")
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    print(f" SPLIT DO DATASET: ")
    print(f"   Treino: {len(X_train)} amostras (80%)")
    print(f"   Teste: {len(X_test)} amostras (20%)")

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    ESCOLHA DOS HIPERPARÂMETROS:     
    ---------------------------------------------------------------------------------------------------------------------------------------
    """
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=42,
        n_jobs=-1,
    )

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Número de árvores no ensemble: 
        Quanto mais árvores, mais robusto (menos variância) mas mais lento computacionalmente 
        - 10 -> 20 árvores: melhora muito 
        - 100 -> 200 árvores: melhora pouco 
        - 200 -> 200+ árvores: melhora praticamente nada
        Ou seja, para um dataset de 300 amostras com 6 features 200 árvores é mais que suficiente e o treinamento 
        é consideravelmente rápdio    
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Profundidade mínima de cada árvore: 
        As árvores crescem até que cada folha tenha < min_samples_leaf. 
        Para datasets pequenos e bem separáveis, árvores profundas são razoavelmente ok. 
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Número mínimo de amostras que uma folha (nó terminal) deve ter:
        2 nós terminais ignora casos únicos/ruidosos, generaliza melhor, ou seja menos chance de overfitting
        Para datasets pequenos, 2 é um bom ponto de partida
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Número de features consideradas em cada split: 
        sqrt = sqrt(n_features)
        com 6 features (sqrt(6)), avalia 2 ou 3 features por split 
        Por que não avaliar todas as 6 de uma vez? 
        -> Se uma feature é dominante, todas as árvores usariam essa feature no nó raíz 
        -> árvores correlacionadas -> ensemble   
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Peso de classe: 
        Pesa as classes inversamente a frequência no dataset
        Caso uma classe apareça muito mais que outras, isso garante que o modelo não ignore as minorias 
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Out-of-bag score:
        Cada árvore usa -63% dos dados para treinar
        - Os -37% restantes são usados para avaliar ESSA árvore 
        Ou seja, o oob é a acurácia média de cada amostra nas árvores que não viram durante o treino 
        - É estimativa de generalização GRATUITA (sem gastar dados de teste)
        Para datasets menores é especialmente valioso
        Se oobScore = Test score -> não há overfitting significativo
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    """
    ---------------------------------------------------------------------------------------------------------------------------------------
    Estado Randomico: 
        Garante reprodutibilidade :
        Treinar de novo gera o mesmo modelo 
    ---------------------------------------------------------------------------------------------------------------------------------------
    """

    # .fit é onde o treino acontece de verdade
    # Internamente o sklearn:
    # - Gera 200 amostras bootstrap do (X_train, y_train)
    # - Para cada amostra treina uma árvore recursivamente
    # - Armazena as 200 árvores no atributo .estimators_

    print(" Treinando RandomForest...", end="", flush=True)
    clf.fit(X_train, y_train)
    print("Concluído")
    print()

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    """
    - Classification Report - 
    Para cada classe mostra: 
        - Precision -> TP/(TP+FP)
            (Precision baixa : muitos falsos positivos) 
        - Recall -> TP/(TP+FN)
            (Recall baixo: o modelo está falhando em pegar casos reais)
        - FI-score -> (Precision * Recall) | (Precision + Recall)
            (Média harmônica entre precision e recall)
        - Support -> número de amostras dessa classe no conjunto de teste
    """

    print("=" * 60)
    print(" RELATÓRIO DE CLASSIFICAÇÃO (conjunto teste)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    """
    - Confusion matrix -  
        Mostra onde o modelo erra 
        - Diagonal principal = acertos 
        - Fora da diagonal = erros 
        
        Para uso industrial: errar "redondo" como "quadrado" pode gerar 
        alocação errada no estoque. Analisar qual tipo de erro é mais caro
    """

    cm = confusion_matrix(y_test, y_pred)
    print(" Matriz de confusão: ")
    print(f"  {'':25s} Predito : {le.classes_[0]:30s} Predito: {le.classes_[1]}")
    for i, row in enumerate(cm): 
        print(f"  Real: {le.classes_[1]:20s} {row[0]:^{len(le.classes_[0]) + 10}d} {row[1]:^{len(le.classes_[1]) + 10}d}")
    print() 
        
    print(f" OOB score (estimativa interna, sem usar test set): {clf.oob_score_:.4f}")
    
    
    '''
    -Cross validation-
        O test split usa apenas 20% dos dados (60 amostras) 
        Com dataset pequeno, esse resultado pode ter sorte ou azar
        
        Cross validation divide o dataset em K partes(folds) e repete
        o treino/avaliação K vezes, cada vez usando um fold diferente como 
        teste 
        
        Resultado: 
        5 acurácias. A média é muito mais confiável que uma única 
        
        StratifiedKFold garante que cada fold mantenha a proporção de 
        classes 
    '''    
    cv = StratifiedKFold(n_splits=5, shuffle=True,random_state=42)
    cv_acc = cross_val_score(clf,X,y_enc, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf,X,y_enc,cv=cv,scoring="f1_weighted")
    
    print()
    print(" Cross Validation 5-fold: ")
    for i, acc in enumerate(cv_acc): 
        bar = "█" * int(acc * 30)
        print(f"    Fold {i+1}: {acc:.4f}  {bar}")
    print(f"    ─────────────────────────────────────────")
    print(f"    Média : {cv_acc.mean():.4f}  ± {cv_acc.std():.4f}")
    print(f"    F1    : {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
 
    '''
    - ROC-AUC Score -
    ROC -> Receiver Operating Characteristic Curve 
        Plota True positive Rate vs False positive Rate para cada
        threshold 
    AUC -> Area under the curve 
        1 -> Separação perfeita
        0.50 -> completamente aleatório 
        0.70 -> aceitável para produção 
        0.9 -> muito bom 
    
    Para classificação binária (duas classes), usamos a coluna de probabilidade 
    da classe positiva. 
    '''
    
    auc = roc_auc_score(y_test,y_proba[:, 1])
    print(f" ROC-AUC         :   {auc:.4f}")
    print()
    
    fi = pd.Series(clf.feature_importances_, index=FEATURE_ORDER)
    fi = fi.sort_values(ascending=False)
    print(" Importância das features (Mean decrease in Gini Impurity): ")
    for feat, imp, in fi.items(): 
        bar = "█" * int(imp * 50)
        print(f"    {feat:15s}  {bar:<50s}  {imp:.4f}")
    print()
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf,MODEL_DIR / "random_forest.pkl")
    joblib.dump(le,MODEL_DIR / "label_encoder.pkl")
    
    metadata = { 
        "feature_order":     FEATURE_ORDER,
        "classes":           list(le.classes_),
        "n_estimators":      clf.n_estimators,
        "oob_score":         round(float(clf.oob_score_), 4),
        "cv_accuracy_mean":  round(float(cv_acc.mean()), 4),
        "cv_accuracy_std":   round(float(cv_acc.std()), 4),
        "cv_f1_mean":        round(float(cv_f1.mean()), 4),
        "roc_auc":           round(float(auc), 4),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "dataset_source":    "synthetic",
        "notes":             (
        "Dataset sintético. Retreinar com dados reais após "
        "validação física em ambiente controlado."
        )
        }
    
    with open(MODEL_DIR / "metadadata.json", "w", enconding="utf-8") as f: 
        json.dump(metadata,f,indent=2,ensure_ascii=False)
        
    print("=" * 60)
    print("  ARTEFATOS SALVOS")
    print("=" * 60)
    for fname in ["random_forest.pkl", "label_encoder.pkl", "metadata.json"]:
        fpath = MODEL_DIR / fname
        size  = fpath.stat().st_size if fpath.exists() else 0
        print(f"  ✓  {fname:25s}  ({size / 1024:.1f} KB)")
    print()
 
    return clf, le

if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     TREINAMENTO — CLASSIFICADOR DE PERFIS METÁLICOS     ║")
    print("║     Mecald / SENAI Pedro Leopoldo                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
 
    X, y   = load_dataset()
    clf, le = train(X, y)
 
    print("Treinamento concluído.")
    print("Execute 'python ml/model.py' para testar a inferência.")