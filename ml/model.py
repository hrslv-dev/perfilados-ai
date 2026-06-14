# =============================================================================
#  model.py — Módulo de Inferência (Predição em Tempo Real)
#  Projeto: Classificação de Perfis Metálicos — Mecald / SENAI Pedro Leopoldo
# =============================================================================
#
#  RESPONSABILIDADE DESTE ARQUIVO
#  ─────────────────────────────────────────────────────────────────────────────
#  Carregar o modelo treinado pelo training.py e disponibilizar uma interface
#  limpa para o predictor (câmera em tempo real) usar.
#
#  SEPARAÇÃO DE RESPONSABILIDADES
#  ─────────────────────────────────────────────────────────────────────────────
#  training.py → "aprende" com os dados         (roda uma vez, offline)
#  model.py    → "usa" o que aprendeu           (roda sempre, em tempo real)
#
#  Analogia: training.py é o período de estudo. model.py é a prova.
#  Você não "re-estuda" a cada prova — você usa o que já aprendeu.
#
#  COMO USAR
#  ─────────────────────────────────────────────────────────────────────────────
#  from ml.model import Classifier
#
#  clf     = Classifier()
#  features = { "circularity": 0.92, "aspect_ratio": 1.01,
#               "holes": 1, "area": 3200.0, "is_hollow": 1, "material": 0 }
#  resultado = clf.predict(features)
#  print(resultado["label"])       # "tubo_redondo"
#  print(resultado["confidence"])  # 0.9742
#
#  COMO TESTAR ISOLADO
#  ─────────────────────────────────────────────────────────────────────────────
#  python ml/model.py
#  → Roda demos automáticos com features simuladas
# =============================================================================

import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuração ─────────────────────────────────────────────────────────────

MODEL_DIR = Path(__file__).parent

# CONFIDENCE_THRESHOLD: limiar mínimo de confiança para aceitar uma predição
# automática sem pedir confirmação do operador.
#
# Como funciona o predict_proba() do RandomForest?
# ─────────────────────────────────────────────────
# Cada uma das 200 árvores vota em uma classe.
# A confiança é a PROPORÇÃO de votos para a classe vencedora.
#
# Exemplo:
#   120 árvores votam "tubo_redondo"    →  60% de confiança
#    80 árvores votam "tubo_quadrado"   →  40% de confiança
#   → Resultado: "tubo_redondo" com 60% de confiança
#
# Por que 0.70 como limiar?
# ──────────────────────────
# Abaixo de 70%: o modelo está em dúvida — menos de 140 das 200 árvores
# concordaram. Nesse caso, melhor pedir confirmação humana do que registrar
# um material errado no estoque.
#
# Em produção real, este valor deve ser ajustado com base na taxa de erro
# observada durante os primeiros meses de uso.

CONFIDENCE_THRESHOLD = 0.70

# Mapeamento para display amigável na interface
DISPLAY_NAMES = {
    "tubo_redondo":             "Tubo Redondo",
    "tubo_quadrado_retangular": "Tubo Quadrado / Retangular",
}

MATERIAL_NAMES = {
    0: "Aço Carbono",
    1: "Inox",
    2: "Alumínio",
}


# =============================================================================
#  COMO O RANDOMFOREST FAZ UMA PREDIÇÃO (PASSO A PASSO)
# =============================================================================
#
#  Quando clf.predict_proba([[0.92, 1.01, 1, 3200, 1, 0]]) é chamado:
#
#  1. O vetor de features é passado por cada uma das 200 árvores.
#
#  2. Para cada árvore, o vetor percorre o caminho da raiz até uma folha,
#     respondendo perguntas binárias:
#
#       Árvore 17, por exemplo:
#         circularity > 0.841?      → SIM (0.92 > 0.841)
#         area > 1500?              → SIM (3200 > 1500)
#         → Folha: [0 amostras quad, 8 amostras round] → vota: tubo_redondo
#
#       Árvore 53, por exemplo:
#         aspect_ratio > 1.05?     → NÃO (1.01 < 1.05)
#         circularity > 0.78?      → SIM (0.92 > 0.78)
#         → Folha: [1 amostra quad, 12 amostras round] → vota: tubo_redondo
#
#  3. Coleta os votos de todas as 200 árvores:
#       tubo_redondo:             194 votos → 97%
#       tubo_quadrado_retangular:   6 votos →  3%
#
#  4. predict_proba retorna: [0.03, 0.97]
#     predict retorna: 1 (índice da classe com mais votos)
#     inverse_transform(1) → "tubo_redondo"
#
#  5. O Classifier.predict() formata esse resultado em um dicionário
#     com label, confiança, flag de baixa confiança e mensagem.
# =============================================================================


class Classifier:
    """
    Interface de inferência do modelo treinado.

    Carrega o RandomForest e o LabelEncoder do disco e expõe
    um método .predict() que recebe o dicionário de features do
    FeatureExtractor e retorna a classificação do perfil.

    Atributos após __init__:
        clf            : RandomForestClassifier carregado
        le             : LabelEncoder carregado
        feature_order  : lista de features na ordem correta (do metadata.json)
        classes        : lista de classes possíveis
    """

    def __init__(self):
        """
        Carrega os artefatos gerados pelo training.py.

        Por que carregar do disco em vez de treinar na hora?
        ─────────────────────────────────────────────────────
        Treinar o modelo demora (alguns segundos). Carregar do disco
        é quase instantâneo. Em um sistema em tempo real que processa
        30 frames por segundo, você não pode treinar a cada frame.

        O modelo é carregado UMA VEZ quando o sistema inicia,
        e depois é reutilizado para todas as predições.
        """
        model_path   = MODEL_DIR / "random_forest.pkl"
        encoder_path = MODEL_DIR / "label_encoder.pkl"
        meta_path    = MODEL_DIR / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(
                "\n[ERRO] Modelo não encontrado.\n"
                "Execute primeiro: python ml/training.py\n"
                f"Caminho esperado: {model_path}"
            )

        # joblib.load() desserializa o objeto Python salvo pelo joblib.dump()
        # O RandomForestClassifier carregado é idêntico ao que foi treinado —
        # incluindo todas as 200 árvores com todos os seus splits.
        self.clf = joblib.load(model_path)
        self.le  = joblib.load(encoder_path)

        # O metadata.json é a fonte da verdade sobre a ordem das features.
        # Usar o arquivo salvo em vez da constante local garante que, mesmo
        # se o código mudar depois, o modelo usa a ordem com que foi treinado.
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        self.feature_order = meta["feature_order"]
        self.classes       = meta["classes"]
        self._meta         = meta

        print(f"[Classifier] Modelo carregado.")
        print(f"  Classes    : {self.classes}")
        print(f"  Features   : {self.feature_order}")
        print(f"  OOB Score  : {meta.get('oob_score', 'N/A')}")
        print(f"  CV Accuracy: {meta.get('cv_accuracy_mean', 'N/A')} ± {meta.get('cv_accuracy_std', 'N/A')}")
        print()

    def predict(
        self,
        features: dict,
        material: int | None = None
    ) -> dict:
        """
        Classifica um perfil com base no dicionário de features.

        Parâmetros:
        ──────────────────────────────────────────────────────────────────────
        features : dict retornado por FeatureExtractor.extract_features()
                   Deve conter as chaves definidas em FEATURE_ORDER.
                   Chaves extras são ignoradas.

        material : int | None
                   Código do material (0=carbono, 1=inox, 2=alumínio).
                   Se passado, sobrescreve o valor em features["material"].
                   Use quando o operador selecionou o material manualmente.

        Retorna: dict com os campos abaixo
        ──────────────────────────────────────────────────────────────────────
        {
            "label":          str    → nome da classe predita
            "label_display":  str    → nome legível para a interface
            "confidence":     float  → probabilidade da classe predita (0-1)
            "confidence_pct": int    → confiança em percentual (0-100)
            "low_confidence": bool   → True se confiança < CONFIDENCE_THRESHOLD
            "all_probas":     dict   → probabilidade de CADA classe
            "message":        str    → mensagem para exibir ao operador
            "features_used":  dict   → as features exatamente como foram usadas
            "error":          str|None → descrição do erro, se houver
        }
        """

        # ─── Injetar material ─────────────────────────────────────────────────
        # Cria uma cópia para não modificar o dict original passado pelo caller
        f = dict(features)

        if material is not None:
            f["material"] = int(material)
        elif "material" not in f:
            # Default: aço carbono (mais comum na Mecald)
            f["material"] = 0

        # ─── Montar vetor de features na ordem correta ────────────────────────
        #
        # Este é o passo mais crítico da inferência.
        #
        # O modelo internamente sabe:
        #   "coluna 0 é circularity, coluna 1 é aspect_ratio..."
        # Se você passar as colunas em ordem errada, ele vai interpretar
        # aspect_ratio como circularity e tudo vai por água abaixo.
        #
        # Usando um DataFrame com as colunas nomeadas, o sklearn garante
        # a ordem e elimina o warning de feature names.

        try:
            X = pd.DataFrame([{feat: f[feat] for feat in self.feature_order}])
        except KeyError as missing_key:
            return {
                "label":          None,
                "confidence":     0.0,
                "low_confidence": True,
                "error":          (
                    f"Feature ausente: {missing_key}. "
                    f"Features recebidas: {list(f.keys())}. "
                    f"Features esperadas: {self.feature_order}"
                )
            }

        # ─── Predição ─────────────────────────────────────────────────────────
        #
        # predict_proba retorna shape (1, n_classes):
        #   [[P(classe_0), P(classe_1)]]
        #
        # np.argmax encontra o índice da maior probabilidade.
        # le.inverse_transform converte o índice de volta para string.

        probas     = self.clf.predict_proba(X)[0]    # array 1D com probabilidades
        pred_idx   = int(np.argmax(probas))           # índice da classe vencedora
        confidence = float(probas[pred_idx])          # confiança = proporção de votos

        label = self.le.inverse_transform([pred_idx])[0]

        # ─── Montar resultado ─────────────────────────────────────────────────
        result = {
            "label":          label,
            "label_display":  DISPLAY_NAMES.get(label, label),
            "confidence":     round(confidence, 4),
            "confidence_pct": round(confidence * 100),
            "low_confidence": confidence < CONFIDENCE_THRESHOLD,
            "all_probas": {
                cls: round(float(p), 4)
                for cls, p in zip(self.classes, probas)
            },
            "features_used": {feat: f[feat] for feat in self.feature_order},
            "error": None,
        }

        # ─── Mensagem para o operador ─────────────────────────────────────────
        if result["low_confidence"]:
            result["message"] = (
                f"⚠  Confiança baixa ({result['confidence_pct']}%). "
                "Confirme a classe visualmente antes de registrar."
            )
        else:
            result["message"] = (
                f"✓  {result['label_display']} — "
                f"{result['confidence_pct']}% de confiança."
            )

        return result

    def explain(self, features: dict) -> str:
        """
        Gera uma explicação textual de por que o modelo classificou assim.

        Não substitui SHAP/LIME para explicabilidade completa, mas fornece
        um resumo rápido das features mais relevantes para esta predição.

        Útil para debug e para o operador entender o resultado.
        """
        result = self.predict(features)
        if result["error"]:
            return f"Erro: {result['error']}"

        f = result["features_used"]
        fi = pd.Series(
            self.clf.feature_importances_,
            index=self.feature_order
        ).sort_values(ascending=False)

        lines = [
            f"CLASSIFICAÇÃO: {result['label_display']}",
            f"CONFIANÇA    : {result['confidence_pct']}%",
            f"",
            f"FEATURES (mais importantes primeiro):",
        ]
        for feat, imp in fi.items():
            val = f[feat]
            bar = "█" * int(imp * 30)
            lines.append(
                f"  {feat:15s} = {val:8.3f}   importância: {bar} {imp:.3f}"
            )

        if result["low_confidence"]:
            lines.append("")
            lines.append(
                f"AVISO: Confiança abaixo do limiar ({CONFIDENCE_THRESHOLD:.0%}). "
                "Confirmar manualmente."
            )

        return "\n".join(lines)

    @property
    def metadata(self) -> dict:
        """Retorna os metadados do modelo (métricas de treinamento)."""
        return self._meta


# =============================================================================
#  TESTES AUTOMÁTICOS (executados com: python ml/model.py)
# =============================================================================

def _run_demos():
    """
    Testa o Classifier com casos conhecidos para validar que o modelo
    está funcionando corretamente após ser carregado do disco.
    """
    clf = Classifier()

    # Casos de teste com resultados esperados
    # Os valores foram escolhidos para simular features reais de cada classe
    test_cases = [
        {
            "name":     "Tubo Redondo — caso típico",
            "expected": "tubo_redondo",
            "features": {
                "circularity":  0.94,   # muito próximo de 1.0 → muito redondo
                "aspect_ratio": 1.02,   # w ≈ h → forma quadrada → tubo redondo de topo
                "holes":        1,      # furo central detectado
                "area":         2800.0, # área típica de tubo médio
                "is_hollow":    1,      # confirma que é oco
                "material":     1,      # inox
            }
        },
        {
            "name":     "Tubo Quadrado — caso típico",
            "expected": "tubo_quadrado_retangular",
            "features": {
                "circularity":  0.72,   # abaixo de 0.785 (quadrado perfeito) → quadrado
                "aspect_ratio": 0.99,   # quase quadrado (50x50mm)
                "holes":        0,      # câmera não capturou o furo (posição)
                "area":         3600.0, # área típica de tubo 50x50mm
                "is_hollow":    0,      # não detectado como oco
                "material":     0,      # aço carbono
            }
        },
        {
            "name":     "Tubo Retangular — 20x40mm",
            "expected": "tubo_quadrado_retangular",
            "features": {
                "circularity":  0.58,   # bem abaixo de 0.78 → muito retangular
                "aspect_ratio": 0.52,   # w=20, h=40 → 20/40 = 0.5
                "holes":        0,
                "area":         1900.0,
                "is_hollow":    0,
                "material":     0,
            }
        },
        {
            "name":     "Caso Ambíguo — circularity na fronteira",
            "expected": None,           # resultado esperado incerto
            "features": {
                "circularity":  0.80,   # exatamente entre redondo (>0.82) e quadrado (<0.78)
                "aspect_ratio": 1.00,
                "holes":        1,
                "area":         3100.0,
                "is_hollow":    1,
                "material":     0,
            }
        },
    ]

    print("=" * 65)
    print("  DEMOS DE INFERÊNCIA")
    print("=" * 65)

    for case in test_cases:
        result = clf.predict(case["features"])

        ok = "✓" if result["label"] == case["expected"] else "?"
        if case["expected"] is None:
            ok = "~"  # caso ambíguo — qualquer resultado é aceitável

        print(f"\n  {ok}  {case['name']}")
        print(f"     Predição  : {result['label_display']}")
        print(f"     Confiança : {result['confidence_pct']}% ", end="")

        # Barra visual de confiança
        pct = result["confidence_pct"]
        filled = int(pct / 5)
        bar = "█" * filled + "░" * (20 - filled)
        color_warn = " ⚠ BAIXA" if result["low_confidence"] else ""
        print(f"[{bar}] {color_warn}")

        print(f"     Mensagem  : {result['message']}")
        print(f"     Probabilidades por classe:")
        for cls, prob in result["all_probas"].items():
            p = round(prob * 100)
            b = "█" * int(p / 3)
            print(f"       {cls:35s}: {b:<20s} {p}%")

    # Teste da função explain()
    print()
    print("=" * 65)
    print("  EXPLICAÇÃO DO MODELO (caso: tubo redondo)")
    print("=" * 65)
    print()
    print(clf.explain(test_cases[0]["features"]))

    # Exibe metadados do modelo
    print()
    print("=" * 65)
    print("  METADADOS DO MODELO")
    print("=" * 65)
    for k, v in clf.metadata.items():
        print(f"  {k:25s}: {v}")


if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     INFERÊNCIA — CLASSIFICADOR DE PERFIS METÁLICOS      ║")
    print("║     Mecald / SENAI Pedro Leopoldo                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    _run_demos()
