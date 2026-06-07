from ml.model import Classifier


clf = Classifier()
# Simulando features extraídas do mundo real: 

tubo_redondo = {
    "circularity": 0.92,
    "aspect_ratio": 1.01,
    "holes": 1,
    "area": 3200.0,
    "is_hollow": 1,
    "material": 0
}

# Simula features de tubo quadrado
tubo_quadrado = {
    "circularity": 0.71,
    "aspect_ratio": 0.98,
    "holes": 0,
    "area": 4100.0,
    "is_hollow": 0,
    "material": 1
}

# Simula features ambíguas (confiança baixa esperada)
ambiguo = {
    "circularity": 0.78,
    "aspect_ratio": 0.95,
    "holes": 1,
    "area": 2800.0,
    "is_hollow": 1,
    "material": 0
}

for nome, feats in [("Tubo Redondo", tubo_redondo), ("Tubo Quadrado", tubo_quadrado), ("Ambíguo", ambiguo)]:
    r = clf.predict(feats)
    print(f"\n[{nome}]")
    print(f"  Predição   : {r['label']}")
    print(f"  Confiança  : {r['confidence']:.1%}")
    print(f"  Baixa conf : {r['low_confidence']}")
    print(f"  Mensagem   : {r['message']}")
    print(f"  Probabilidades: {r['all_probas']}")   