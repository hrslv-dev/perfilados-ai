import numpy as np
import pandas as pd

"""
Essa classe é responsável por coletar e armazenar amostras de features para análise estatística.
Ou seja, coleta as features de cada frame e armazena para posterior análise.
"""
class SessionCollector:
    def __init__(self, target_samples):
        self.target_samples = target_samples
        self.samples = []

    def add(self, features):
        self.samples.append(features)

    def is_complete(self):
        return len(self.samples) >= self.target_samples

    def count(self):
        return len(self.samples)

    def get_samples(self):
        return self.samples


"""
Essa classe é responsável por agregar as amostras coletadas e contruir uma lista com as features agregadas estatísticamente.
Utiliza-se média e desvio padrão para cada feature coletada
"""


class StatisticAggregator:
    def build_feature_vector(self, samples):
        feature_vector = {}

        feature_vector["circularity"] = (
            sum(s["circularity"] for s in samples)/ len(samples)
        )
        feature_vector["aspect_ratio"] = (
            sum(s["aspect_ratio"] for s in samples) / len(samples)
        )
        feature_vector["holes"] = round(
            sum(s["holes"] for s in samples) / len(samples)
        )
        feature_vector["area"] = (
            sum(s["area"] for s in samples) / len(samples)
        )
        feature_vector["is_hollow"] = round(
            sum(s["is_hollow"] for s in samples)/ len(samples)
        )
        return feature_vector
"""
Essa classe é responsável por fazer a predição com base nas features agregadas.
Ela usa o modelo de Random Forest treinado para fazer as predições
(model = clf())
"""


class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, feature_vector: dict, material_id: int):
        if material_id is None:
            raise ValueError(
                "Material não informado"
            )
        if material_id < 0 or material_id > 2:
            raise ValueError(
                "Material Inváldio"
            )

        result = self.model.predict(feature_vector)
        prediction = result["label"]
        confidence = result["confidence_pct"]
        is_confiable = result["low_confidence"]
        message = result["message"]
        return prediction, confidence, is_confiable, message
