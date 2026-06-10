import numpy as np
import pandas as pd

"""
Essa classe é responsável por coletar e armazenar amostras de features para análise estatística.
Ou seja, coleta as features de cada frame e armazena para posterior análise.
"""


class SessionColecter:
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


class StatisticAgregator:
    def build_feature_vector(self, samples):
        df = pd.DataFrame(samples)
        final_features = {}

        for column in df.columns:
            final_features[f"{column}_mean"] = df[column].mean()
            final_features[f"{column}_std"] = df[column].std()

        return final_features


"""
Essa classe é responsável por fazer a predição com base nas features agregadas.
Ela usa o modelo de Random Forest treinado para fazer as predições
(model = clf())
"""


class PredictionService:
    def __init__(self, model):
        self.model = model

    def predict(self, feature_vector):
        X = pd.DataFrame([feature_vector])
        prediction = self.model.predict(X)[0]

        confidence = np.max(self.model.predict_proba(X))
        return prediction, confidence
