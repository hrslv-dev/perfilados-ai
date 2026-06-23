from aggregation.aggregator import (
    PredictionService,
    SessionCollector,
    StatisticAggregator,
)
from ml.model import Classifier
from vision.contours import ContourDetector
from vision.features import FeatureExtractor
from vision.preprocessor import Preprocessor


class Analyzer:
    def __init__(self, samples_target):
        self.preprocessor = Preprocessor()
        self.contour_detector = ContourDetector()
        self.extractor = FeatureExtractor()
        self.classifier = Classifier()
        self.collector = SessionCollector(target_samples=samples_target)
        self.aggregator = StatisticAggregator()
        self.predictor = PredictionService(self.classifier)

    def analyze(self, frame, material_id):
        processed = self.preprocessor.process(frame)
        largest_contour, hierarchy, all_contours = self.contour_detector.find_largest(
            processed
        )
        features = None

        if largest_contour is not None:
            features = self.extractor.extract_features(
                largest_contour, hierarchy, all_contours=all_contours
            )
            if not self.collector.is_complete():
                self.collector.add(features)

        # Se atingiu o alvo faz a predição:
        if self.collector.is_complete():
            samples = self.collector.get_samples()
            feature_vector = self.aggregator.build_feature_vector(samples)
            prediction, confidence, is_confiable, message = self.predictor.predict(
                feature_vector, material_id
            )
            self.collector.reset()
            return {
                "ready": True,
                "prediction": prediction,
                "confidence": confidence,
                "is_confiable": is_confiable,
                "message": message,
                "features": features,
                "processed_frame": processed,
            }
        self.collector.reset()
        # Retorna o status atual se ainda não estiver pronto
        return {
            "ready": False,
            "prediction": None,
            "features": features,
            "processed_frame": processed,
        }

        # Cabelo de pelé não cresce
