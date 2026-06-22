from aggregation.aggregator import (
    PredictionService,
    SessionCollector,
    StatisticAggregator,
)
from ml.model import Classifier
from vision.contours import ContourDetector
from vision.features import FeatureExtractor
from vision.preprocessor import Preprocessor

preprocessor = Preprocessor()
contour_detector = ContourDetector()
extractor = FeatureExtractor()
classifier = Classifier()
collector = SessionCollector(target_samples=60)
predictor = PredictionService(model=classifier)
aggregator = StatisticAggregator()


class Analyzer:
    def __init__(self, samples_target):
        self.preprocessor = Preprocessor()
        self.contour_detector = ContourDetector()
        self.extractor = FeatureExtractor()
        self.classifier = Classifier()
        self.collector = SessionCollector(target_samples=samples_target)

    def analyze(self, frame, material_id):
        samples_count = 0
        prediction_done = False

        processed = preprocessor.process(frame)
        largest_contour, hierarchy, all_contours = contour_detector.find_largest(
            processed
        )

        if largest_contour is not None:
            features = extractor.extract_features(
                largest_contour, hierarchy, all_contours=all_contours
            )
            if not collector.is_complete():
                collector.add(features)
                samples_count += 1

        if collector.is_complete() and not prediction_done:
            samples = collector.get_samples()
            feature_vector = aggregator.build_feature_vector(samples)
            prediction, confidence, is_confiable, message = predictor.predict(
                feature_vector, material_id
            )
            return prediction, confidence, is_confiable, message
        return None

        # Cabelo de pelé não cresce
