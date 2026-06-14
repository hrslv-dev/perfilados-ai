"""
test.py — Loop de validação de features em tempo real.

FIX aplicado:
  - Usa Renderer em vez de duplicar lógica de rendering inline
  - Usa novo Preprocessor (CLAHE + adaptativo) em vez de Otsu
  - Corrigido formato :.2f em holes (era int)
  - extract_features agora recebe all_contours para count_holes correto
"""

import cv2

from aggregation.aggregator import (
    PredictionService,
    SessionCollector,
    StatisticAggregator
)
from camera.capture import CameraCapture
from ml.model import Classifier
from vision.contours import ContourDetector
from vision.features import FeatureExtractor
from vision.preprocessor import Preprocessor  # FIX: arquivo renomeado
from vision.renderer import Renderer

camera = CameraCapture()
preprocessor = Preprocessor()  # FIX: agora usa CLAHE + adaptativo
contour_detector = ContourDetector()
extractor = FeatureExtractor()
renderer = Renderer()
classifier = Classifier()
collector = SessionCollector(target_samples=60)
predictor = PredictionService(model=classifier)
aggregator = StatisticAggregator()


"""
Pipeline Completo;

    1. Captura do frame da câmera
    2. Pré-Processamento (CLAHE + adaptativo)
    3. Detecção de contornos
    4. Extração de features
    5. Predição de classe
    6. Renderização do resultado
"""
"""
Loop principal que chama as classes de coleta, agregação e predição
"""


camera.connect()
samples_count = 0
prediction_done = False

while True:
    frame = camera.read_frame()
    if frame is None:
        continue
    processed = preprocessor.process(frame)
    contours, hierarchy, all_contours = contour_detector.find_contours(processed)

    # collector estava ligado aos contornos e não aos frames, agora com essa alteração o foco vai mais para o objeto e não
    # para o contorno em si

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        features = extractor.extract_features(
            largest_contour, hierarchy, all_contours=all_contours
        )
        renderer.draw_complete_overlay(frame, features)

        if samples_count < 60:
            collector.add(features)
            print(
                f"[{samples_count:03d}] |"
                f"contornos={len(contours)} | "
                f"circ={features['circularity']:.3f} | "
                f"ar={features['aspect_ratio']:.3f} | "
                f"holes={features['holes']} | "  # FIX: sem :.2f
                f"hollow={features['is_hollow']}"
                f"area={features['area']:.0f} | "
            )
            samples_count += 1

    cv2.imshow("Industrial Vision", frame)
    cv2.imshow("Threshold (CLAHE + Adaptativo)", processed)

    if collector.is_complete() and not prediction_done:
        samples = collector.get_samples()
        feature_vector = aggregator.build_feature_vector(samples)
        result = predictor.predict(feature_vector)
        prediction = result["label"]
        low_confidence = result["low_confidence"]
        confidence = result["confidence_pct"]

        print(f"Classificação: {prediction} \n")
        print("=================================")
        print(f"Confiança: {confidence}")
        print("=================================")
        print(f"É confiável: {low_confidence}")
        print("=================================")
        prediction_done = True

    if cv2.waitKey(1) == 27:  # ESC para sair
        break
camera.release()
cv2.destroyAllWindows()
