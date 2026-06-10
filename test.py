"""
test.py — Loop de validação de features em tempo real.

FIX aplicado:
  - Usa Renderer em vez de duplicar lógica de rendering inline
  - Usa novo Preprocessor (CLAHE + adaptativo) em vez de Otsu
  - Corrigido formato :.2f em holes (era int)
  - extract_features agora recebe all_contours para count_holes correto
"""

import cv2

from agregation.agregator import PredictionService, SessionColecter, StatisticAgregator
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
collecter = SessionColecter(target_samples=60)
predictor = PredictionService(model=classifier)
agregator = StatisticAgregator()

camera.connect()
frame_count = 0

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
while True:
    frame = camera.read_frame()
    processed = preprocessor.process(frame)
    contours, hierarchy, all_contours = contour_detector.find_contours(processed)

    for contour in contours:
        features = extractor.extract_features(
            contour, hierarchy, all_contours=all_contours
        )

        frame_count += 1
        if frame_count <= 60:
            collecter.add(features)
            print(
                f"[{frame_count:03d}] |"
                f"contornos={len(contours)} | "
                f"area={features['area']:.0f} | "
                f"circ={features['circularity']:.3f} | "
                f"ar={features['aspect_ratio']:.3f} | "
                f"holes={features['holes']} | "  # FIX: sem :.2f
                f"hollow={features['is_hollow']}"
            )

        renderer.draw_complete_overlay(frame, features)

    cv2.imshow("Industrial Vision", frame)
    cv2.imshow("Threshold (CLAHE + Adaptativo)", processed)

    if collecter.is_complete():
        samples = collecter.get_samples()
        features_vector = agregator.build_feature_vector(samples)


        if cv2.waitKey(1) == 27:  # ESC para sair
        break

camera.release()
