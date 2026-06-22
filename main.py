"""
test.py — Loop de validação de features em tempo real.

FIX aplicado:
  - Usa Renderer em vez de duplicar lógica de rendering inline
  - Usa novo Preprocessor (CLAHE + adaptativo) em vez de Otsu
  - Corrigido formato :.2f em holes (era int)
  - extract_features agora recebe all_contours para count_holes correto
"""

import time

import cv2

from aggregation.aggregator import (
    PredictionService,
    SessionCollector,
    StatisticAggregator,
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

material_id = int(input("Material do perfilado\n0=carbono\n1=Inox\n2=Alumínio\n"))
camera.connect()

start_time = time.time()
capture_delay = 15  # segundos

samples_count = 0
prediction_done = False

while True:
    frame = camera.read_frame()
    if frame is None:
        continue
    processed = preprocessor.process(frame)
    largest_contour, hierarchy, all_contours = contour_detector.find_largest(processed)

    elapsed = time.time() - start_time
    if elapsed >= capture_delay:
        if largest_contour is not None:
            features = extractor.extract_features(
                largest_contour, hierarchy, all_contours=all_contours
            )
            renderer.draw_complete_overlay(frame, features)

            # Começa a coletar as features aqui
            if not collector.is_complete():
                collector.add(features)
                print(
                    f"[{samples_count:03d}] |"
                    f"circ={features['circularity']:.3f} | "
                    f"ar={features['aspect_ratio']:.3f} | "
                    f"holes={features['holes']} | "  # FIX: sem :.2f
                    f"hollow={features['is_hollow']}"
                    f"area={features['area']:.0f} | "
                )
                samples_count += 1
    else:
        remaining = capture_delay - elapsed
        cv2.putText(
            frame,
            f"Waiting... ({remaining:.1f}s)",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
    cv2.imshow("Industrial Vision", frame)
    cv2.imshow("Threshold (CLAHE + Adaptativo)", processed)

    if collector.is_complete() and not prediction_done:
        print("Processando previsão do modelo....")
        print(".................")
        samples = collector.get_samples()
        feature_vector = aggregator.build_feature_vector(samples)

        print("Feature Vector: ")
        print(feature_vector)
        print("-------------------")
        print("Classificando...")
        prediction, confidence, is_confiable, message = predictor.predict(
            feature_vector, material_id
        )

        print(f"Classificação: {prediction}")
        print("=================================")
        print(f"Confiança: {confidence}")
        print("=================================")
        print(f"É pouco confiável: {is_confiable}")
        print("=================================")
        print(f"Mensagem: \n{message}")
        prediction_done = True
        print("Predição realizada com sucesso ! ")
        print("===============||================")

    if cv2.waitKey(1) == 27:  # ESC para sair
        break
camera.release()
cv2.destroyAllWindows()
