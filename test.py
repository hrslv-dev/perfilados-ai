"""
test.py — Loop de validação de features em tempo real.

FIX aplicado:
  - Usa Renderer em vez de duplicar lógica de rendering inline
  - Usa novo Preprocessor (CLAHE + adaptativo) em vez de Otsu
  - Corrigido formato :.2f em holes (era int)
  - extract_features agora recebe all_contours para count_holes correto
"""

from camera.capture import CameraCapture
from vision.preprocessor import Preprocessor  # FIX: arquivo renomeado
from vision.contours import ContourDetector
from vision.features import FeatureExtractor
from vision.renderer import Renderer

import cv2

camera = CameraCapture()
preprocessor = Preprocessor()  # FIX: agora usa CLAHE + adaptativo
contour_detector = ContourDetector()
extractor = FeatureExtractor()
renderer = Renderer()  # FIX: Renderer agora é usado

camera.connect()

frame_count = 0

# TESTES a realizar:
#   1. Girar o perfilado → aspect_ratio e circularity devem se manter estáveis
#   2. Distanciar da câmera → area muda, circularity e aspect_ratio NÃO devem mudar
#   3. Testar com 2 geometrias diferentes → confirmar separação das classes
#   4. Estatística: 100 frames → média e desvio padrão de cada feature
#      Se desvio padrão for pequeno = feature confiável para o modelo

while True:
    frame = camera.read_frame()
    processed = preprocessor.process(frame)
    contours, hierarchy, all_contours = contour_detector.find_contours(processed)

    features_history = []

    for contour in contours:
        # FIX: passa all_contours originais para count_holes usar o mesmo índice do hierarchy
        features = extractor.extract_features(
            contour, hierarchy, all_contours=all_contours
        )
        
        frame_count += 1
        if frame_count <= 100:
            features_history.append(features)
            print(
                f"[{frame_count:03d}] "
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

    if cv2.waitKey(1) == 27:  # ESC para sair
        break

camera.release()
