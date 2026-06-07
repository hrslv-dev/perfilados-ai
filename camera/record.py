from camera.capture import CameraCapture
from vision.preprocessor import Preprocessor  # FIX: arquivo renomeado
from vision.contours import ContourDetector
from vision.features import FeatureExtractor
from vision.renderer import Renderer

import cv2


class Recorder:

    def start_recording():
        camera = CameraCapture()
        preprocessor = Preprocessor()  # FIX: agora usa CLAHE + adaptativo
        contour_detector = ContourDetector()
        extractor = FeatureExtractor()
        renderer = Renderer()  # FIX: Renderer agora é usado

        camera.connect()

        frame_count = 0

        while True:
            frame = camera.read_frame()
            processed = preprocessor.process(frame)
            contours, hierarchy, all_contours = contour_detector.find_contours(
                processed
            )

            for contour in contours:
                features = extractor.extract_features(
                    contour, hierarchy, all_contours=all_contours
                )
                frame_count += 1
                if frame_count <= 100:

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
