import time

import cv2

from camera.capture import CameraCapture
from services.analyzer import (
    Analyzer,  # Ajuste o import conforme sua estrutura de pastas
)
from vision.renderer import Renderer

# Inicialização
camera = CameraCapture()
renderer = Renderer()
analyzer = Analyzer(samples_target=100)

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

    elapsed = time.time() - start_time

    # Se passou do tempo de espera e a predição ainda não foi feita
    if elapsed >= capture_delay and not prediction_done:
        # O Analyzer faz todo o trabalho de visão, extração e coleta
        result = analyzer.analyze(frame, material_id)

        processed_frame = result["processed_frame"]
        features = result["features"]

        # Se encontrou features neste frame, desenha na tela e imprime no terminal
        if features is not None:
            renderer.draw_complete_overlay(frame, features)

            # Imprime no terminal apenas durante a fase de coleta
            if not result["ready"]:
                print(
                    f"[{samples_count:03d}] | "
                    f"circ={features['circularity']:.3f} | "
                    f"ar={features['aspect_ratio']:.3f} | "
                    f"holes={features['holes']} | "
                    f"hollow={features['is_hollow']} | "
                    f"area={features['area']:.0f} | "
                )
                samples_count += 1

        # Se o Analyzer retornou "ready=True", significa que acumulou as amostras e fez a predição
        if result["ready"]:
            print("\nProcessando previsão do modelo....")
            print(".................")
            print("Classificando...")
            print(f"Classificação: {result['prediction']}")
            print("=================================")
            print(f"Local: {result['icon']}")
            print("==================================")
            print(f"Prateleira : {result['shelf']}")
            print("===================================")
            print(f"Confiança: {result['confidence']}")
            print("=================================")
            print(f"É pouco confiável: {result['is_confiable']}")
            print("=================================")
            print(f"Mensagem: \n{result['message']}")
            print("Predição realizada com sucesso ! ")
            print("===============||================\n")

            prediction_done = True

        # Mostra a janela do Threshold mesmo se não encontrou features
        cv2.imshow("Threshold (CLAHE + Adaptativo)", processed_frame)

    # Se ainda está no período de delay inicial
    elif elapsed < capture_delay:
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

    # Mostra a janela principal com o frame original/desenhado
    cv2.imshow("Industrial Vision", frame)

    if cv2.waitKey(1) == 27:  # ESC para sair
        break

camera.release()
cv2.destroyAllWindows()
