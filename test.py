
from camera.capture import CameraCapture
from vision.preprocessor import Preprocessor
from vision.contours import ContourDetector
from vision.features import FeatureExtractor

import cv2;
camera = CameraCapture()
preprocessor_noOtsu = Preprocessor(use_otsu=False)
preprocessor = Preprocessor()
contour_detector = ContourDetector()
extractor = FeatureExtractor()

camera.connect()

frame_count = 0
# -------#


# TESTES: 
    # Girar o perfilado quadrado por exemplo para validar as features
    #   Caso mantenham-se constantes -> features confiáveis

    # Distanciar o objeto da câmera -> Aspect Ratio e circularity não devem mudar
            # Enquanto area MUDA MUITO 

    # Teste de discriminação: 
        # Classificar 4 geometrias diferentes
        # Anotar o aspect ratio, circularity e holes
        # Montar tabela 
    # DEFINIR SE AS CLASSES REALMENTE SE SEPARAM (pois caso o limiar de diferença entre uma classe para outra seja muito
    # baixo, o modelo pode ter dificuldade)
    # Estatśitica: 
        # Captar 100 frames. 
        # Calcular média, mínimo, máximo, desvio padrão de cada feature
        # Se desvio padrão for pequeno = feature confiável 
        # OPCIONAL : Montar tabela de média e desvio padrão de cada feature 
        
# RESULTADOS: 
# -- #
    
while True: 
    frame = camera.read_frame()
    
    processed = preprocessor.process(frame)
    # Teste com no Otsu para ver a diferença    
    processed_noOtsu = preprocessor_noOtsu.process(frame)
    contours, hierarchy = contour_detector.find_contours(processed)

    for contour in contours: 
        area = cv2.contourArea(contour)

        features = extractor.extract_features(contour, hierarchy)
        frame_count +=1

        if frame_count <= 100: 
            print(len(contours))

        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = features["aspect_ratio"]
        circularity = features["circularity"]
        holes = features["holes"]
        is_hollow = features["is_hollow"]

        cv2.drawContours(frame,[contour], -1,(0,255,0),2)
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(frame,f"Area: {int(area)}",
                    (x,y -10),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (0,255,0),
                    2)
        cv2.putText(frame, f"AR: {aspect_ratio:.2f}",
                    (x,y - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,
                    (0,255,255),
                    2)
        cv2.putText(frame,f"Circ: {circularity:.2f}", 
                    (x,y - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,0), 
                    2)
        cv2.putText(frame,f"Holes:{holes}", 
                    (x,y -80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6,
                    (255,255,0),
                    2)
        cv2.putText(frame,f"Is_hollow: {is_hollow}",
                    (x,y - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255,255,0), 
                    2) 
            
    
    cv2.imshow("Industrial Vision", frame)    
    cv2.imshow("Threshhold image", processed)


    if cv2.waitKey(1) == 27: 
        break 

camera.release()