# Responsabilidade do features: 
    # Receber contornos 
    # Retornar dados tabulares 
import cv2

class FeatureExtractor: 
    # Extrai as features do modelo: 
        # area
        # Perimetro 
        # altura, largura e cumprimento 
        # aspect_ratio (formato)
        # circularidade
    def extract_features(self,contour): 
            area = cv2.contourArea(contour)

            perimeter = cv2.arcLength(contour, True)

            x,y,w,h = cv2.boundingRect(contour)

            aspect_ratio = w/h 

            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

            features = { 
                  "area": area,
                  "perimeter": perimeter, 
                  "width": w, 
                  "height": h, 
                  "aspect_ratio": aspect_ratio, 
                  "circularity": circularity
            }

            return features