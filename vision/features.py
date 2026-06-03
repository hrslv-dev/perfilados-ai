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
    def extract_features(self,contour, hierarchy): 
            area = cv2.contourArea(contour)

            perimeter = cv2.arcLength(contour, True)

            x,y,w,h = cv2.boundingRect(contour)

            aspect_ratio = w/h 

            circularity = (4 * 3.14159 * area) / (perimeter * perimeter)

            holes = self.count_holes(hierarchy)

            is_hollow = int(holes > 0)

            features = { 
                  "area": area,
                  "perimeter": perimeter, 
                  "width": w, 
                  "height": h, 
                  "aspect_ratio":aspect_ratio, 
                  "circularity": circularity,
                  "holes": holes,
                  "is_hollow" : is_hollow
            }

            return features
    
    # Conta todos os contornos filhos no frame inteiro 
    def count_holes(self,hierarchy): 
        # Responsabilidade : contar contornos internos 
        holes = 0
        if hierarchy is None: 
             return 0 

        for item in hierarchy[0]: 
             if item[3] != -1: 
                  holes+=1 
        return holes      
    
    # PROBLEMA AQUI:
        # Se a camera captura 3 objetos, cada um com seus buracos
        # count_holes retorna a soma de buracos de todos os objetos
        # Ou seja: um perfil sem buraco pode receber holes = 3 por causa de outros contornos de ruído 
