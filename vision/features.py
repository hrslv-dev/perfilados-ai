# Responsabilidade do features: 
    # Receber contornos 
    # Retornar dados tabulares 
import math
import cv2

class FeatureExtractor: 
    # Extrai as features do modelo: 
        # area
        # Perimetro 
        # altura, largura e cumprimento 
        # aspect_ratio (formato)
        # circularidade
    def extract_features(self,contour, hierarchy, all_contours=None): 
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x ,y ,w ,h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 1.0 
            
            # FIX HERE: 
            if perimeter > 0: 
                 circularity = (4 * math.pi * area) / (perimeter ** 2)

            else: 
                 circularity = 0.0

            # FIX HERE: 
                # passa all_contours para count_holes poder filtrar por área 
            holes = self.count_holes(hierarchy, all_contours)
            is_hollow = int(holes > 0)

            features = { 
                  "circularity": circularity,
                  "aspect_ratio":aspect_ratio, 
                  "holes": holes,
                  "area": area,
                  "is_hollow" : is_hollow,
                  "perimeter": perimeter, 
                  "x": x,
                  "y": y,
                  "width": w, 
                  "height": h
            }

            return features
    
    # Conta todos os contornos filhos no frame inteiro 
    def count_holes(self,hierarchy, all_contours=None, min_hole_area=100): 
    # FIX Antes: Contava todos os contornos filhos no frame inteiro
        # AGORA: Agora só conta um contorno como buraco se: 
        # Tem pai (hierarchy [i][3] != -1) -> é filho/interno
        # Tem área >= min_hole_area - não é ruido de reflexo 

        holes = 0
        if hierarchy is None: 
             return 0 
        for i, item in enumerate(hierarchy[0]):  
             if item[3] != -1: # Tem pai -> buraco
                if all_contours is not None: 
                    hole_area = cv2.contourArea(all_contours[i])
                    if hole_area >= min_hole_area: 
                         holes+=1
                else: 
                    # fallback sem filtro de area  
                    holes+=1

        return holes      
    

    # PROBLEMA AQUI:
        # Se a camera captura 3 objetos, cada um com seus buracos
        # count_holes retorna a soma de buracos de todos os objetos
        # Ou seja: um perfil sem buraco pode receber holes = 3 por causa de outros contornos de ruído
         
        # O QUE DEVE SER FEITO: 
        # Contar apenas buracos de contornos que tem a area significativa
