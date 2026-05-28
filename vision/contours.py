# Ponte entre imagem e matemática
# pixels (preprocessemt) -> Objetos geométricos

import cv2

# Responsabilidade da classe : 
    # Detectar contornos 
    # Filtrar cotornos ruins 
    # Calcular bounding box 
    # Detectar contornos internos 
class ContourDetector:

    def find_contours(self,binary_image): 
        
        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL, # PEGUE APENAS CONTORNOS EXTERNOS 
            cv2.CHAIN_APPROX_SIMPLE # SIMPLIFIQUE PONTOS REDUNDANTES 
        )
        return contours # Lista de contornos 
        # Sequência de pontos geométricos (x1, y1).. 
    
    # Desenha linhas, bordas sobre a imagem
    def draw_contours(
            self,
            frame, 
            contours
    ): 
        cv2.drawContours(
            frame,
            contours, 
            -1, # Desenha todos os contornos 
            (0,255,0), # Cor verde 
            2 ) # Espessura da linha 
        return frame
