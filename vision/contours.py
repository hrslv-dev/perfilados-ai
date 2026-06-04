# Ponte entre imagem e matemática
# pixels (preprocessemt) -> Objetos geométricos

import cv2

# Responsabilidade da classe : 
    # Detectar contornos 
    # Filtrar cotornos ruins 
    # Calcular bounding box 
    # Detectar contornos internos 
class ContourDetector:

    def __init__(self,min_area=2000): 
        self.min_area = min_area

    def find_contours(self,binary_image): 
        
        ''' 
        -------------------------------------------------------------
        min_area : filtro de area mínima em px2
        - Original : 500 -> capta ainda muito ruido de fundo 
        - Agora : 2000 -> descarta ruído pequeno, mantém perfis reais 
        Teste na segunda-feira: 
        Se o perfil parecer desaparecer -> reduzar a min_area para 1000
        Se ainda houver ruído aumentar para 4000
        -------------------------------------------------------------
        '''

        contours, hierarchy = cv2.findContours(
            binary_image,
            cv2.RETR_TREE,  
            cv2.CHAIN_APPROX_SIMPLE # SIMPLIFIQUE PONTOS REDUNDANTES 
        )
        filtered_contours = [c for c in contours if cv2.contourArea(c) > self.min_area]
        # Retorna contornos filtrados | hierarquia completa | todos os contornos 
        # hierarchy e all_contours devem usar os mesmos índices para count_holes 
        return filtered_contours, hierarchy, contours
    
    def find_largest(self,binary_image): 
        ''' 
        Retorna apenas o maior contorno 
        Útil quando o perfil é o objeto principal da cena
        *Usar no pipeline de inferência do modelo ML 
        (não no teste que queremos ver tudo)
        '''
        filtered, hierarchy, all_contours = self.find_contours(binary_image)
        if not filtered: 
            return None, hierarchy, all_contours
        largest = max(filtered,key=cv2.contourArea)
        return largest, hierarchy, all_contours

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
