# Uma imagem no openCV == numpy.ndarray
# O que é um array numpy? Uma matriz

# Exemplo de matriz grayscale: (Cinza em tons de preto e branco diferentes): 
# [ 0,0,255 ]
# [ 0, 255, 255 ]
# Cada valor representa intenisdade luminosa 

import cv2

# Entidade (captura da cêmra)
class CameraCapture: 
    # Assim que CameraCapture for inicializado a câmera será inicializada
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)        
        # Se cap não estiver aberto manda exceção. 
        if not self.cap.isOpened(): 
            raise Exception("Not possible to open camera")
        
    
