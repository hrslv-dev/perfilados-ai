''' 
FIX: 
        ---------------------------------------------------------
Substituição do OTSU global para CLAHE + threshold adaptativo 
Por que isso resolveria o problema? 
Otsu assume histograma bimodal (fundo claro / objeto escuro bem separado)
    Em iluminação industrial, reflexo e fundo não são ideais
    Ou seja o OTSU FALHOU 


    -- BUSCAR ENTENDER MAIS SOBRE CLAHE e adaptativeThreshold -- 
    CLAHE + adaptativeThreshold opera localmente: cada região da imagem tem seu próprio 
    limiar calculado a partir da média local. 
    Resultado: O contorno aparece mesmo com sombra, reflexo ou fundo irregular

        -----------------------------------------------------------
'''



import cv2 
import numpy as np

class Preprocessor:
    
    def __init__(self,
                blur_kernel=(9, 9),
                # CLAHE -> equalização de histograma adaptativa local
                clahe_clip = 2.0, 
                clahe_grid = (8,8),
                # Threshold adaptativo: Tamanho do bloco local (deve ser impar)
                adaptative_block=31,
                adaptative_c=4,
                # Morfologia: limpa ruído sem destruir a forma 
                morph_kernel=5): 

        self.blur_kernel = blur_kernel
        self.clahe = cv2.createCLAHE(
            clipLimit=clahe_clip,
            titleGridSize=clahe_grid
        )

        self.adaptative_block = adaptative_block
        self.adaptative_c = adaptative_c
        
        # Kernel eliptico preserva melhor as borrdas de tubos circulares 
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, 
            (morph_kernel, morph_kernel)
        )


    # Aqui entra a imagem colorida em BGR 
    def to_grayscale(self,frame): 
        # Transformando BGR em intensidade luminosa 
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        return gray

    # Blur = suavização matemática
    # Coloca-se um blur para que ocorra uma suavização na imagem para reduzir o ruído 
    def apply_blur(self,gray_frame): 
        blurred = cv2.GaussianBlur(gray_frame, self.blur_kernel, 0)
        return blurred
    
    def apply_clahe(self,blurred): 
        # Equaliza contraste localmente (OTSU não faz isso)
        return self.clahe.apply(blurred)
    
    # Se pixel > threshold -> branco 
    # Se não -> preto 
    def apply_threshold(self, enhanced):
        # Threshold adaptativo : cada bloco de adapatative_block * adaptative_block px 
        # calcula seu próprio limiar pela média Gaussiana local
        # THRESH_BINARY_INV : objeto escuro em fundo claro -> branco 
        # Se o objeto for mais claro que o fundo usa THRESH_BINARY 
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            self.adaptative_block, 
            self.adaptative_c
            )
        return thresh
    
    def apply_morphology(self,binary): 
        # Abertura (erosão + dilatação): recome ruído sem encolher objetivo 
    
    
    def process(self, frame): 
        gray = self.to_grayscale(frame)
        blurred = self.apply_blur(gray)
        thresh = self.apply_threshold(blurred)
        return thresh