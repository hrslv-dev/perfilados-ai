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
    '''
    ----------------------------------------------------------------------------------------------------------
    KERNEL -> O que é um kernel nesse contexto? É uma matriz númerica que é usada para transformar 
    a imagem. Cada pixel da imagem é um valor em uma matriz númerica. Ao se aplicar um kernel (outra matriz)
    é feito uma multiplicação da matriz que representa a imagem com a matriz kernel, o que gera novos valores 
    para a matriz de imagem, ou seja, a imagem se modifica. 
    ----------------------------------------------------------------------------------------------------------

    '''

    def __init__(self,
                blur_kernel=(9, 9),
                clahe_clip = 2.0, 
                clahe_grid = (8,8),
                # CLAHE -> equalização de histograma adaptativa local
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

    '''
    ----------------------------------------------------------------------------------------------------------
    GaussianBlur -> É um filtro de suavização, o que reduz os ruídos presentes na imagem, 
    é como se os pixels que diferem muito dos outros fossem cortados.
    Ele parte da ideia que pixels mais centralizados tem maior peso que os pixels nas extremidades, então
    é aplicado a imagem o seguinte comando matemático: 
    1/16 = [1 2 1]  -> essa matriz define que os pixels mais alinhados ao centro tem maior peso
           [2 4 1]
           [1 2 1] 
    Essa matriz surge da curva em forma de sino -> f(x) = e -x2
    ----------------------------------------------------------------------------------------------------------
    '''

    # Blur = suavização matemática
    # Coloca-se um blur para que ocorra uma suavização na imagem para reduzir o ruído 
    def apply_blur(self,gray_frame): 
        blurred = cv2.GaussianBlur(gray_frame, self.blur_kernel, 0)
        return blurred
    

    ''' 
    ----------------------------------------------------------------------------------------------------------
    CLAHE -> Clahe divide a imagem em pequenos blocos e cada bloco é tratado separadamente 
    contraste é a diferença entre pixels claros e escuros, e o clahe aumenta o contraste de cada região 
    individualmente. 
    ----------------------------------------------------------------------------------------------------------
    '''    

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
        # Abertura (erosão + dilatação): remove ruído sem encolher objeto 
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
        # Fechamento (dilatação + erosão) : fechab buracos de reflexo no metal
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self.kernel)
        return cleaned    
    
    def process(self, frame): 
        ''' Pipeline completo. Retorna imagem binária pronta para findCountors'''
        gray = self.to_grayscale(frame)
        blurred = self.apply_blur(gray)
        enhanced = self.apply_clahe(blurred)
        binary = self.apply_threshold(enhanced)
        cleaned = self.apply_morpholog(binary)
        return cleaned