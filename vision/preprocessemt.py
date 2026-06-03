import cv2 

# Classe do preprocessador 
    # Parâmetros 
    # Configurações
    # Comportamento reutilizavel 
class Preprocessor:

    # Inicialização dos parâmetros
        # Controlam o comportamento matemático do preprocessamento 
    def __init__(self,
                 # Controla a intensidade do blur da imagem
                 blur_kernel=(9, 9),
                 # Separação entre preto e branco
                 threshold_value=120,
                 use_otsu=True):
        self.blur_kernel = blur_kernel
        self.threshold_value = threshold_value
        self.use_otsu = use_otsu

    # Aqui entra a imagem colorida em BGR 
    def to_grayscale(self,frame): 
        # Transformando BGR em intensidade luminosa 
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        return gray

    # Blur = suavização matemática
    # Coloca-se um blur para que ocorra uma suavização na imagem para reduzir o ruído 
    def apply_blur(self,gray_frame): 
        blurred = cv2.GaussianBlur(
            gray_frame, 
            self.blur_kernel, 
            0
        )
        return blurred
    
    # Se pixel > threshold -> branco 
    # Se não -> preto 
    def apply_threshold(self,blurred_frame):

        if self.use_otsu:  
            _, thresh = cv2.threshold(
                blurred_frame,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU  
                )
        else: 
            _, thresh = cv2.threshold(
                blurred_frame, 
                self.threshold_value, 
                255,
                cv2.THRESH_BINARY
            )
        return thresh

    def process(self, frame): 
        gray = self.to_grayscale(frame)
        blurred = self.apply_blur(gray)
        thresh = self.apply_threshold(blurred)
        return thresh