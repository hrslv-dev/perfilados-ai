# Uma imagem no openCV == numpy.ndarray
# O que é um array numpy? Uma matriz

# Exemplo de matriz grayscale: (Cinza em tons de preto e branco diferentes):
# [ 0,0,255 ]
# [ 0, 255, 255 ]
# Cada valor representa intenisdade luminosa

import cv2


# Entidade (captura da cêmra) -> capturar frames
class CameraCapture:
    # Assim que CameraCapture for inicializado a câmera será inicializada
    # Definindo a altura e largura padrões da imagem
    def __init__(self, source=1, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height

        self.cap = None

    # Conexão com a câmera utilizada
    def connect(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise Exception(f"Não foi possível conectar a camera")

        # Define a resolução de imagem usada
        # Com a troca de Câmera isso pode ser ACEITO, APROXIMADO, OU IGNORADO
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read_frame(self):

        # Validação de estado
        if self.cap is None:
            raise Exception("A camera não foi conectada")

        # Operação
        ret, frame = self.cap.read()

        # Validação do frame
        if not ret:
            raise Exception("Falha ao capturar frame")
        return frame

    # Limpeza de recursos
    def release(self):

        if self.cap is not None:
            self.cap.release()

        cv2.destroyAllWindows()
