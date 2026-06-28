"""
O sistema precisa saber:
    Estado anterior
        |
    Estado atual
        |
    Diferença
        |
    Evento
"""

"""
Não adianta fazer algo como:
    if actual != previous:
        register_event()
Porque a câmera pode acabar oscilando bastante, e
como o foco é visão computacional para contagem de materiais,
fica inconsistente esse modo de registro.
"""

"""
Redundância:
    Deve ser aplicado um modelo de redundância utilizando tempo,
    caso após certa quantidade de tempo o estado de mudança não tenha se
    mantido, o evento é descartado.
"""


class EventManager:
    def __init__(self):
        # Quantidade oficialmente aceita
        self.confirmed_count = None
        # Possível nova quantidade
        self.candidate_count = None
        # Momento que a mudança apareceu
        self.change_start_time = None
        # Tempo mínimo
        self.confirmation_time = 3
