"""
Qual problema essa classe resolve? 
    O que o sistema faz até agora: 
    Imagem -> Contornos -> Quantidade
    
    Mas estoque não trabalha com quantidade 
    Estoque trabalha com movimentações 
    
    Estoque trabalha com mudanças de estados
"""

import time 
from datetime import datetime

class EventManager:
    
    '''
    Responsável por detectar alterações confirmadas na quantidade de periflados 
    e transforma-las em eventos de movimentação de estoque.
    
    ---------------------------------------------------------------------------
    Essa classe pode ser considerada uma máquina de estados (State Machine)
    --------------------------------------------------------------------------- 
    Responsabilidade da classe: 
        - Receber contagem 
        - Comparar com o estado atual 
        - Detectar mudança 
        - Confirmar mudanças 
        - Gerar evento 
        - Registrar evento 
        
        Essa classe DEVE possuir memória. Ela precisa lembrar as coisas. 
        
        Ou seja, o que essa classe faz publicamente? 
        Atualiza o estado usando esta nova contagem
        
        Ela deve responder se houve um evento, ou não 
    '''
    
    def __init__(self, confirmation_time=5, logger=None):
        # Quantidade oficialmente aceita
        self.confirmed_count = None
        # Possível nova quantidade
        self.candidate_count = None
        # Momento que a mudança apareceu
        self.change_start_time = None
        # Tempo mínimo
        self.confirmation_time = confirmation_time
        # Logger (Ponto de referência para registro de eventos)
        self.logger = logger

    # Coordena o fluxo de geração de eventos 
    def update(self,current_count):        
        if not self._detect_change(current_count): 
            return None
        
        if not self._confirm_change(): 
            return None 
        
        event = self._create_event()
        self._update_state()
        self._log_event(event)
        
        return event

    # Verifica se existe mudança candidata 
    def _detect_change(self, current_count): 
        '''
        Analisa a nova contagem recebida, atualiza o estado interno da detecção de mudanças e 
        informa se existe uma mudança candidata aguradando confirmação 
        '''
    
        # Essa linha pergunta: Já existe um estado oficial? 
        if self.confirmed_count is None:
            return False
        
        # Nada mudou, mas já existe um valor confirmado 
        # Então os valores candidatos devem ser limpados 
        if current_count == self.confirmed_count: 
            self.candidate_count = None 
            self.change_start_time = None 
            return False

        # Primeira vez que estou vendo uma mudança 
        # Existe uma mudança candidata, mas ela não foi confirmada ainda 
        if self.candidate_count is None: 
            self.candidate_count = current_count 
            self.change_start_time = time.time()
            return True 
                
        # Mudança não estabilizada ainda 
        # Aplica "nova" mudança aos valores das variavéis 
        if current_count != self.candidate_count: 
            self.candidate_count = current_count
            self.change_start_time = time.time()
            return True
        
        '''
        Dentro dessa função já existe uma certa camada de aplicações redundantes, que evitam 
        o "erro do sistem" por assim dizer 
        '''
        # Candidato estabilizado e nada mudou desde o último frame        
        return True
        

    ''' 
    O que significa confirmar mudança? 
    Determinar se a mudança candidata possui evidências suficientes para ser considerada uma 
    movimentação real de estoque 
    
    O que ela deve retornar para conversar com o resto da classe? 
    Um valor True ou False. 
    
    Por que? 
    Porque esse método deve responder uma única pergunta: 
        " A mudança já pode ser aceita? " 
    '''
    def _confirm_change(self):
        # Nenhuma mudança sendo acompanhada 
        if self.change_start_time is None: 
            return False 
        
        elapsed_time = time.time() - self.change_start_time
        return elapsed_time >= self.confirmation_time
            

    def _update_state(self): 
        '''
        Atualiza o estado interno após uma mudança confirmada.     
        
        '''
    
        self.confirmed_count = self.candidate_count 
        self.candidate_count = None 
        self.change_start_time = None

    def _create_event(self): 
        '''
        Cria um dicionário representando um evento de movimentação 
        '''
        
        if self.candidate_count > self.confirmed_count: 
            event_type = "ENTRY"
        else: 
            event_type = "EXIT"
            
        return { 
                "timestamp": datetime.now().strftime("%d/%m/%Y %H:%M:%S"), 
                "event_type": event_type, 
                "previous_count": self.confirmed_count, 
                "current_count": self.candidate_count, 
                "difference": abs(self.candidate_count - self.confirmed_count)
                }  

    def _log_event(self,event): 
        '''
        Envia o evento para o logger, caso exista
        '''
        if self.logger is not None: 
            self.logger.save(event)
        
        