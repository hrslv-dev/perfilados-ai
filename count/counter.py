"""
Essa classe desacopla a origem da origem da contagem, tornando possível 
escalar isso para um modelo YOLO de capta quantidade de algo especifico. 
"""


"""
- Por que essa classe existe? 
    Serve para fornecer ao restante do sistema uma representação confiável da quantidade de 
    perfilados detectados na imagem. 
    
    Mas o que EXATAMENTE ela deve fazer? 
        1 . Receber contornos 
        2 . Conta-los 
        3 . Validar contagem 
        4 . Fornecer uma interface simples para outras classes

- Por que ela não pertence ao EventManager? 
    Porque EventManager possui como responsabilidade apenas a camada da arquitetura que trata
    da geração e manipulação de eventos. Ou seja, a responsabilidade dessa classe não condiz com 
    a função dessa classe ProfileCounter. 
- Por que ela não pertence ao ContourDetector?
    Porque ContourDetector é responsável apenas por detectar contornos, ou seja, receber a imagem 
    binária, detectar contornos dentro dessa imagem, e retorna-los para outras classes. 
- Qual princípio de orientação objeto estamos aplicando? 
    O principal princípo é o S do SOLID de Robert Cecil Marin, que significa Single Responsibility Principle. 
    Este principio diz respeito a que toda classe, função, objeto, variável, deve receber uma única responsabilidade
    clara e compreensível . 
"""

''' 
--------------------------------------------------------------------------------
Responsável por transformar contornos detectados em uma quantidade de perfilados. 
--------------------------------------------------------------------------------
'''

class ProfileCounter:

    '''
    Qual será a interface pública da classe? 
        O que as outras classes precisam pedir ao ProfileCounter? 
            "Conte quantos perfilados existem" 
   
    Como ela seria usada EXTERNAMENTE: 
        counter = ProfileCounter()
        count = counter.count(filtered_contours)
    
    A classe ProfileCounter é Stateless (não precisa de possuir memória)
    Ela só precisa receber um valor -> realizar algo com ele -> retornar outro valor 
    Sem precisar "lembrar" o valor que ela retornou antes ou o que chegou antes s

    '''
    def __init__(self):
         pass 
     
    # Maestro (método pública)
    def count(self,contours):    
        self._validate_input(contours)
        return self._count_profiles(contours)
                
    # Valida o valor do input (método interno)
    def _validate_input(self,contours): 
        if contours is None: 
            raise ValueError("A lista de contornos não pode ser None")
        '''
            Por que mandar ValueError aqui? 
                Porque o valor ser None, majoritariamente acontece por erro de programação 
                então o melhor a se fazer é falhar diretamente. 
        '''
        if not hasattr(contours,"__len__"): 
            raise TypeError("O objeto recebido não pode ser contado")
        '''
            Aqui o que acontece é uma confirmação de que se o objeto recebido no input, pode 
            ser contado, caso não seja, retorna erro. 
        '''
        
    # Conta realmente os perfis (método interno)
    def _count_profiles(self, contours):   
        return len(contours)
        