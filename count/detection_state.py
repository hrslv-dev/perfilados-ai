from enum import Enum 

class DetectionState(Enum): 
    
    NO_CHANGE = 0 
    CANDIDATE_CREATED  = 1
    CANDIDATE_UPDATED = 2 
    WAITING_CONFIRMATION = 3 