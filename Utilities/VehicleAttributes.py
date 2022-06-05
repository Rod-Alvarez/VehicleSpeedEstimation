from dataclasses import dataclass


@dataclass
class Attributes:
    '''
    Data class to hold speed attributes of each individual tracked object
    '''
    prev_center = [] # last known anchor point
    cur_center = [] # current anchor point
    speed = [] # list of speed ratings to get average
    counter: int = 0 # drives how long before we check speed
    avg_speed: float = 0.0 # current average speed
