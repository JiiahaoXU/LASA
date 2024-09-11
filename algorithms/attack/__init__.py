from .agr import *
from .lie import *
from .naive import *

def attack(attack):

    attacks = {'random_attack': random_attack,
               'signflip_attack': signflip_attack,
               'agrTailoredTrmean': agrTailoredTrmean,
               'noise_attack': noise_attack,
               'agrAgnosticMinMax': agrAgnosticMinMax,
               'label_flip': non_attack,
               'lie_attack': lie_attack,
               'byzmean_attack': byzmean_attack,
               'agrTailoredKrumBulyan': agrTailoredKrumBulyan,
               'agrAgnosticMinSum': agrAgnosticMinSum,
               'non_attack': non_attack,
    }

    return attacks[attack]