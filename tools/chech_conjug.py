# controlla se un vb flesso corrisp a un certo vb infin
from mlconjug3 import Conjugator
from time import sleep

conjugator = Conjugator(language="it") # così usa l'italiano

def check_conjugation(verb, conjugation):
    """
    Verifie si une forme est bien une flexion d'un verbe.

    Args:
        verb (str): The verb to check.
        conjugation (str): Forme a verifier.

    Returns:
        bool: Vrai si la forme est bien une flexion du verbe.

    """
    try:
        verb_conjs = conjugator.conjugate(verb).iterate()

    except:
        for k in range(10000):
            l = 0

        print(f'except : {verb}, {conjugation}')

        return False


    #return (('Indicativo', 'Indicativo presente', 'egli/ella', conjugation) in verb_conjs)
    return (('Indicativo', 'Indicativo presente', '3s', conjugation) in verb_conjs)


def get_conj(verb):
    """
    Retourne la conjugaison du verbe.

    Args:
        verb (str): Le verbe.

    Returns:
        str: La conjugaison du verbe.

    """
    verb_conjs = conjugator.conjugate(verb).iterate()

    for verb_conj in verb_conjs:
        if verb_conj[0] == "Indicativo" and verb_conj[1] == "Indicativo presente" and verb_conj[2] == "3s":
            return verb_conj[3]

    return None