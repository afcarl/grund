import numpy as np


def is_neighbour(entity, other, diag=True):
    thresh = np.sqrt(2.) if diag else 1.
    return np.linalg.norm(entity.position - other.position) <= thresh
