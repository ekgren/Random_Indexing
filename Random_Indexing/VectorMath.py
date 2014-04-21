import numpy as np

__author__ = "Ariel Ekgren, https://github.com/ekgren/"


def cosine(u, v):
    """
    """

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if (u_norm > 0.0) and (v_norm > 0.0):
        dist = 1.0 - np.dot(u, v) / (u_norm * v_norm)
    else:
        dist = np.inf

    return dist
