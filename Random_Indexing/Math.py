__author__ = 'Ariel Ekgren, https://github.com/ekgren/'
import numpy as np

def cosine(u, v):
    """
    """
    #u = np.array(u, dtype=np.float)
    #v = np.array(v, dtype=np.float)

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if (u_norm > 0.0) and (v_norm > 0.0):
        dist = 1.0 - np.dot(u, v) / (u_norm * v_norm)
    else:
        dist = np.inf

    if np.isnan(dist):
        return np.inf
    else:
        return dist
