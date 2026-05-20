import numpy as np

def frobenius_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b, ord="fro"))
