import numpy as np

def some_function(truth, observers):

    dim = truth.shape[0]
    num_observers = observers.size

    R_invs = np.zeros(shape=(dim, dim, num_observers))
    Z = np.zeros(shape=(dim, num_observers))

    for i in range(3):
        R_invs[i, i, :] = 1/(0.001**2)   # positional uncertaintiy of +- 384 km (VERY CONSERVATIVE, most sensor can do better)
        R_invs[3 + i, 3 + i, :] = 1/(0.01**2) # velocity uncertainty ~ +- 0.01 km/s

    Z_transpose = np.array([truth.x + np.random.multivariate_normal(mean=np.zeros(dim), cov=np.linalg.inv(R_invs[:,:,j])) for j in range(num_observers)])
    Z = Z_transpose.T

    return Z, R_invs

