import numpy as np
from spdnet import model
from SPDMetrics import MSELogEuclideanDist, MSEAffineInvariantDist

def test_log_exp():
    x = np.array([[1, 0, 0],
                  [0, 3, 0],
                  [0, 0, 6]])

    u,s,v = np.linalg.svd(x)
    print(u,s,v)
    log_s = np.log(s)
    y= u @ np.diag(log_s) @ np.transpose(v)
    u,s,v = np.linalg.svd(y)
    exp_s = np.exp(s)
    print(u @ np.diag(exp_s) @ np.transpose(v))


def test_log_euclidean_metric():
    x = np.array([[1, 0, 0],
                  [0, 3, 0],
                  [0, 0.1, 6]])
    m = MSELogEuclideanDist()
    m.update_state(10*x, 2*x)
    print('Final result: ', m.result().numpy())

def test_affine_invariant_metric():
    x = np.array([[1, 0, 0],
                  [0, 3, 0.1],
                  [0, 0.1, 6]])
    m = MSEAffineInvariantDist()
    m.update_state(10*x, 2*x)
    print('Final result: ', m.result().numpy())


if __name__ == "__main__":
    #test_log_euclidean_metric()
    #test_affine_invariant_metric()
    test_log_exp()