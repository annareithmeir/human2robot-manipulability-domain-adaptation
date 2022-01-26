#import tensorflow as tf
#from tensorflow_riemopt.manifolds import SPDLogEuclidean
#from tensorflow_riemopt.manifolds import SPDAffineInvariant

from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm

import numpy as np
from scipy.linalg import eigvalsh


# class MSELogEuclideanDist(tf.keras.metrics.Metric):
#     def __init__(self, name='log_euclidean_dist', **kwargs):
#         super(MSELogEuclideanDist, self).__init__(name=name, **kwargs)
#         self.dist = self.add_weight(name='logEuclidean', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         manifold = SPDLogEuclidean()
#         tmp=manifold.dist(y_true, y_pred)
#         tmp=tf.cast(tmp, tf.float32)
#         tmp = tf.reduce_mean(tf.math.square(tmp))
#         self.dist.assign(tmp)

#     def result(self):
#         return self.dist


# class MSEAffineInvariantDist(tf.keras.metrics.Metric):
#     def __init__(self, name='affine_invariant_dist', **kwargs):
#         super(MSEAffineInvariantDist, self).__init__(name=name, **kwargs)
#         self.dist = self.add_weight(name='affineInvariant', initializer='zeros')

#     def update_state(self, y_true, y_pred,sample_weight=None):
#         manifold = SPDAffineInvariant()
#         tmp=manifold.dist(y_true, y_pred)
#         tmp=tf.cast(tmp, tf.float32)
#         tmp = tf.reduce_mean(tf.math.square(tmp))
#         self.dist.assign(tmp)

#     def result(self):
#         return self.dist


def distance_riemann(A, B):
    r"""Riemannian distance between two covariance matrices A and B.
    .. math::
        d = {\left( \sum_i \log(\lambda_i)^2 \right)}^{1/2}
    where :math:`\lambda_i` are the joint eigenvalues of A and B
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Riemannian distance between A and B
    """
    return np.sqrt((np.log(eigvalsh(A, B))**2).sum())


def distance_wasserstein(A,B):
    return sqrtm(np.trace(A+B)-2*np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A))))

# d = norm(logm(A) - logm(B), 'fro');
def distance_logeuc(A,B):
    diff = logm(A) - logm(B)
    return np.linalg.norm(diff, axis=(-2, -1), ord="fro", keepdims=False)



