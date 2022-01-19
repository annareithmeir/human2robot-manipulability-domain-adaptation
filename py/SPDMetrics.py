import tensorflow as tf
from tensorflow_riemopt.manifolds import SPDLogEuclidean
from tensorflow_riemopt.manifolds import SPDAffineInvariant


class MSELogEuclideanDist(tf.keras.metrics.Metric):
    def __init__(self, name='log_euclidean_dist', **kwargs):
        super(MSELogEuclideanDist, self).__init__(name=name, **kwargs)
        self.dist = self.add_weight(name='logEuclidean', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        manifold = SPDLogEuclidean()
        tmp=manifold.dist(y_true, y_pred)
        tmp=tf.cast(tmp, tf.float32)
        tmp = tf.reduce_mean(tf.math.square(tmp))
        self.dist.assign(tmp)

    def result(self):
        return self.dist


class MSEAffineInvariantDist(tf.keras.metrics.Metric):
    def __init__(self, name='affine_invariant_dist', **kwargs):
        super(MSEAffineInvariantDist, self).__init__(name=name, **kwargs)
        self.dist = self.add_weight(name='affineInvariant', initializer='zeros')

    def update_state(self, y_true, y_pred,sample_weight=None):
        manifold = SPDAffineInvariant()
        tmp=manifold.dist(y_true, y_pred)
        tmp=tf.cast(tmp, tf.float32)
        tmp = tf.reduce_mean(tf.math.square(tmp))
        self.dist.assign(tmp)

    def result(self):
        return self.dist

