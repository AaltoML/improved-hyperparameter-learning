from numpy import array, float64, pi

# Numpy:
# from numpy import abs, sqrt, log, exp, zeros_like
# from scipy.special import erfc

# TensorFlow:
from tensorflow import zeros_like, where
from tensorflow.math import abs, sqrt, log, exp, erfc

# fmt: off
# Data arrays copied from GPML matlab code
c = array([ 0.00048204, -0.00142906, 0.0013200243174, 0.0009461589032,
     -0.0045563339802, 0.00556964649138, 0.00125993961762116,
     -0.01621575378835404, 0.02629651521057465, -0.001829764677455021,
     2*(1-pi/3), (4-pi)/3, 1, 1])

r = array([ 1.2753666447299659525, 5.019049726784267463450,
      6.1602098531096305441, 7.409740605964741794425,
      2.9788656263939928886 ])

q = array([ 2.260528520767326969592,  9.3960340162350541504,
     12.048951927855129036034, 17.081440747466004316, 
      9.608965327192787870698,  3.3690752069827527677 ])
# fmt: on


def np_logphi(z, grad=True):
    lp = zeros_like(z)  # allocate memory

    id1 = (z * z) < float64(0.0492)  # first case: close to zero
    lp0 = -z[id1] / sqrt(float64(2 * pi))
    f = float64(0)
    for i in range(14):
        f = lp0 * (c[i] + f)
    lp[id1] = -2 * f - log(float64(2))

    id2 = z < float64(-11.3137)  # second case: very small
    num = float64(0.5641895835477550741)
    for i in range(5):
        num = -z[id2] * num / sqrt(float64(2)) + r[i]
    den = float64(1.0)
    for i in range(6):
        den = -z[id2] * den / sqrt(float64(2)) + q[i]
    e = num / den
    lp[id2] = log(e / 2) - z[id2] ** 2 / 2

    id3 = (~id2) & (~id1)
    lp[id3] = log(erfc(-z[id3] / sqrt(float64(2))) / 2)  # third case: rest

    if not grad:
        return lp

    else:  # compute first derivative
        dlp = zeros_like(z)  # allocate memory
        dlp[id2] = abs(den / num) * sqrt(
            float64(2 / pi)
        )  # strictly positive first derivative
        dlp[~id2] = exp(-z[~id2] * z[~id2] / 2 - lp[~id2]) / sqrt(
            float64(2 * pi)
        )  # safe computation
        """
        if nargout>2                                     % compute second derivative
          d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
          if nargout>3                                    % compute third derivative
            d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
          end
        end
        """
        return lp, dlp


def logphi(z, grad=True):
    id1 = (z * z) < float64(0.0492)  # first case: close to zero
    lp0 = -z / sqrt(float64(2 * pi))
    f = float64(0)
    for i in range(14):
        f = lp0 * (c[i] + f)
    lp_id1 = -2 * f - log(float64(2))

    id2 = z < float64(-11.3137)  # second case: very small
    num = float64(0.5641895835477550741)
    for i in range(5):
        num = -z * num / sqrt(float64(2)) + r[i]
    den = float64(1.0)
    for i in range(6):
        den = -z * den / sqrt(float64(2)) + q[i]
    e = num / den
    lp_id2 = log(e / 2) - z**2 / 2

    id3 = (~id2) & (~id1)
    lp_id3 = log(erfc(-z / sqrt(float64(2))) / 2)  # third case: rest

    lp = where(id1, lp_id1, where(id2, lp_id2, lp_id3))

    if not grad:
        return lp

    else:  # compute first derivative
        dlp_id2 = abs(den / num) * sqrt(
            float64(2 / pi)
        )  # strictly positive first derivative
        dlp_else = exp(-z * z / 2 - lp) / sqrt(float64(2 * pi))  # safe computation
        dlp = where(id2, dlp_id2, dlp_else)
        """
        if nargout>2                                     % compute second derivative
          d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
          if nargout>3                                    % compute third derivative
            d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
          end
        end
        """
        return lp, dlp


import tensorflow as tf


@tf.custom_gradient
def new_log_inv_probit(z):
    lp, dlp = logphi(z, grad=True)

    def grad(dout):
        return dout * dlp

    return lp, grad


import gpflow


class NewProbitBernoulli(gpflow.likelihoods.Bernoulli):
    def __init__(self, **kwargs):
        super().__init__(self._invlink, **kwargs)

    def _invlink(self, F):
        return tf.math.exp(new_log_inv_probit(F))

    def _scalar_log_prob(self, F, Y):
        return new_log_inv_probit(F * tf.cast(Y, tf.float64))
