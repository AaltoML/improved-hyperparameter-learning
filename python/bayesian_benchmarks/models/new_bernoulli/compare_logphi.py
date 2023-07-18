import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import norm

import logphi  # GPML version


def inv_probit(x, jitter=np.float64(1e-3)):  # GPflow version
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


def gpflow_log_inv_probit(x, jitter):
    return tf.math.log(inv_probit(x, jitter))


@tf.function
def new_log_inv_probit(x, jitter):
    # no more need for jitter
    return logphi.new_log_inv_probit(x)


log_inv_probit = new_log_inv_probit


def log_inv_probit_grad(x, jitter=np.float64(1e-3)):  # GPflow version
    xtf = tf.Variable(x)
    with tf.GradientTape() as tape:
        logphi = log_inv_probit(xtf, jitter)
    return tape.gradient(logphi, xtf)


x = np.linspace(-10, 10, 201)
jittervals = [-3, -6, -9, -12, -15]
# jittervals = [-3, -6, -15, -16, -17, -18]

fig, axes = plt.subplots(2, 1, sharex=True)
plt.sca(axes[0])
plt.plot(x, logphi.logphi(x, grad=False), label="GPML logphi")
for jitter in jittervals:
    plt.plot(
        x,
        log_inv_probit(x, np.float64(10**jitter)),
        label=f"GPflow log(phi), jitter=1e{jitter}",
    )
plt.plot(x, norm.logcdf(x), "k--", label="Scipy norm.logcdf")
plt.legend(loc="best")
plt.ylabel(r"$\log(\Phi(f))$")
plt.xlabel(r"$f$")

plt.sca(axes[1])
plt.plot(x, logphi.logphi(x, grad=True)[1], label="GPML logphi")
for jitter in jittervals:
    plt.plot(
        x,
        log_inv_probit_grad(x, np.float64(10**jitter)),
        label=f"GPflow log(phi), jitter=1e{jitter}",
    )
plt.legend(loc="best")
plt.ylabel(r"$\frac{\mathrm{d}}{\mathrm{d}f} \log(\Phi(f))$")
plt.xlabel(r"$f$")
plt.xlim(x[0], x[-1])

plt.show()
