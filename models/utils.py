import numpy as np
import jax
import jax.numpy as jnp

def jax_loggauss_multidim_01(x, k):
    return (-jnp.log(jnp.sqrt((2 * jnp.pi) ** k)) -0.5 * jnp.sum(jnp.square(x)))

vmap_jax_loggauss_multidim_01 = (jax.vmap(jax_loggauss_multidim_01, in_axes=(0,
    None)))

def jax_gauss(x, mu, sigma):
    res = (1 / jnp.sqrt(2 * jnp.pi * sigma ** 2) * jnp.exp(-0.5 * (x - mu) ** 2
            / sigma ** 2))
    return res

def jax_loggauss(x, mu, sigma):
#    print((-jnp.log(jnp.sqrt(2 * jnp.pi * sigma ** 2)) -0.5 *
#        (x - mu) ** 2 / sigma ** 2).shape)
    return (-jnp.log(jnp.sqrt(2 * jnp.pi * sigma ** 2)) -0.5 *
        (x - mu) ** 2 / sigma ** 2)

vmap_jax_loggauss = jax.vmap(jax_loggauss, in_axes=(1, 0, 0))

def jax_dot(a, b):

    return jnp.dot(a, b)

vmap_jax_dot = jax.vmap(jax_dot, in_axes=(0, 0))

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def relu(x):
    return np.maximum(0, x)

def gauss(x, mu, sigma):
    res = (1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-0.5 * (x - mu) ** 2
            / sigma ** 2))
    res[~np.isfinite(res)] = 0
    return res

def loggauss(x, mu, sigma):
    return (-np.log(np.sqrt(2 * np.pi * sigma ** 2)) -0.5 *
        (x - mu) ** 2 / sigma ** 2)

