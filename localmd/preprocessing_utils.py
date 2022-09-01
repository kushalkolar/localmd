import jax
import jax.scipy
import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial


@partial(jit)
def get_noise_estimate(trace):
    output_welch = jax.scipy.signal.welch(trace, noverlap=128)
    start = int(256/4 + 1)
    end = int(256/2 + 1)

    indices = jnp.arange(start, end)
    values = jnp.take(output_welch[1], indices) * 0.5
    sum_values = jnp.sqrt(jnp.sum(values))

    return sum_values / (end - start)


get_noise_estimate_vmap = jit(vmap(get_noise_estimate, in_axes=(0)))

@partial(jit)
def get_mean(trace):
    return jnp.mean(trace, axis = 1)

@partial(jit)
def sum_and_normalize(trace, normalizer):
    return jnp.sum(trace, axis = -1) / normalizer


@partial(jit)
def center_and_get_noise_estimate(trace, mean):
    centered_trace = trace - mean
    return get_noise_estimate(centered_trace)

center_and_get_noise_estimate_vmap = jit(vmap(center_and_get_noise_estimate, in_axes=(0,0)))


@partial(jit)
def standardize_data(data, mean, std):
    centered = data - mean
    standardized = centered / std
    return standardized
