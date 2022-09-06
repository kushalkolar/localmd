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


@partial(jit)
def get_mean(trace):
    return jnp.mean(trace)

@partial(jit)
def center(trace):
    mean = get_mean(trace)
    return trace - mean

center_vmap = vmap(center, in_axes=(0))

@partial(jit)
def center_and_noise_normalize(trace):
    mean = get_mean(trace)
    centered_trace = trace - mean
    noise_est = get_noise_estimate(centered_trace)
    return centered_trace / noise_est

center_and_noise_normalize_vmap = jit(vmap(center_and_noise_normalize, in_axes=(0)))


@partial(jit)
def standardize_block(block):
    '''
    Input: 
        block: jnp.array. Dimensions (d1, d2, T)
    '''
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1*d2, T), order="F")
    updated_2d = center_and_noise_normalize_vmap(block_2d)
    updated_3d = jnp.reshape(updated_2d, (d1, d2, T), order="F")
    return updated_3d