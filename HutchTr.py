import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import vmap

key = jr.PRNGKey(1)

def hutchinson_trace_estimator(key, A, m, p = 1):
    #n = dimension of A
    n = A.shape[0]

    #Generate m random vectors of size n
    random_vecs = jr.rademacher(key, shape = (m,n), dtype=float)

    def vAv(v):
        #Vector-matrix multiplication
        def AvMult(Av,_):  
            return A @ Av, None
        
        #Compute A^p*v using repeated matrix-vector multiplication
        Av, _ = jax.lax.scan(AvMult, v, xs=None, length = p)

        #Vector-vector multiplication
        return v @ Av

    #Compute v_i^T*A^p*v_i for each of the m random vectors
    samples = vmap(vAv)(random_vecs)

    #Sort samples to avoid rounding errors while summing
    indices = jnp.argsort(jnp.abs(samples))
    samples_sorted = samples[indices]

    #Calcuate (1/m)*sum_i(v_i^T*A^p*v_i)
    trace_estimation = jnp.mean(samples_sorted)
    
    return trace_estimation
