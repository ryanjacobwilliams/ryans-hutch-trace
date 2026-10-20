import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import time

key = jr.PRNGKey(1)
key_A, key_hutch = jr.split(key)

def hutchinson_trace_estimator(key, A, m, p = 1):
    #n = dimension of A
    n = A.shape[0]
    
    #Generate m random vectors of size n
    # random_vecs = jr.rademacher(key, shape = (m,n), dtype=float)
    random_vecs =jnp.array([[-1.0,1.0,-1.0],[1.0,1.0,1.0],[-1.0,-1.0,1.0]])

    jax.debug.print("Vectors:\n{}\n", random_vecs)

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

    jax.debug.print("Samples: {}", samples)

    #Sort samples to avoid rounding errors while summing
    indices = jnp.argsort(jnp.abs(samples))
    samples_sorted = samples[indices]

    jax.debug.print("Sum: {}\n", jnp.sum(samples_sorted))

    #Calcuate (1/m)*sum_i(v_i^T*A^p*v_i)
    trace_estimation = jnp.mean(samples_sorted)
        
    return trace_estimation
    
#Generate test matrix
# A = jr.normal(key_A, shape=(20000, 20000))
# A = jnp.diag(jnp.arange(1.0, 1001.0))
# A = A + jnp.ones((1000,1000))*.01
A = jnp.array([[10.0,1.0,2.0],[1.0,20.0,1.0],[2.0,1.0,10.0]])

#p = matrix degree
p = 1

#m = number of random vectors
m = 10000

#Prepare the estimator for Just-In-Time (JIT) compilation
hte = jax.jit(hutchinson_trace_estimator, static_argnames=("m", "p"))
#Compile the estimator ahead of execution to exclude compilation time from benchmarking
hte_compiled = hte.lower(key_hutch, A, m, p).compile()

tic = time.perf_counter()
#Estimate trace for A
estimated_trace = hte_compiled(key_hutch, A).block_until_ready()
toc = time.perf_counter()
print("Estimated trace (compute time):", estimated_trace)
print(f"Computation time of estimated trace(compute time): {toc - tic:.6f} seconds\n")

# Calculate the actual trace of A^p
def actual(A, p):
    return jnp.trace(jnp.linalg.matrix_power(A, p))
# Prepare the actual trace function for Just-In-Time (JIT) compilation
at = jax.jit(actual, static_argnames=("p",))
# Compile the actual trace function ahead of execution
at_compiled = at.lower(A, p).compile()

tic = time.perf_counter()
#Compute actual trace
actual_trace = at_compiled(A, p).block_until_ready()
toc = time.perf_counter()
print("Actual trace:", actual_trace)
print(f"Computation time of actual trace: {toc - tic:.6f} seconds\n")

#Absolute forward error
AFE = abs(actual_trace - estimated_trace)
print(f"Absolute forward error: {AFE*100:.3f}")
#Relative forward error
RFE = AFE / abs(actual_trace)
print(f"Relative forward error: {RFE*100:.3f} %")

