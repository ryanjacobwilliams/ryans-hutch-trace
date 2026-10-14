import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt
import time

key = jr.PRNGKey(3)
key_A, key_hutch = jr.split(key)

# def hutchinson_trace_estimator(key, A, m, power = 1):                                   # m = number of random vectors
#     n = A.shape[0]                                                                      # n = dimension of the matrix                                  
#     random_vecs = jr.choice(key, a=jnp.array([-1.0,1.0]), shape = (m,n))                    # Generates mxn matrix of the rademacher vectors

#     def vAv(v):      
#         Av = v                                                                                                            
#         for i in range(power):  
#             Av = A @ Av                                                                 # Matrix-vector multiplication
#         return v @ Av

#     return jnp.mean(vmap(vAv)(random_vecs))                                             # Mean of vAv on all random vectors at once

# def hutchinson_trace_estimator_lax(key, A, m, power = 1):                                   # m = number of random vectors
#     n = A.shape[0]                                                                      # n = dimension of the matrix                                  
#     # random_vecs = jr.rademacher(key, shape = (m,n), dtype=float)                                     # Generates mxn matrix of the rademacher vectors
#     random_vecs =jnp.array([[-1.0,1.0,-1.0],[1.0,1.0,1.0],[-1.0,-1.0,1.0]])

#     def vAv(v):                                                                                                                 

#         def AvMult(Av,_):  
#             return A @ Av, None                                                       # Matrix-vector multiplication

#         Av, _ = jax.lax.scan(AvMult, v, xs=None, length = power)
        
#         return v @ Av

#     samples = vmap(vAv)(random_vecs)

#     indices = jnp.argsort(jnp.abs(samples))
#     samples_sorted = samples[indices]


#     return jnp.mean(samples_sorted)


# A = jr.normal(key_A, shape=(20000, 20000))
A = jnp.diag(jnp.arange(1.0, 1001.0))                                                   # diagonal matrix
A = A + jnp.ones((1000,1000))*.01
# A = jnp.array([[10.0,1.0,2.0],[1.0,20.0,1.0],[2.0,1.0,10.0]])
power = 1

def actual(A, power):
    return jnp.trace(jnp.linalg.matrix_power(A, power))

at = jax.jit(actual, static_argnames=("power",))
at.lower(A, power).compile()

tic = time.perf_counter()
actual_trace = at(A, power).block_until_ready()         # Calculates actual trace using numpy methods
toc = time.perf_counter()
print("Actual trace:", actual_trace)
print(f"Computation time of actual trace: {toc - tic:.6f} seconds")

# hte = jax.jit(hutchinson_trace_estimator_lax, static_argnames=("m", "power"))
# hte.lower(key_hutch, A, 10000, power).compile()

# tic = time.perf_counter()
# estimated_trace = hutchinson_trace_estimator(key_hutch, A, 10000, power).block_until_ready()
# toc = time.perf_counter()
# print("Estimated trace (total time):", estimated_trace)
# print(f"Computation time of estimated trace (total time): {toc - tic:.6f} seconds")
# tic = time.perf_counter()
# estimated_trace = hutchinson_trace_estimator_lax(key_hutch, A, 1, power).block_until_ready()
# toc = time.perf_counter()
# print("Estimated trace (compute time):", estimated_trace)
# print(f"Computation time of estimated trace using lax (compute time): {toc - tic:.6f} seconds")


# AFE = abs(actual_trace - estimated_trace)               # Absolute forward error
# print(f"Absolute forward error: {AFE*100:.3f}")
# RFE = AFE / abs(actual_trace)                           # Relative forward error
# print(f"Relative forward error: {RFE*100:.3f} %")


def hutchinson_trace_convergence(key, A, m, power = 1):                                                                     # m = number of random vectors
    n = A.shape[0]                                                                                                          # n = dimension of the matrix                                  
    random_vecs = jr.rademacher(key, shape=(m,n), dtype=jnp.float64)
    # random_vecs = jr.uniform(key, shape = (m,n), dtype=jnp.float64, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))                      # Generates mxn matrix of the rademacher vectors
    # random_vecs = jr.normal(key, shape=(m, n), dtype=jnp.float64)

    # random_vecs =jnp.array([[-1.0,1.0,-1.0],[1.0,1.0,1.0],[-1.0,-1.0,1.0]])

    def vAv(v):      
        def AvMult(Av,_):  
            return A @ Av, None                                                                             # Matrix-vector multiplication
        
        Av, _ = jax.lax.scan(AvMult, v, xs=None, length = power)
                
        return v @ Av

    samples = vmap(vAv)(random_vecs)


    estimates = jnp.cumsum(samples) / jnp.arange(1, m + 1)

    return estimates, samples     


m_max = 10000

estimates, samples = hutchinson_trace_convergence(
    key_hutch,
    A,
    m_max,
    power
)

estimates.block_until_ready()

errors = (
    jnp.abs(actual_trace - estimates)

)

m_values = jnp.arange(1, m_max + 1)

std = jnp.std(samples, ddof=1)

standard_error = std / jnp.sqrt(m_values)                                               #Expected absolute error

relative_standard_error = (
    standard_error                                                                      #Expected relative error
)

x = jnp.log10(m_values.astype(float))
y = jnp.log10(errors)

coeffs = jnp.polyfit(x, y, 1)

slope = coeffs[0]
intercept = coeffs[1]

y_fit = 10**intercept * m_values**slope

plt.loglog(m_values, y_fit, label=f"Fit: slope = {slope:.3f}")

plt.loglog(
    m_values,
    errors,
    label="Hutchinson AFE"
)

plt.loglog(
    m_values,
    relative_standard_error,
    "--",
    label=r"Absolute standard error $\sigma/\sqrt{m}$"
)

plt.xlabel("Number of random vectors $m$")
plt.ylabel("Absolute forward error")
plt.title("Convergence of Hutchinson Trace Estimator\n1000x1000 matrix with Normal vectors")
plt.legend()
plt.grid(True)

plt.show()

plt.semilogx(m_values, errors, label="Hutchinson AFE")
plt.semilogx(m_values, y_fit, label="Polynomial fit")
plt.semilogx(m_values, relative_standard_error, "--",
             label="Standard error")

plt.ylim(0, 2)  # Only display errors between 0 and 2

plt.xlabel("Number of random vectors $m$")
plt.ylabel("Absolute forward error")
plt.legend()
plt.grid(True)
plt.show()