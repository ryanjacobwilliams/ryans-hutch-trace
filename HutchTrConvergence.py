import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr
from jax import vmap
import matplotlib.pyplot as plt

key = jr.PRNGKey(1)
key_A, key_hutch = jr.split(key)

def hutchinson_trace_convergence(key, A, m, p = 1):
    #n = dimension of A
    n = A.shape[0]
    
    #Generate m random vectors of size n
    random_vecs = jr.rademacher(key, shape=(m,n), dtype=jnp.float64)
    # random_vecs = jr.uniform(key, shape = (m,n), dtype=jnp.float64, minval=-jnp.sqrt(3), maxval=jnp.sqrt(3))
    # random_vecs = jr.normal(key, shape=(m, n), dtype=jnp.float64)
    # random_vecs =jnp.array([[-1.0,1.0,-1.0],[1.0,1.0,1.0],[-1.0,-1.0,1.0]])

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

    #Computes the cumulative mean of the samples to track convergence of the Hutchinson trace estimator
    estimates = jnp.cumsum(samples) / jnp.arange(1, m + 1)

    return estimates, samples

#Generate test matrix
# A = jr.normal(key_A, shape=(20000, 20000))
# A = jnp.diag(jnp.arange(1.0, 1001.0))
# A = A + jnp.ones((1000,1000))*.01
A = jnp.array([[10.0,1.0,2.0],[1.0,20.0,1.0],[2.0,1.0,10.0]])

#p = matrix degree
p = 1

#m_max = number of random vectors the cumulation terminates at
m_max = 10000
# Calculate the actual trace of A^p
def actual(A, p): 
    return jnp.trace(jnp.linalg.matrix_power(A, p))
# Prepare the actual trace function for Just-In-Time (JIT) compilation
at = jax.jit(actual, static_argnames=("p",))
# Compile the actual trace function ahead of execution
at_compiled = at.lower(A, p).compile()

#Compute actual trace
actual_trace = at_compiled(A)

#Compute the running Hutchinson trace estimates and individual samples
estimates, samples = hutchinson_trace_convergence(key_hutch, A, m_max, p)

#Calculate the absolute forward error of each running estimate
errors = (jnp.abs(actual_trace - estimates))

#Generate the sample sizes from 1 to m_max
m_values = jnp.arange(1, m_max + 1)

#Estimate the standard deviation of the individual Hutchinson samples
std = jnp.std(samples, ddof=1)

#Calculate the estimated standard error at each sample size
standard_error = std / jnp.sqrt(m_values)

#Transform sample sizes and errors to log10 scale for linear regression
x = jnp.log10(m_values.astype(float))
y = jnp.log10(errors)

#Fit a straight line to the log-log data to estimate the convergence rate
coeffs = jnp.polyfit(x, y, 1)

#Extract the slope (convergence exponent) and intercept
slope = coeffs[0]
intercept = coeffs[1]

#Transform the fitted line back to the original scale
y_fit = 10**intercept * m_values**slope

#LOG-LOG PLOT: ABSOLUTE ERROR VS NUMBER OF VECTORS
#Plot the fitted convergence rate on a log-log scale
plt.loglog(m_values, y_fit, label=f"Fit: slope = {slope:.3f}")

#Plot the observed absolute forward errors
plt.loglog(m_values, errors, label="Hutchinson AFE")

#Plot the estimated standard error with theoretical sigma*m^(-1/2) scaling
plt.loglog(m_values,standard_error, "--",label=r"Absolute standard error $\sigma/\sqrt{m}$")

#Label and display the log-log convergence plot
plt.xlabel("Number of random vectors $m$")
plt.ylabel("Absolute forward error")
plt.title("Convergence of Hutchinson Trace Estimator\n1000x1000 matrix with Normal vectors")
plt.legend()
plt.grid(True)
plt.show()

#SEMILOG PLOT: ABSOLUTE ERROR VS NUMBER OF VECTORS
#Plot the same errors with a logarithmic x-axis and linear y-axis
plt.semilogx(m_values, errors, label="Hutchinson AFE")
plt.semilogx(m_values, y_fit, label="Polynomial fit")
plt.semilogx(m_values, standard_error, "--", label="Standard error")

#Restrict the displayed error range to improve readability
plt.ylim(0, 2)

#Label and display the semi-log convergence plot
plt.xlabel("Number of random vectors $m$")
plt.ylabel("Absolute forward error")
plt.legend()
plt.grid(True)
plt.show()

#LOG-LOG PLOT: RMSE VS NUMBER OF VECTORS
#Calculate the squared error of each running trace estimate
squared_errors = (estimates - actual_trace)**2

#Calculate cumulative RMSE over all running estimates up to each m
RMSE= jnp.sqrt(jnp.cumsum(squared_errors) / m_values)

#Calculate the theoretical cumulative RMS using the harmonic sum of inverse sample sizes
cumulative_RMS = standard_error*jnp.sqrt(jnp.cumsum(1.0 / m_values))

#Plot cumulative RMSE and estimated standard error on log-log axes
plt.figure(figsize=(10, 6))
plt.loglog(m_values, RMSE, label="Cumulative RMSE")
plt.loglog(m_values, cumulative_RMS, "--",label=r"Root mean square error $\sigma/\sqrt{m}\sqrt{\sum_{k=1}^m\frac{1}{k}}$")

#Label and display the cumulative RMSE convergence plot
plt.xlabel("Number of Random Vectors (m)")
plt.ylabel("RMSE")
plt.title("Hutchinson Trace Estimator: RMSE Convergence")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.show()

#VALIDATION
# #Display the computed values for comparison with manual calculations
# print("Samples:", samples)
# print("Estimates:", estimates)
# print("Actual Trace:", actual_trace)
# print("Squared Errors:", squared_errors)
# print("Cumulative RMSE:", RMSE)

# #Define the expected results for the three predetermined test vectors
# expected_samples = jnp.array([40.0, 48.0, 36.0])
# expected_estimates = jnp.array([40.0, 44.0, 124.0 / 3.0])
# expected_rmse = jnp.array([0.0, jnp.sqrt(8.0), jnp.sqrt(160.0 / 27.0)])

# #Verify that the computed samples, estimates, and RMSE match expectations
# assert jnp.allclose(samples, expected_samples)
# assert jnp.allclose(estimates, expected_estimates)
# assert jnp.allclose(RMSE, expected_rmse)

# print("All validation tests passed!")