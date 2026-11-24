import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import vmap

from HutchTr import hutchinson_trace_estimator

# Define Potential Energy for p-Gaussian
def U_gauss(q, p=4):
    return jnp.sum(jnp.abs(q)**p) / p

# Define Potential Energy for Neal's Funnel
def U_neal(q):
    v = q[0]
    x = q[1:]

    return (v**2 / 18
            + (len(x) / 2) * v
            + 0.5 * jnp.exp(-v) * jnp.sum(x**2))


# Define F(Q,q)
def F(Q, q, U, n=100):
    s = jnp.linspace(0, 1, n + 1)

    # Points along the line from q to Q
    points = q + s[:, None] * (Q - q)

    # Gradient of potential energy
    grad_U = jax.grad(U)

    # Calculate gradient at each point
    gradients = vmap(grad_U)(points)

    # Approximate integral using trapezoidal method
    return jnp.trapezoid(gradients, s, axis=0)

# Define AVF
def AVF(q, p, tau, U, iterations=10):

    # Initial guess for Q with no force
    Q = q + tau * p

    # Function whose root we want to find
    def G(Q):
        return Q - q - tau*p + (tau**2 / 2)*F(Q, q, U)

    # Jacobian of G with respect to Q
    J_G = jax.jacfwd(G)

    # Newton's method
    for _ in range(iterations):
        Q = Q + jnp.linalg.solve(J_G(Q), -G(Q))

    # Calculate new momentum
    P = p - tau * F(Q, q, U)

    return Q, P

# Define Hamiltonian (Assuming M = I)
def H(q, p, U):
    return U(q) + 0.5 * jnp.dot(p, p)

#====================================
# Test Energy Conservation for AVF
#====================================

# Test parameters
q = jnp.array([1.0, 2.0, 3.0])
p = jnp.array([0.5, 0.5, 0.5])

# Perform one AVF step
Q_gauss, P_gauss = AVF(q, p, tau=0.1, U=U_gauss)

print("==============================")
print("AVF Test for p-Gauss")
print("==============================")
print("Energy before:", H(q, p, U_gauss))
print("Energy after:", H(Q_gauss, P_gauss, U_gauss))
print("Energy difference:", H(Q_gauss, P_gauss, U_gauss) - H(q, p, U_gauss))

Q_neal, P_neal = AVF(q, p, tau=0.1, U=U_neal)

print("==============================")
print("AVF Test for Neal's Funnel")
print("==============================")
print("Energy before:", H(q, p, U_neal))
print("Energy after:", H(Q_neal, P_neal, U_neal))
print("Energy difference:", H(Q_neal, P_neal, U_neal) - H(q, p, U_neal))

#====================================


# Jacobian with respect to q
DqF = jax.jacfwd(F, argnums=1)

# Jacobian with respect to Q
DQF = jax.jacfwd(F, argnums=0)

# Define the second degree AVF Jacobian function
def AVF_Jacobian(key, DqF, DQF, tau, m):

    # Difference between the two matrices
    A = DqF - DQF
    print("Jacobian:", A)

    # Estimate the trace of the difference
    trace_A = jnp.trace(A)
    trace_DQF = hutchinson_trace_estimator(key, DQF, m, 2)
    trace_DqF = hutchinson_trace_estimator(key, DqF, m, 2)

    # Approximate the Jacobian determinant
    return 1 + (tau**2 / 2) * trace_A + (tau**4 / 8) * (-trace_A)**2 + (tau**4 / 8) * (trace_DQF - trace_DqF)


# Calculate the two Jacobian matrices
Dq = DqF(Q_gauss, q, U_gauss)
DQ = DQF(Q_gauss, q, U_gauss)

# Exact determinant from Lemma 6
I = jnp.eye(len(q))
tau = 0.1

exact = jnp.linalg.det(I + (tau**2 / 2)*Dq) / \
        jnp.linalg.det(I + (tau**2 / 2)*DQ)

# Hutchinson approximation
key = jax.random.PRNGKey(1)

print("==============================")
print("Determinate test for p-Gauss")
print("==============================")
estimate = AVF_Jacobian(key, Dq, DQ, tau, m=100)
print("Exact Jacobian determinant:", exact)
print("Hutchinson approximation:", estimate)
print("Absolute error:", jnp.abs(exact - estimate))


# Calculate the two Jacobian matrices
Dq = DqF(Q_neal, q, U_neal)
DQ = DQF(Q_neal, q, U_neal)

# Exact determinant for Neal's Funnel
exact = jnp.linalg.det(I + (tau**2 / 2)*Dq) / \
        jnp.linalg.det(I + (tau**2 / 2)*DQ)

print("==============================")
print("Determinate test for Neal's Funnel")
print("==============================")
estimate = AVF_Jacobian(key, Dq, DQ, tau, m=100)
print("Exact Jacobian determinant:", exact)
print("Hutchinson approximation:", estimate)
print("Absolute error:", jnp.abs(exact - estimate))
