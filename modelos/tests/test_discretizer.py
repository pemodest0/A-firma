import numpy as np
from src.adaptive_discretizer import discretize_system


def lorenz(t_max=50, dt=0.01):
    # simple Lorenz integrator
    sigma = 10.0
    rho = 28.0
    beta = 8/3
    n = int(t_max/dt)
    xs = np.zeros(n)
    ys = np.zeros(n)
    zs = np.zeros(n)
    x,y,z = 0.1, 0.0, 0.0
    for i in range(n):
        dx = sigma*(y-x)
        dy = x*(rho-z)-y
        dz = x*y - beta*z
        x += dx*dt
        y += dy*dt
        z += dz*dt
        xs[i]=x; ys[i]=y; zs[i]=z
    return xs


def test_entropy_increases_with_noise():
    x = lorenz(t_max=5, dt=0.01)
    G1, labels1, _, stats1 = discretize_system(x, method='entropy', max_bins=8)
    # add small noise
    x_noisy = x + np.random.normal(scale=0.01, size=x.shape)
    G2, labels2, _, stats2 = discretize_system(x_noisy, method='entropy', max_bins=8)
    assert stats2['entropy_symbolic'] >= stats1['entropy_symbolic'] - 1e-6
