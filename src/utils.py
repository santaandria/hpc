import numpy as np
from scipy.special import gamma
import scipy.optimize as sc_opt


def mev_fun(y, pr, N, C, W):
    """MEV distribution function for numerical minimization."""
    # Assuming N, C, W can be arrays for vectorized operations
    return np.sum((1 - np.exp(-((y / C) ** W))) ** N) - N.size * pr


def mev_quant(Fi, x0, N, C, W, potmode=True, thresh=0):
    """Compute the MEV quantile for a given non-exceedance probability Fi."""

    # Setup for vectorized optimization, assuming N, C, W are arrays
    def vectorized_mev_fun(y):
        return mev_fun(y, Fi, N, C, W)

    # Vectorized optimization
    quant, info, ier, mesg = sc_opt.fsolve(vectorized_mev_fun, x0, full_output=True)
    flags = ier != 1  # Non-successful results are flagged

    if potmode:
        quant += thresh

    return quant, flags


def wei_fit_pwm(sample, threshold=0):
    sample = np.asarray(sample)
    wets = sample[sample > threshold]
    x = np.sort(wets)
    if len(x) == 0:
        return np.array([np.nan, np.nan, np.nan])
    M0hat = np.mean(x)
    M1hat = 0.0
    n = x.size
    for ii in range(n):
        real_ii = ii + 1
        M1hat = M1hat + x[ii] * (n - real_ii)
    M1hat = M1hat / (n * (n - 1))
    c = M0hat / gamma(np.log(M0hat / M1hat) / np.log(2))
    w = np.log(2) / np.log(M0hat / (2 * M1hat))
    return np.array([n, c, w])
