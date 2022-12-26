from time import time

import numpy as np
from scipy.special import digamma

from data_simulation import get_simulated_data


def main():
    np.random.seed(0)
    sim_train, sim_test = get_simulated_data()

    n_starts = 10

    # Simulated data.
    for dim in [5, 10, 20, 30, 40, 50]:
        Sigma = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    Sigma[i, j] = 1
                elif abs(i - j) == 1:
                    Sigma[i, j] = 0.9
                else:
                    Sigma[i, j] = 0

        # Do multiple random starts.
        best_params = {}
        best_time = 0
        best_bound = -np.Inf
        for start in range(n_starts):
            params, time, bound = run_VI(sim_train[dim], Sigma)
            if bound > best_bound:
                best_params = params
                best_time = time
                best_bound = bound

        # TODO: Compute average hold-out log probability.
        logl = 0

        # TODO: Make plots.

    # TODO: Robot data. Use sample covariance for Sigma.


def run_VI(data, Sigma):
    """Run VI.

    :param data: The data.
    :param Sigma: The known covariance.
    :return: (params, time, bound).
    """

    start_time = time()
    N = len(data)
    dim = len(data[0, 0, :])

    # Hyperparameters.
    K = 20
    alpha = 1
    lambda_1 = np.zeros(dim)  # Prior mean set to zero.
    lambda_2 = 0.1  # TODO: scale "appropriately" for comparison across dimensions.
    threshold = 1e-10
    max_iter = 100  # TODO: remove if convergence check is implemented.

    # TODO: Does this initialization look good?
    # Randomly initialize phi.
    weights = np.ones(shape=K)
    partial_stick_breakoffs = np.random.beta(1, alpha, size=K)
    partial_stick_breakoffs[-1] = 1
    weights[0] = partial_stick_breakoffs[0]
    weights[1:] = partial_stick_breakoffs[1:] * (1 - partial_stick_breakoffs[:-1]).cumprod()
    phi = np.random.multinomial(1, weights, size=N).astype(float)

    gamma_1 = np.zeros(K, dtype=float)
    gamma_2 = np.zeros(K, dtype=float)
    tau_1 = np.zeros((K, dim), dtype=float)
    tau_2 = np.zeros(K, dtype=float)

    old_bound = -np.Inf
    converged = False
    n_iter = 0
    while not converged and n_iter < max_iter:
        # Update gamma and tau.
        phi_sums = np.sum(phi, axis=0)
        for i in range(K):
            gamma_1[i] = 1 + phi_sums[i]
            gamma_2[i] = alpha + np.sum(phi_sums[i + 1:])
            tau_sum = np.zeros(dim)
            for n in range(N):
                tau_sum += phi[n, i] * data[n, 0, :]
            tau_1[i, :] = lambda_1 + tau_sum
            tau_2[i] = lambda_2 + phi_sums[i]

        # Update phi.
        digamma_sum = digamma(gamma_1 + gamma_2)
        E_log_V1 = digamma(gamma_1) - digamma_sum
        E_log_V0 = digamma(gamma_2) - digamma_sum

        E_eta = np.zeros((K, dim))
        E_a_eta = np.zeros(K)
        for i in range(K):
            # Define m := Sigma^{-1} * tau_1, so that Sigma * m = tau_1 (a system of equations is numerically stable).
            m = np.linalg.solve(Sigma, tau_1[i])
            # Now Sigma^{-1} * tau_1 / tau_2 = m / tau_2.
            E_eta[i, :] = m / tau_2[i]

            # Use the same trick:
            # tau_1^T * Sigma^{-1} * tau_1 / (2 * tau_2^2) + 1 / (2 * tau_2) = tau_1^T * m / (2 * tau_2^2) + 1 / (2 *
            # tau_2)
            E_a_eta[i] = np.matmul(tau_1[i], m) / (2 * tau_2[i] ** 2) + 1 / (2 * tau_2[i])

        for n in range(N):
            u = np.zeros(K)
            for i in range(K):
                u_1 = E_log_V1[i]
                u_2 = np.matmul(E_eta[i], data[n, 0, :])
                u_3 = E_a_eta[i]
                u_4 = np.sum(E_log_V0[:i])
                u[i] = np.exp(u_1 + u_2 - u_3 + u_4)
            phi[n, :] = u / np.sum(u)

        # TODO: Check for convergence.
        # Stop if relative change in log marginal probability bound is smaller than the threshold.
        bound = -np.Inf
        if abs(bound - old_bound) / abs(old_bound) < threshold:
            # converged = True
            pass
        old_bound = bound
        n_iter += 1

    params = {'phi': phi, 'gamma_1': gamma_1, 'gamma_2': gamma_2, 'tau_1': tau_1, 'tau_2': tau_2}
    return params, time() - start_time, old_bound


if __name__ == '__main__':
    main()
