import numpy as np
import scipy


def _generate_simulated_data(BASE_MEASURE, COVARIANCE_MATRIX, alpha, data_points, truncation_level):
    weights = np.ones(shape=truncation_level)
    partial_stick_breakoffs = np.random.beta(1, alpha, size=truncation_level)
    partial_stick_breakoffs[-1] = 1
    weights[0] = partial_stick_breakoffs[0]
    weights[1:] = partial_stick_breakoffs[1:] * (1 - partial_stick_breakoffs[:-1]).cumprod()

    data = np.random.multinomial(1, weights, size=data_points)

    mus_all_clusters = BASE_MEASURE.rvs(size=truncation_level)

    tbret = []
    for data_point_ in data:
        cluster = np.where(data_point_ == 1)
        mean_cluster = mus_all_clusters[cluster].reshape(-1)
        simulated_data_point = np.random.multivariate_normal(mean=mean_cluster, cov=COVARIANCE_MATRIX, size=1)
        tbret.append(simulated_data_point)

    return np.array(tbret)


def get_simulated_data():
    all_dimensions = [5, 10, 20, 30, 40, 50]
    TRUNCATION_LEVEL = 20
    alpha = 1
    data_points = 200
    simulated_data_all_dimensions = {}
    hold_out_all_dimensions = {}
    for dimension in all_dimensions:
        covariance_matrix = np.zeros((dimension, dimension))
        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    covariance_matrix[i, j] = 1
                elif abs(i - j) == 1:
                    covariance_matrix[i, j] = 0.9
                else:
                    covariance_matrix[i, j] = 0

        COVARIANCE_MATRIX = covariance_matrix

        # Quick fix for singular covariance matrix.
        # (https://stackoverflow.com/questions/41515522/numpy-positive-semi-definite-warning)
        min_eig = np.min(np.real(np.linalg.eigvals(COVARIANCE_MATRIX)))
        if min_eig < 0:
            COVARIANCE_MATRIX -= 10 * min_eig * np.eye(dimension)

        BASE_MEASURE = scipy.stats.multivariate_normal(mean=np.array([0 for _ in range(dimension)]),
                                                       cov=np.eye(dimension), allow_singular=True)

        simulated_data = _generate_simulated_data(BASE_MEASURE=BASE_MEASURE, COVARIANCE_MATRIX=COVARIANCE_MATRIX,
                                                  alpha=alpha, data_points=data_points,
                                                  truncation_level=TRUNCATION_LEVEL)

        simulated_data_all_dimensions[dimension] = simulated_data[:100]
        hold_out_all_dimensions[dimension] = simulated_data[100:]
    return simulated_data_all_dimensions, hold_out_all_dimensions


if __name__ == '__main__':
    data = get_simulated_data()
    print(data)
