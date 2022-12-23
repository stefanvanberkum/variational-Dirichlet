import numpy as np
import scipy


def _generate_simulated_data(BASE_MEASURE, COVARIANCE_MATRIX, alpha, data_points, truncation_level):
    weights = np.ones(shape=(truncation_level))
    betas = np.random.beta(alpha, 1, size=(truncation_level))
    weights[0] = betas[0]
    weights[1:] = betas[1:] * (1 - betas[:-1]).cumprod()

    data = np.random.multinomial(1, weights, size=data_points)

    mus_all_clusters = BASE_MEASURE.rvs(size=truncation_level)

    tbret = []
    for data_point_ in data:
        cluster = np.where(data_point_ == 1)
        mean_cluster = mus_all_clusters[cluster].reshape(-1)
        simulated_data_point = np.random.multivariate_normal(mean=mean_cluster,
                                                             cov=COVARIANCE_MATRIX,
                                                             size=1)
        tbret.append(simulated_data_point)

    return np.array(tbret)


def get_simulated_data():
    all_dimensions = [5, 10, 20, 30, 40, 50]
    TRUNCATION_LEVEL = 20
    alpha = 1
    data_points = 100
    simulated_data_all_dimensions = {}
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

        BASE_MEASURE = scipy.stats.multivariate_normal(mean=np.array([0 for _ in range(dimension)]),
                                                       cov=np.ones((dimension, dimension)),
                                                       allow_singular=True)

        simulated_data = _generate_simulated_data(BASE_MEASURE=BASE_MEASURE,
                                                  COVARIANCE_MATRIX=COVARIANCE_MATRIX,
                                                  alpha=alpha,
                                                  data_points=data_points,
                                                  truncation_level=TRUNCATION_LEVEL)

        simulated_data_all_dimensions[dimension] = simulated_data
    return simulated_data_all_dimensions


if __name__ == '__main__':
    data = get_simulated_data()
    print(data)