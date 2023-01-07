import numpy as np

from scipy.stats import multivariate_normal

from ExponentialFamily import Multivariate_Normal_known_variance


class MvNormalPrior():
    #Conjugate prior of multivariable normal exponential family where the covariance matrix is known
    def __init__(self, cov, taus = None) -> None:
        cov = np.array(cov)
        self.dist = Multivariate_Normal_known_variance(cov)
        self.dim = len(cov)
        if taus is None:
            taus = np.ones(self.dim)

        self.taus = np.array(taus)
        self.cov = cov #covariance matrix of the distribution that is sampled for
        self.precision = np.linalg.inv(cov) #precision matrix of known covariance matrix
        self.det = np.linalg.det(cov)
        self.precision_det = np.linalg.det(self.precision)

    def h(self, x):
        return (2*np.pi)**(-self.dim/2)

    def T(self, x):
        pass

    def A(self, taus):
        lambda_1, lambda_2 = self.get_lambdas(taus)
        return (lambda_1.T@self.precision@lambda_1)/(2*lambda_2) - 0.5*self.dim*(np.log(lambda_2) + np.log(self.precision_det))

    def theta(self, taus): #calculate parameters theta from natural parameters eta
        lambda_1, lambda_2 = self.get_lambdas(taus)
        cov_0 = self.precicion / lambda_2
        mu_0 = (self.precision@lambda_1)/lambda_2
        return (mu_0, cov_0)

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        mu_0, cov_0 = theta
        lambda_2 = np.mean(self.precision/cov_0) #might not be exact fractions therefore take mean
        lambda_1 = np.linalg.solve(self.precision, mu_0*lambda_2)
        return (lambda_1, lambda_2)

    def expected_eta(self, taus):
        expected_values = np.empty((len(taus),self.dim))
        for k, tau in enumerate(taus): 
            lambda_1, lambda_2 = self.get_lambdas(tau)
            expected_values[k,:] = (self.precision@lambda_1)/lambda_2
        return expected_values

    def expected_A(self, taus):
        expected_values = np.empty(len(taus))
        for k, tau in enumerate(taus): 
            lambda_1, lambda_2 = self.get_lambdas(tau)
            expected_values[k] = lambda_1.T@self.precision@lambda_1/(2*(np.square(lambda_2))) + self.dim/(2*lambda_2)
        return expected_values

    def rvs(self, N, taus = None):
        if taus is not None:
            taus = np.array(taus)
            lambda_1, lambda_2 = self.get_lambdas(taus)
        else:
            lambda_1, lambda_2 = self.get_lambdas(self.taus)

        cov_0 = self.precision/lambda_2
        mu_0 =  cov_0@lambda_1
        return multivariate_normal.rvs(mu_0, cov_0, size = N)

    def log_expect(self, etas):
        LL = 0.0
        for Lambda in etas:
            lambda_1, lambda_2 = self.get_lambdas(Lambda)
            v1 = self.h(Lambda[np.newaxis,:])
            v2 = self.expected_eta(Lambda[np.newaxis,:])@lambda_1
            v3 = lambda_2*self.expected_A(Lambda[np.newaxis,:])
            v4 = self.A(Lambda[np.newaxis,:])
            LL += self.h(Lambda[np.newaxis,:]) + self.expected_eta(Lambda[np.newaxis,:])@lambda_1 - lambda_2*self.expected_A(Lambda[np.newaxis,:])   - self.A(Lambda[np.newaxis,:])
        
        return LL

    def predictive_log_prob(self, X, taus):
        Lambda = taus.copy()
        Lambda[-1] += 1
        Lambda[:-1] += X
        return self.A(Lambda) - self.A(taus) + self.dist.log_h(X)

    def get_lambdas(self, etas):
        if len(etas[:,np.newaxis].shape) == 3:
            lambda_1, lambda_2 = np.array_split(etas, [etas.shape[1]-1], axis = 1)
        else:
            lambda_1, lambda_2 = np.array_split(etas, [len(etas)-1])
        lambda_1 = lambda_1.T
        return (lambda_1, lambda_2)

    

    def log_pdf(self, x):
        pass
