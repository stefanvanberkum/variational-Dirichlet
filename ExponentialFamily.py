import numpy as np

from scipy.stats import multivariate_normal


class Multivariate_Normal_known_variance():
    def __init__(self, cov, etas = None) -> None:
        cov = np.array(cov)
        self.cov = cov
        self.precision = np.linalg.inv(cov)
        self.det = np.linalg.det(cov)

        if etas is None:
            mu = np.zeros(len(cov))
            etas = self.precision@mu
        self.etas = etas 

    def rvs(self, N, eta = None, random_state = None):
        if eta is not None:
            mu = self.theta(eta.T)
        else:
            mu = self.etas

        return multivariate_normal.rvs(mu, self.cov, size = N, random_state=random_state)

    def h(self, x): #where every element in x is a row vector
        #dim = len(x)
        #v1 = 1/np.sqrt(((2*np.pi)**dim)*self.det)
        #v2 = np.exp(-0.5*(x.T@self.precision@x))
        #return v1*v2
        return np.exp(self.log_h(x))

    def log_h(self, x):
        dim = len(x)
        v1 = -0.5*(x.T@self.precision@x)
        v2 = (dim/2)*np.log(2*np.pi)
        v3 = 0.5*np.log(self.det)
        return v1 - v2 - v3


    def T(self, x):
        return x

    def A(self, eta):
        mu = self.theta(eta)
        return 0.5*(mu.T@self.precision@mu)
      
    def theta(self, eta): #calculate parameters theta from natural parameters eta
        mu = np.linalg.solve(self.precision, eta)
        return mu

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        mu = theta
        return self.inverse@mu

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x, eta):
        return np.log(self.h(x)) + x.T@self.T(x) - self.A(eta)
