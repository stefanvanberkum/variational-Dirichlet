import numpy as np

#Do we have do make our own?
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import beta


from scipy.special import psi as digamma
from scipy.special import multigammaln

class ExponentialFamily:
    def __init__(self, dimension) -> None:
        self.dimension = dimension 
        "The dimension of the sufficient statistics, T, and the natural parameter, eta"

    def h(self, x):
        pass

    def T(self, x):
        pass

    def A(self, eta):
        pass

    def theta(self, eta): #calculate parameters theta from natural parameters eta
        pass

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        pass

    def rvs(self, N, eta, random_state = None):
        pass

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x, eta):
        return np.log(self.h(x)) + x.T@self.T(x) - self.A(eta)

    def _empty(self):
        """
        @return: An empty array of the correct size for T or eta.
        """
        return np.empty((self.dimension,), dtype=np.float64)

    def _check_shape(self, arg):
        """
        @return: True if the argument, for example T or eta, is the correct shape.
        """
        return arg.shape == (self.dimension,)


class Multivariate_Normal_known_variance(ExponentialFamily):
    def __init__(self, cov, k) -> None:
        cov = np.array(cov)
        self.cov = cov
        self.precision = np.linalg.inv(cov)
        self.det = np.linalg.det(cov)
        self.dim = k

        super().__init__(k)

    def rvs(self, N, eta = None, random_state = None):
        if eta is not None:
            mu = self.theta(eta.T)
        else:
            mu = self.etas

        return multivariate_normal.rvs(mu, self.cov, size = N, random_state=random_state)

    def h(self, x): #where every element in x is a row vector
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


class Multivariate_Normal(ExponentialFamily):
    def __init__(self, k) -> None:
        ExponentialFamily.__init__(self, k * (k + 1))
        self.dim = k

    def rvs(self, eta, N, random_state = None):
        mu, cov = self.theta(eta)
        return multivariate_normal.rvs(mu, cov, size = N, random_state=random_state)

    def h(self, x):
        return np.exp(self.log_h(x))

    def log_h(self, x):
        return 0.0

    def T(self, x):
        "@return: T(x), the sufficient statistics of x"
        x = np.asarray(x)
        T = self._empty()
        T[:self.dim] = x
        T[self.dim:] = -np.outer(x, x).reshape((self.dim**2,))
        return T

    def A(self, eta):
        mu, W = self.theta(eta)
        det = np.linalg.det(W)
        return np.array([
            .5 * (self.dim*np.log(2*np.pi) - np.log(det(W))),
            .5 * np.dot(np.dot(mu.T,W),mu)
        ])
      
    def theta(self, eta): #calculate parameters theta from natural parameters eta
        eta1, eta2 = eta
        W = eta2*2
        return (np.linalg.solve(W,eta1), eta2*2)

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        mu, W = theta
        eta = self._empty()
        eta[:self.dim] = np.dot(W, mu)
        eta[self.dim:] = .5 * W.reshape((self.dim**2,))
        return eta

    def exp_T(self, eta):
        (mu, W) = self.theta(eta)
        k = mu.shape(2)
        return (mu, -(np.linalg.inv(W)+np.outer(mu,mu)).reshape((k**2,)))



class Wishart(ExponentialFamily):

    def __init__(self, p=2):
        """
        Initialise this exponential family.
        @arg p: the number of dimensions
        """
        ExponentialFamily.__init__(self, p ** 2 + 1)

        self.dim = p
        "The number of dimensions."

    def h(self, x):
        return np.ones(len(x))

    def T(self, x):
            "@return: T(x), the sufficient statistics of x"
            x = np.asarray(x)
            T = self._empty()
            T[0] = np.log(np.linalg.det(x))
            T[1:] = x.reshape(self.dim**2,)
            return T

    def exp_T(self, eta):
            """
            @arg eta: The natural parameters.
            @return: The expectation of T, the sufficient statistics, given eta.
            """
            assert self._check_shape(eta)
            n, V = self.theta(eta)
            T = self._empty()
            T[0] = self.dim*np.log(2.) + np.log(np.linalg.det(V)) + sum(digamma((n-i)/2.) for i in range(self.dim))
            T[1:] = (n*V).reshape(self.dim**2,)
            return T

    def x(self, T):
            "@return: x(T), the x that has the sufficient statistics, T"
            return np.asarray(T[1:].reshape((self.dim, self.dim)))

    def eta(self, theta):
            "@return: eta(theta), the natural parameter, eta, that corresponds to the canonical parameter, theta"
            (n, V) = self._n_V_from_theta(theta)
            eta = self._empty()
            eta[0] = n-self.dim-1.
            eta[1:] = (-np.linalg.inv(V)).reshape((self.dim**2,))
            eta /= 2.
            return eta

    def theta(self, eta):
            "@return: theta(eta), the canonical parameter, theta, that corresponds to the natural parameter, eta"
            assert self._check_shape(eta)
            n = 2.*eta[0] + 1. + self.dim
            V = -np.linalg.inv(2.*eta[1:].reshape((self.dim,self.dim)))
            return (n, V)

    def A(self, eta):
            "@return: The normalization factor (log partition)"
            n, V = self.theta(eta)
            return np.array((.5 * (
                n * (self.dim * np.log(2.) + np.log(np.linalg.det(V)))
            ) + multigammaln(n/2., self.dim),))

    def rvs(self, eta, size=1):
            """
            @param eta: the natural parameters
            @param size: the size of the sample
            @return: A sample of sufficient statistics

            This uses the method detailed by
            U{Smith & Hocking<http://en.wikipedia.org/wiki/Wishart_distribution#Drawing_values_from_the_distribution>}.
            """
            from scipy.stats import norm, chi2
            X = np.empty((size, self.dimension), np.float64)
            n, V = self.theta(eta)
            L = np.linalg.cholesky(V)
            std_norm = norm(0,1)
            for sample_idx in range(size):
                while True: # avoid singular matrices by resampling until the determinant is != 0.0
                    A = np.zeros((self.dim, self.dim), dtype=np.float64)
                    for i in range(self.dim):
                        A[i,:i] = std_norm.rvs(size=i)
                        A[i,i] = np.sqrt(chi2.rvs(n-i))
                    if np.linalg.det(A) != 0.0:
                        break
                X[sample_idx] = self.T(np.dot(L, np.dot(A, np.dot(A.T, L.T))))
            return X

    def exp_log_det_W(self, eta):
            """
            Return the expectation of the log of the determinant of W given eta
            @arg eta: Natural parameters
            @return: log|det(W)|
            """
            n, V = self.theta(eta)
            return self.dim * np.log(2.) + np.log(np.linalg.det(V)) + sum(digamma((n-i)/2.) for i in range(self.dim))



class NormalWishart(ExponentialFamily):
    """
    A Normal-Wishart distribution in p dimensions.
    This is a conjugate prior for the multivariate normal distribution.

     - M{W ~ Wishart(nu,S)}
     - M{mu|W ~ Normal(mu_0, inv(W)/kappa_0)}

    where the parameters are:

     - M{nu} : degrees of freedom of precision
     - M{S} : precision
     - M{mu_0} : mean
     - M{kappa_0} : prior strength

    The exponential family parameterisation:

     - x = (mu, W)
     - T(x) = [(log|W|-p.log(2.pi))/2, -mu'.W.mu/2, self.mvn.eta(x)]
     - theta = (nu, S, kappa_0, mu_0)
     - eta(theta) = [kappa_0, nu-p, -kappa_0.mu_0, kappa_0.mu_0.mu_0'-inv(S)]
     - A(theta) = p/2[(p+1)log 2 - (nu-p-1)log(pi) - log kappa_0] + nu/2 log|S| + log Gamma_p(nu/2)
    """

    def __init__(self, p=2):
        """
        Initialise this exponential family.
        @arg p: the number of dimensions
        """
        self.wishart = Wishart(p)
        "The Wishart distribution"

        self.mvn = Multivariate_Normal(p)
        "The Normal distribution"

        ExponentialFamily.__init__(self, self.mvn.dimension + 2)

        self.dim = p
        "The dimension of mu"

    def h(self, x):
        return 1.0

    def T(self, x):
          "@return: T(x), the sufficient statistics of x"
          mu, W = x
          T = self._empty()
          T[2:] = self.mvn.eta(x)
          T[0] = (np.log(np.linalg.det(W)) - self.dim * np.log(2*np.pi)) / 2.
          T[1] = - np.dot(T[2:2+self.dim], mu) / 2.
          return T

    def exp_T(self, eta):
            """
            @arg eta: The natural parameters.
            @return: The expectation of T, the sufficient statistics, given eta.
            """
            nu, S, _kappa_0, mu_0 = self.theta(eta)
            T = self._empty()
            T[0] = (np.log(np.linalg.det(S)) + sum(digamma((nu-i)/2.) for i in range(self.dim)) - self.dim*np.log(np.pi))/2.
            v = self.mvn.eta((mu_0, nu*S))
            T[2:] = self.mvn.eta((mu_0, nu*S))
            T[1] = - np.dot(mu_0, T[2:2+self.dim]) / 2.
            return T

    def x(self, T):
            "@return: x(T), the x that has the sufficient statistics, T"
            assert self._check_shape(T), 'T is not the correct shape: %s' % str(T.shape)
            return self.mvn.theta(T[2:])

    def eta(self, theta):
            "@return: eta(theta), the natural parameter, eta, that corresponds to the canonical parameter, theta"
            nu, S, kappa_0, mu_0 = self._unpack_theta(theta)
            eta = self._empty()
            eta[0] = nu - self.dim
            eta[1] = kappa_0
            eta[2:2+self.dim] = kappa_0 * mu_0
            eta[2+self.dim:] = -(kappa_0 * np.outer(mu_0, mu_0) + np.linalg.inv(S)).reshape((self.dim**2,))
            return eta

    def theta(self, eta):
            "@return: theta(eta), the canonical parameter, theta, that corresponds to the natural parameter, eta"
            assert self._check_shape(eta)
            nu = eta[0] + self.dim
            kappa_0 = eta[1]
            mu_0 = eta[2:2+self.dim] / kappa_0
            S = -np.linalg.inv(kappa_0 * np.outer(mu_0, mu_0) + eta[2+self.dim:].reshape((self.dim,self.dim)))
            return (nu, S, kappa_0, mu_0)

    def A(self, eta):
            "@return: The normalization factor (log partition)"
            nu, S, kappa_0, _mu_0 = self.theta(eta)
            return np.array(((
                ((self.dim+1.)*np.log(2.) - (nu-self.dim-1.)*np.log(np.pi) - np.log(kappa_0)) * self.dim / 2.
                + np.log(np.linalg.det(S)) * nu / 2.
                + multigammaln(nu/2., self.dim)
            ),))

    def rvs(self, eta, size=1):
            """
            @param eta: the natural parameters
            @param size: the size of the sample
            @return: A sample of sufficient statistics
            """
            #from IPython.Debugger import Pdb; Pdb().set_trace()
            samples = np.empty((size, self.dimension))
            nu, S, kappa_0, mu_0 = self.theta(eta)
            Ws = [self.wishart.x(W) for W in self.wishart.rvs(self.wishart.eta((nu, S)), size=size)]
            assert len(Ws) == size
            for i, W in enumerate(Ws):
                mu = self.mvn.x(self.mvn.rvs(eta=self.mvn.eta((mu_0,W*kappa_0)),size=1)[0])
                samples[i] = self.T((mu,W))
            assert len(samples) == size
            return samples

    def _unpack_theta(self, theta):
        "Extract components of theta and check/correct their shapes."
        (nu, S, kappa_0, mu_0) = theta

        S = np.asarray(S)
        assert S.shape == (self.dim, self.dim)

        mu_0 = np.asarray(mu_0)
        assert mu_0.shape == (self.dim,)

        return (nu, S, kappa_0, mu_0)


class MvNormalPrior(ExponentialFamily):
    #Conjugate prior of multivariable normal exponential family where the covariance matrix is known
    #
    def __init__(self, cov, k) -> None:
        cov = np.array(cov)
        self.dist = Multivariate_Normal_known_variance(cov, k)
        self.dim = k
        self.det = np.linalg.det(cov)

        super().__init__(k + 1)
        self.cov = cov #covariance matrix of the distribution that is sampled for
        self.precision = np.linalg.inv(cov)

    def h(self, x):
        return np.full(len(x), (2*np.pi)**(-self.dim/2))

    def T(self, x):
        pass

    def A(self, taus):
        lambda_1, lambda_2 = self.get_lambdas(taus)
        return ((lambda_1.T@self.precision.T)@lambda_1)/(2*lambda_2) + 0.5*self.dim*np.log(lambda_2*self.det)

    def theta(self, taus): #calculate parameters theta from natural parameters eta
        lambda_1, lambda_2 = self.get_lambdas(taus)
        cov_0 = self.precicion / lambda_2
        mu_0 = (self.precision@lambda_1)/lambda_2
        return (mu_0, cov_0)

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        mu_0, cov_0 = theta
        lambda_2 = self.precision[0, 0]/cov_0[0, 0] #only need to calculate one fraction
        lambda_1 = np.linalg.solve(self.precicion, mu_0*lambda_2)
        return (lambda_1, lambda_2)

    def expected_x(self, taus):
        expected_values = np.empty((len(taus),self.dim))
        for k, tau in enumerate(taus): 
            lambda_1, lambda_2 = self.get_lambdas(tau)
            expected_values[k,:] = (self.precision@lambda_1)/lambda_2
        return expected_values

    def exp_T(self, taus):
        T = self._empty()
        T[-self.dim:] = self.expected_x(taus[None,:])
        v = T[self.dim:]
        T[:-self.dim] = 0
        return T

    def rvs(self, N, taus):
        taus = np.array(taus)
        lambda_1, lambda_2 = self.get_lambdas(taus)

        cov_0 = self.precision/lambda_2
        mu_0 =  cov_0@lambda_1
        return multivariate_normal.rvs(mu_0, cov_0, size = N)


    def get_lambdas(self, etas):
        
        if len(etas[:,np.newaxis].shape) == self.dim + 1:
            #etas = np.flip(etas, axis = 1)
            lambda_2, lambda_1 = np.array_split(etas,[1] ,axis = 1)
        else:
            #etas = np.flip(etas)
            lambda_2, lambda_1 = np.array_split(etas, [1])
        lambda_1 = lambda_1.T
        return (lambda_1, lambda_2)

    def predictive_prob(self, X, taus):
        return np.exp(self.predictive_log_prob(X, taus))

    def log_pdf(self, x):
        pass



class ConjugatePrior:
    def __init__(self, likelihood, prior) -> None:
        self.likelihood = likelihood 
        self.prior = prior 
        self.likelihood_dimension = self.likelihood.dimension
        "The dimension of the natural parameters and sufficient statistics of the likelihood."

        self.strength_dimension = self.prior.dimension - self.likelihood_dimension
        "The dimension of the strength part of the prior's natural parameters."

    def predictive_log_prob(self, X, taus):
        #dim = len(X)
        Lambda = taus.copy()
        Lambda[:self.strength_dimension] += 1
        Lambda[self.strength_dimension:] += X
        v1 = self.prior.A(Lambda)
        v2 = self.prior.A(taus)
        v3 = self.likelihood.log_h(X)
        return self.prior.A(Lambda) - self.prior.A(taus) + self.likelihood.log_h(X)

class MvnConjugatePrior(ConjugatePrior):
    """
    Multivariate Normal distribution with a Normal-Wishart conjugate prior.
    """
    def __init__(self, p=2):
        "Initialise the Normal-Wishart - Multivariate Normal conjugate prior pair"
        ConjugatePrior.__init__(self, Multivariate_Normal(p), NormalWishart(p))

    def expected_A(self, tau):
        """
        @arg tau: the parameters of the prior in standard form
        @return: the expectation of the log normalisation factor of the likelihood given the prior's parameters
        """
        nu, S, kappa_0, mu_0 = self.prior.theta(tau)
        return (
         self.prior.dim / kappa_0
         + nu * np.dot(mu_0, np.dot(S, mu_0))
         + self.prior.dim * np.log(np.pi)
         - np.log(np.linalg.det(S))
         - sum(digamma((nu-i)/2.) for i in range(self.prior.dim))
        ) / 2.

class MvnKnownCovConjugatePrior(ConjugatePrior):
    """
    Multivariate Normal distribution with a Normal-Wishart conjugate prior.
    """
    def __init__(self, cov, p=2):
        "Initialise the Normal-Wishart - Multivariate Normal conjugate prior pair"
        ConjugatePrior.__init__(self, Multivariate_Normal_known_variance(cov, p), MvNormalPrior(cov, p))

    #def expected_A(self, taus):
    #    expected_values = np.empty(len(taus))
    #    for k, tau in enumerate(taus): 
    #        lambda_1, lambda_2 = self.prior.get_lambdas(tau)
    #        expected_values[k] = (self.likelihood.precision@lambda_1).T@lambda_1/(2*(lambda_2**2)) + self.likelihood.dim/(2*lambda_2)
    #    return expected_values

    def expected_A(self, tau):
        lambda_1, lambda_2 = self.prior.get_lambdas(tau)
        expected_values = (self.likelihood.precision@lambda_1).T@lambda_1/(2*(lambda_2**2)) + self.likelihood.dim/(2*lambda_2)
        return expected_values

    def log_expect(self, etas):
        LL = 0.0
        for Lambda in etas:
            lambda_1, lambda_2 = self.prior.get_lambdas(Lambda)
            LL += self.prior.h(Lambda[np.newaxis,:]) + self.prior.expected_x(Lambda[np.newaxis,:])@lambda_1 - lambda_2*self.expected_A(Lambda[np.newaxis,:])   - self.prior.A(Lambda[np.newaxis,:])
        
        return LL


    """ Unflipped prefered version
    class MvNormalPrior(ExponentialFamily):
    #Conjugate prior of multivariable normal exponential family where the covariance matrix is known
    #
    def __init__(self, cov, k) -> None:
        cov = np.array(cov)
        self.dist = Multivariate_Normal_known_variance(cov, k)
        self.dim = len(cov)
        self.strength_dim = 1 #dimension for scalar value
        self.det = np.linalg.det(cov)

        super().__init__(k + 1)
        self.cov = cov #covariance matrix of the distribution that is sampled for
        self.precision = np.linalg.inv(cov)

    def h(self, x):
        return np.full(len(x), (2*np.pi)**(-self.dim/2))

    def T(self, x):
        pass

    def A(self, taus):
        lambda_1, lambda_2 = self.get_lambdas(taus)
        return ((lambda_1.T@self.precision.T)@lambda_1)/(2*lambda_2) + 0.5*self.dim*np.log(lambda_2*self.det)

    def theta(self, taus): #calculate parameters theta from natural parameters eta
        lambda_1, lambda_2 = self.get_lambdas(taus)
        cov_0 = self.precicion / lambda_2
        mu_0 = (self.precision@lambda_1)/lambda_2
        return (mu_0, cov_0)

    def eta(self, theta): #calculate natural parameters eta from parameters theta
        mu_0, cov_0 = theta
        lambda_2 = self.precision[0, 0]/cov_0[0, 0] #only need to calculate one fraction
        lambda_1 = np.linalg.solve(self.precicion, mu_0*lambda_2)
        return (lambda_1, lambda_2)

    def expected_x(self, taus):
        expected_values = np.empty((len(taus),self.dim))
        for k, tau in enumerate(taus): 
            lambda_1, lambda_2 = self.get_lambdas(tau)
            expected_values[k,:] = (self.precision@lambda_1)/lambda_2
        return expected_values

    def exp_T(self, taus):
        T = self._empty(len(taus))
        T[:,:self.dim] = self.expected_x(taus)
        v = T[:,self.dim:]
        T[:,self.dim:] = self.expected_A(taus)[:,None]
        return T

    def expected_A(self, taus):
        expected_values = np.empty(len(taus))
        for k, tau in enumerate(taus): 
            lambda_1, lambda_2 = self.get_lambdas(tau)
            expected_values[k] = (self.precision@lambda_1).T@lambda_1/(2*(lambda_2**2)) + self.dim/(2*lambda_2)
        return expected_values

    def rvs(self, N, taus):
        taus = np.array(taus)
        lambda_1, lambda_2 = self.get_lambdas(taus)

        cov_0 = self.precision/lambda_2
        mu_0 =  cov_0@lambda_1
        return multivariate_normal.rvs(mu_0, cov_0, size = N)

    def log_expect(self, etas):
        LL = 0.0
        for Lambda in etas:
            lambda_1, lambda_2 = self.get_lambdas(Lambda)
            LL += self.h(Lambda[np.newaxis,:]) + self.expected_x(Lambda[np.newaxis,:])@lambda_1 - lambda_2*self.expected_A(Lambda[np.newaxis,:])   - self.A(Lambda[np.newaxis,:])
        
        return LL

    def get_lambdas(self, etas):
        if len(etas[:,np.newaxis].shape) == self.dim + 1:
            lambda_1, lambda_2 = np.array_split(etas, etas.shape[1]-1, axis = 1)
        else:
            lambda_1, lambda_2 = np.array_split(etas, len(etas)-1)
        lambda_1 = lambda_1.T
        return (lambda_1, lambda_2)

    def predictive_prob(self, X, taus):
        return np.exp(self.predictive_log_prob(X, taus))

    def predictive_log_prob(self, X, taus):
        #dim = len(X)
        Lambda = taus.copy()
        Lambda[-1] += 1
        Lambda[:-1] += X
        v1 = self.dist.log_h(X)
        v2 = self.A(Lambda)
        v3 =  self.A(taus)
        v = self.A(Lambda) - self.A(taus) + self.dist.log_h(X)
        return self.A(Lambda) - self.A(taus) + self.dist.log_h(X)

    def log_pdf(self, x):
        pass
    """



