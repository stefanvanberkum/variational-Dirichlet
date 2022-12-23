from ast import Lambda
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi as digamma
from scipy.special import betaln

#RANDOM_SEED = 1
RANDOM_SEED = None
np.random.seed(RANDOM_SEED)

#Do we have do make our own?
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import beta

from ExponentialFamily import Multivariate_Normal_known_variance, Multivariate_Normal, \
MvNormalPrior, NormalWishart, MvnKnownCovConjugatePrior, MvnConjugatePrior

#GMM consisting of Gaussians with diagonal covariance matrices
# **Used for testing** Use scikit impl instead?
class GaussianMixtureModel:
    def __init__(self, mus, sigmas, weights) -> None:
        #Mus: Array of means for each gaussian component one value per dimension
        #Sigmas: Array of diagonal elements of each covariance matrix
        #Weights: probabilities of picking reapective gaussian components i.e. mixing proportions of the model
        assert mus.shape == sigmas.shape 
        assert len(weights) == len(mus)


        self.mus = np.array(mus)
        self.sigmas = np.array(sigmas)
        self.weights = np.array(weights)
        self.components = [multivariate_normal(self.mus[i], np.diag(self.sigmas[i])) for i in range(len(mus))]

    #sample N random samples from the distribution
    def rvs(self, N):
        pass

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        pass


#Dirichlet Process
class DPMixtureModel:
    def __init__(self, DP, Exp_family_dist) -> None:
        self.DP = DP
        self.Exp_family_dist = Exp_family_dist

    #sample N random samples from the distribution
    def rvs(self, N, Lambda):
        Z, etas = self.DP.rvs(N, Lambda)
        v = np.argwhere(Z)
        indices = np.squeeze(np.argwhere(Z), axis = 1)
        dim = self.Exp_family_dist.dim
        X = np.empty((0,dim))

        for i in indices:
            X = np.append(X, self.Exp_family_dist.rvs(Z[i], etas[i]))

        return np.reshape(X, (N, dim))

    def pdf(self, x):
        return np.exp(self.log_pdf(x))

    def log_pdf(self, x):
        pass


#Stick breaking model described in the paper
class SBModel:
    def __init__(self, K, G0, alpha = 1) -> None:
        self.K = K 
        self.G0 = G0
        self.Beta = beta(1, alpha)

    #sample component number and parameters for K components and N values
    #returns component choice for every n i.e. zn for every xn 
    #returns parameters ie eta_k 
    def rvs(self, N, Lambda):
        V = self.Beta.rvs(size = self.K, random_state = RANDOM_SEED)
        V[self.K - 1] = 1
        thetas = self.get_thetas(V)
        eta_star = self.G0.rvs(N, Lambda)

        Z = np.random.multinomial(N, thetas)
        
        return Z, eta_star

    def get_thetas(self, V):
        factors = np.cumprod(1-V)
        thetas = np.zeros(self.K)
        thetas[0] = V[0]

        for i in range(1, self.K):
            thetas[i] = V[i]*factors[i-1]

        return thetas 

#VI DP process
class VI_SBmodel:
    def __init__(self, ConjugatePrior, alpha, _lambda) -> None:
        self.gammas = None 
        self.taus = None 
        self.phis = None 
        self.ConjugatePrior = ConjugatePrior
        self.prior = ConjugatePrior.prior
        self.dist = ConjugatePrior.likelihood
        self.dim = 0
        self.X = None
        self.K = None
        self.N = 0
        self.Beta = beta(1, alpha)
        self.d = ConjugatePrior.strength_dimension

        #hyper-parameters
        self.alpha = alpha
        self._lambda = _lambda 


    def rvs(self, N):
        pass

    def fit(self, X, K, maxiter = 10, limit = 1e-3):
        self.X = X
        self.K = K
        self.dim = X.shape[1]
        #d2 = len(self._lambda)
        #self._lambda = np.reshape(np.repeat(self._lambda, repeats = self.K), (self.K, d2))
        self.N = len(X)
        self.gammas = np.empty((self.K-1, 2))
        self._randomise()
        diff = 10e20
        old_ELBO = diff 
        i = 0
        while i<maxiter and diff > limit:
            self.__update_taus()
            self.__update_phis()
            #self._update_order()
            self.__update_gammas()

            #ELBO = self.ELBO()
            #print("ELBO:", ELBO)
            #diff = np.abs(ELBO - old_ELBO)
            #
            #old_ELBO = ELBO
            print(i)
            i += 1
        print(i)


    def __update_phis(self):
        #expected values logV_i and log(1 - V_i)
        expected_log_v, expected_log_ones_minus_v = self.v_expectations()
        expected_log_v = np.append(expected_log_v, 0)
        expected_log_ones_minus_v = np.append(expected_log_ones_minus_v, 0)


        #expected_As = self.ConjugatePrior.expected_A(self.taus)
        expected_As = np.array([self.ConjugatePrior.expected_A(tau) for tau in self.taus])
        #expected_etas =  self.prior.exp_T(self.taus)
        expected_etas = np.array([self.prior.exp_T(tau) for tau in self.taus])

        for n in range(self.N):
            for k in range(self.K):
                v1 = expected_log_v[k]
                v12 = expected_etas[k,self.d:]
                v2 =  np.dot(expected_etas[k,self.d:], self.X[n])
                v3 =  expected_As[k]
                E = expected_log_v[k] + np.dot(expected_etas[k,self.d:], self.X[n]) -  expected_As[k]
                #E = expected_log_v[k] + self.X[n].T@expected_etas[k] -  expected_As[k] 
                for j in range(k):
                    E += expected_log_ones_minus_v[j]
                self.phis[n,k] = E

        # the terms that don't depend on the X[n]s
        #sum_E_log_1_minus_Vj = np.array([expected_log_ones_minus_v[:i].sum() for i in range(self.K)])
        #partial_E = expected_log_v + sum_E_log_1_minus_Vj - expected_As
        #
        ## for each datum reestimate phi
        ##import IPython; IPython.Debugger.Pdb().set_trace()
        #for n in range(self.N):
        #    E = partial_E.copy() + np.dot(expected_etas, self.X[n])
        #    E_exp = np.exp(E-E.max()) # scale to avoid numerical errors
        #    self.phis[n,:] = E_exp/E_exp.sum()



        #E = np.zeros(self.K)
        #E[:-1] += expected_log_v
        #log_one_minus_v_cum_sum = np.cumsum(expected_log_ones_minus_v)
        #log_one_minus_v_cum_sum = np.roll(log_one_minus_v_cum_sum, 1)
        #log_one_minus_v_cum_sum[0] = 0
        #E[:-1] += log_one_minus_v_cum_sum
        #
        #expected_As = self.prior.expected_A(self.taus)
        #expected_etas =  self.prior.expected_eta(self.taus)
        #
        #E_base = E.copy()
        #for n in range(self.N):
        #    E = E_base.copy()
        #    for k in range(self.K):
        #        E[k] += self.X[n].T@expected_etas[k] - expected_As[k]
        #    self.phis[n,:] = E
        #
        m = np.max(self.phis, axis = 1)
        self.phis = np.exp(self.phis - m[:,None])
        sums = np.sum(self.phis, axis = 1)
        for n in range(self.N):
            for k in range(self.K):
                if self.phis[n, k] != 0.0:
                    self.phis[n, k] /= sums[k]
        #self.phis = self.phis/np.sum(self.phis, axis = 1)[:,None]
        v = expected_etas/2
        d = 2

    def __update_gammas(self):

        for k in range(self.K - 1):
             self.gammas[k,0] = 1 
             for n in range(self.N):
                 self.gammas[k,0] += self.phis[n,k]

        for k in range(self.K-1):
            self.gammas[k,1] = self.alpha
            for n in range(self.N):
                for j in range(k+1, self.K):
                    self.gammas[k,1] += self.phis[n,j]

        #phisum = np.sum(self.phis, axis = 0)
        #v1 = np.delete(1 + phisum.T, self.K-1)
        #v2 = self.gammas[:,0]
        #self.gammas[:,0] = np.delete(1 + phisum.T, self.K-1)
        #phicumsum = np.cumsum(np.flip(phisum))
        #phicumsum = np.flip(phicumsum)
        #phicumsum = np.roll(phicumsum, -1)
        #v3 = np.delete(phicumsum, self.K-1)
        #v4 = self.gammas[:,1]
        #self.gammas[:,1] = self.alpha + np.delete(phicumsum, self.K-1)

    def __update_taus(self):

        for k in range(self.K):
            self.taus[k,self.d:] = self._lambda[self.d:]
            for n in range(self.N):
                self.taus[k,self.d:] += self.phis[n,k]*self.X[n]

        for k in range(self.K):
            self.taus[k,:self.d] =  self._lambda[:self.d]
            for n in range(self.N):
                v4 = self._lambda[-1:]
                v6 = self.taus[k,-1:]
                self.taus[k,:self.d] += self.phis[n,k]


        #phisum = np.sum(self.phis, axis = 0)
        #for i in range(self.K):
        #    v1 = self._lambda[:self.dim]
        #    w = self.phis[:,i][:,None]
        #    v2 = np.sum(self.phis[:,i][:,None]*self.X, axis = 0)
        #    self.taus[:,1:] = self._lambda[:self.dim] + np.sum(self.phis[:,i][:,None]*self.X, axis = 0)
        #v3 = self._lambda[1:]
        #self.taus[:,:1] = (self._lambda[1:] + phisum)[:,None]


    def _update_order(self):
        """
        Reorders components to make largest first. Does not reorder gamma so _update_gamma should be
        called next in updating order.
        """
        expected_component_sizes = -self.phis.sum(axis=0)
        permutation = expected_component_sizes.argsort()
        new_phi = np.empty_like(self.phis)
        for n in range(self.N):
            for k in range(self.K):
                new_phi[n,k] = self.phis[n,permutation[k]]
        self.phis = new_phi
        new_tau = np.empty_like(self.taus)
        for k in range(self.K):
            new_tau[k] = self.taus[permutation[k]]
        self.taus = new_tau
        expected_component_sizes = self.phis.sum(axis=0)


    def _randomise(self): #TODO rewrite
        """
        Randomise the variational parameters.
        """
        from numpy.random import dirichlet
        self.phis = dirichlet(np.ones(self.K), size=self.N)
        self.taus = np.outer(np.ones(self.K), self._lambda)
        self.gammas[:,0] = 1.
        self.gammas[:,1] = self.alpha


    def get_thetas(self):
        "@return: An array specifying the probability for a new point coming from each component."
        # chance of selecting this component given have not selected any of previous
        p_v_i_one = self.gammas[:,0] / self.gammas.sum(axis=1)

        p_select_component = np.empty(self.K) # result
        stick_left = 1.0 # amount of stick left in stick breaking representation of dirichlet process
        for k, q in enumerate(p_v_i_one):
            # chance of selecting this component
            p_select_component[k] = stick_left * q
            stick_left *= (1. - q) # reduce amount of stick left
        p_select_component[-1] = stick_left
        return p_select_component

    def ELBO(self):
        expected_log_v, expected_log_ones_minus_v = self.v_expectations()

        #LL for Vs
        alphas, betas = np.split(self.gammas, 2, axis = 1)
        alphas = np.squeeze(alphas)
        betas = np.squeeze(betas)
        value = (alphas - 1)*expected_log_v + (betas - 1)*expected_log_ones_minus_v - betaln(alphas, betas)
        ELBO = np.sum(value)
        value = (self.alpha - 1)*expected_log_ones_minus_v - betaln(1, self.alpha)
        ELBO -= np.sum(value)

        #LL eta
        d2 = len(self._lambda)
        Lambda = np.repeat(self._lambda[np.newaxis,:], repeats = self.K, axis = 0)
        ELBO += self.ConjugatePrior.log_expect(Lambda)
        ELBO -= self.ConjugatePrior.log_expect(self.taus)
        ELBO = np.squeeze(ELBO)

        #LL Z
        for n in range(self.N):
            phi = self.phis[n].copy()
            phi_cum_sum = np.flip(np.cumsum(np.flip(phi[:,np.newaxis])))
            #phi_cum_sum.squeeze(axis = 1)
            phi_cum_sum = np.roll(phi_cum_sum, -1)
            phi_cum_sum[-1] = 0

            #avoid cases where phi is 0
            #phi[phi == 0.0] = 1.0

            ELBO += np.sum(phi_cum_sum*np.append(expected_log_ones_minus_v, 0) + phi*np.append(expected_log_v, 0))


        #LL X
        expected_A = np.array([self.ConjugatePrior.expected_A(tau) for tau in self.taus])
        expected_eta = np.array([self.prior.exp_T(tau) for tau in self.taus])

        #expected_eta = self.prior.exp_T(self.taus)
        #expected_A = self.prior.expected_A(self.taus)
        for n, x in enumerate(self.X):
            for k in range(self.K):
                if self.phis[n,k] != 0.0:
                    #v1 = self.phis[n,k]
                    v2 = self.dist.log_h(x)
                    v3 = np.squeeze(x.T@expected_eta[k,self.d:].T)
                    v4 = expected_A[k]
                    v5 = self.phis[n, k]*(self.dist.log_h(x) + np.squeeze(x.T@expected_eta[k,self.d:].T) - expected_A[k] - np.log(self.phis[n, k]))
                    ELBO += np.squeeze(v5)

        return ELBO


    def predictive_log_prob(self, X):
        N = len(X)
        mixing_proportions = self.get_thetas()
        T = self.ConjugatePrior.likelihood.T(X)
        log_predictive = np.array([self.ConjugatePrior.predictive_log_prob(T, self.taus[k]) for k in range(self.K)]).T
        return np.sum(np.exp(log_predictive) * mixing_proportions, axis = 1)

    def v_expectations(self):
        v = digamma(np.sum(self.gammas, axis = 1))
        expected_log_v = digamma(self.gammas[:,0]) - v
        expected_log_ones_minus_v = digamma(self.gammas[:,1]) - v
        return (expected_log_v, expected_log_ones_minus_v)


 ######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################
######################################################################TEST#######################################################

def sample_data(K, alpha, cov, cov2, Lambda, N_samples, dim = 2):
    G0 = MvNormalPrior(cov2, dim)
    SBP = SBModel(K, G0, alpha) 
    mixture_dist = Multivariate_Normal_known_variance(cov, dim)
    Dirichlet_process_MM = DPMixtureModel(SBP, mixture_dist)

    samples = Dirichlet_process_MM.rvs(N_samples, Lambda)
    return samples

def test_sampling():
    #samples = sample_data(K = 20, alpha = 0.1, cov = np.diag([1,1]), cov2 = np.diag([1000, 1000]), Lambda = [1,1,1], N_samples =1000)
    samples = sample_data(K = 20, alpha = 5, cov = np.diag([0.5,0.5]), cov2 = np.diag([0.01, 0.01]), Lambda = [1,1,1], N_samples =100)
    x, y = np.split(samples, 2, axis = 1)
    plt.scatter(x, y)
    plt.show()


#Only for testing
def expected_means(cov, taus):
    mus = np.empty_like(taus[:,1:])
    lambda_2, lambda_1 = np.array_split(taus, [1], axis = 1)
    prec = np.linalg.inv(cov)
    for k in range(len(taus)):
        mu_0 = prec@lambda_1[k]
        mu_0 /= lambda_2[k]
        mus[k,:] = cov@mu_0
    print("expected center positions: ", mus)


def test_VI(samples):
    #samples = sample_data(K = 20, alpha = 0.1, cov = np.diag([2,2]), cov2 = np.diag([1000, 1000]), Lambda = [1,1,1], N_samples =100)
    #samples = sample_data(K = 20, alpha = 5, cov = np.diag([0.5,0.5]), cov2 = np.diag([0.01, 0.01]), Lambda = [1,1,1], N_samples =100)
    x, y = np.split(samples, 2, axis = 1)
    plt.scatter(x, y)
    #plt.show()
    samples_cov = np.cov(samples.T)
    sample_mean = np.mean(samples, axis = 0)

    #1
    #alpha = 1
    #d1, d2 = samples.shape
    #Lambda = np.empty(d2 + 1)
    #Lambda[-1] = 1
    ##Lambda[:d2] = np.squeeze((samples_cov@sample_mean[:,np.newaxis]).T, axis = 0)
    #Lambda[:d2] = sample_mean

    #2
    samples_cov = np.diag([0.5,0.5])
    alpha = 5
    Lambda = np.full(3, 0.1)
    Lambda[2] = 0.1 

    ConjugatePrior = MvnKnownCovConjugatePrior(samples_cov, 2)

    SBP_VI = VI_SBmodel(ConjugatePrior, alpha, Lambda)
    SBP_VI.fit(samples, 20, 20, 1e-8)
    expected_means(samples_cov, SBP_VI.taus)

    max_x, max_y = np.amax(samples, axis = 0)
    min_x, min_y = np.amin(samples, axis = 0)

    X = np.linspace(min_x, max_x, num = 50)
    Y = np.linspace(min_y, max_y, num = 50)
    Z = np.empty((len(X), len(Y)))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            #print("x:",x, "y:", y)
            Z[j, i] = np.exp(SBP_VI.predictive_log_prob(np.array([x, y])))

    plt.contour(X, Y, Z)
    plt.show()




def test_VI_unknown_variance(samples):
    #samples = sample_data(K = 20, alpha = 0.1, cov = np.diag([2,2]), cov2 = np.diag([1000, 1000]), Lambda = [1,1,1], N_samples =100)
    
    x, y = np.split(samples, 2, axis = 1)
    plt.scatter(x, y)
    #plt.show()
    samples_cov = np.cov(samples.T)
    sample_mean = np.mean(samples, axis = 0)

    #samples_cov = np.diag([1.0, 1.0])
    #sample_mean = np.array([0.0, 0.0])

    ConjugatePrior = MvnConjugatePrior(2)

    #1
    alpha = 10
    dim = 2
    Lambda = ConjugatePrior.prior.eta((dim + 1, samples_cov, 0.1, sample_mean))
    T = np.array([ConjugatePrior.likelihood.T(sample.T) for sample in samples])

    SBP_VI = VI_SBmodel(ConjugatePrior, alpha, Lambda)
    SBP_VI.fit(T, 20, 500, 1e-8)
    #expected_means(samples_cov, SBP_VI.taus)

    max_x, max_y = np.amax(samples, axis = 0)
    min_x, min_y = np.amin(samples, axis = 0)

    X = np.linspace(min_x, max_x, num = 50)
    Y = np.linspace(min_y, max_y, num = 50)
    Z = np.empty((len(X), len(Y)))

    for i, x in enumerate(X):
        for j, y in enumerate(Y):
            #print("x:",x, "y:", y)
            Z[j, i] = np.exp(SBP_VI.predictive_log_prob(np.array([x, y])))

    plt.contour(X, Y, Z)
    plt.show()






if __name__ == "__main__":
    #test_sampling()
    X = samples = sample_data(K = 20, alpha = 5, cov = np.diag([0.2,0.2]), cov2 = np.diag([0.001, 0.001]), Lambda = [1,1,1], N_samples =1000)
    #test_VI(X)
    test_VI_unknown_variance(X)
