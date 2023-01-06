import numpy as np
import matplotlib.pyplot as plt
from scipy.special import psi as digamma
from scipy.special import betaln

RANDOM_SEED = 1
np.random.seed(RANDOM_SEED)

from scipy.stats import multivariate_normal
from scipy.stats import beta

from ExponentialFamily import Multivariate_Normal_known_variance
from ExpPrior import MvNormalPrior

from data_simulation import generate_simulated_data, get_simulated_data


#GMM consisting of Gaussians with diagonal covariance matrices
# **Used for testing** Use scikit impl instead?
class GaussianMixtureModel:
    def __init__(self, mus, covs, weights) -> None:
        #Mus: Array of means for each gaussian component one value per dimension
        #Sigmas: Array of diagonal elements of each covariance matrix
        #Weights: probabilities of picking reapective gaussian components i.e. mixing proportions of the model
        assert len(weights) == len(mus)
        assert len(weights) == len(covs)


        self.mus = np.array(mus)
        self.covs = np.array(covs)
        self.weights = np.array(weights)
        self.dim = self.mus.shape[1]
        self.K = len(self.weights)
        self.components = [multivariate_normal(self.mus[i], np.diag(self.covs[i])) for i in range(len(weights))]

    #sample N random samples from the distribution
    def rvs(self, N):
        comp_count = np.squeeze(np.random.multinomial(N, self.weights, size = 1), axis = 0)
        X = np.empty((0,self.dim))
        for i, N in enumerate(comp_count):
            v = self.components[i].rvs(N)
            X = np.append(X, v, axis = 0)
        np.random.shuffle(X)
        return X
            

    def pdf(self, X):
        X = np.array(X)
        Likelihood = np.zeros(len(X))

        for i, x in enumerate(X):
            for c in range(self.K):
                Likelihood[i] += self.weights[c]*self.components[c].pdf(x)

        return Likelihood


    def log_pdf(self, x):
        pass


#Dirichlet Process
class DPMixtureModel:
    def __init__(self, DP, Exp_family_dist) -> None:
        self.DP = DP
        self.Exp_family_dist = Exp_family_dist

    #sample N random samples from the distribution
    def rvs(self, N):
        Z, etas = self.DP.rvs(N)
        v = np.argwhere(Z)
        indices = np.squeeze(np.argwhere(Z), axis = 1)
        dim = len(self.Exp_family_dist.rvs(1, random_state = RANDOM_SEED))
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
    def rvs(self, N):
        V = self.Beta.rvs(size = self.K, random_state = RANDOM_SEED)
        V[self.K - 2] = 1
        thetas = self.get_thetas(V)
        eta_star = self.G0.rvs(N)

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
    def __init__(self, prior, alpha, _lambda) -> None:
        self.gammas = None 
        self.taus = None 
        self.phis = None 
        self.prior = prior
        self.dist = prior.dist #The distribution that the prior is prior too
        self.dim = 0
        self.X = None
        self.K = None
        self.N = 0
        self.Beta = beta(1, alpha)

        #hyper-parameters
        self.alpha = alpha
        self.Lambda = _lambda 

    def fit(self, X, K, maxiter = 100, limit = 1e-5, N_runs = 10, plot = False):
        self.X = X
        self.K = K
        self.dim = X.shape[1]
        self.N = len(X)
        best_ELBO = -np.inf
        #Run algorithm N_run number of times
        #Before each run shuffle samples and re-initialize parameter values
        for run in range(N_runs):
            self._initialize()
            diff = np.inf
            old_ELBO = self.ELBO()
            #ELBO (Evidence lower bound) is minimized in the VI algorithm
            #The algorithm has converged when ELBO remains constant
            ELBO = old_ELBO
            i = 0
            while i<maxiter and diff > limit:
                if plot:
                    self.plot(plt)
                    plt.show()
                self._update()
                old_ELBO = ELBO
                ELBO = self.ELBO()
                #Save best parameter values
                if ELBO > best_ELBO:
                    best_ELBO = ELBO
                    best_phis = self.phis.copy()
                    best_taus = self.taus.copy()
                    best_gammas = self.gammas.copy()
                #print("iteration:",i,"ELBO:", ELBO)
                diff = np.abs(ELBO - old_ELBO)
                i += 1
            print("run nr:", run, "ELBO:",best_ELBO)

        #When the algorithm has converged for each run the best parameters are saved
        self.phis = best_phis
        self.taus = best_taus
        self.gammas = best_gammas


    def _update_phis(self):
        expected_log_v, expected_log_ones_minus_v = self.v_expectations()
        expected_log_v = np.append(expected_log_v, 0)
        expected_log_ones_minus_v = np.append(expected_log_ones_minus_v, 0)
        expected_As = self.prior.expected_A(self.taus)
        expected_etas =  self.prior.expected_eta(self.taus)

        for n in range(self.N):
            for k in range(self.K):
                E = expected_log_v[k] + self.X[n].T@expected_etas[k] -  expected_As[k] 
                for j in range(k):
                    E += expected_log_ones_minus_v[j]
                self.phis[n,k] = E

        m = np.max(self.phis, axis = 1)
        self.phis = np.exp(self.phis - m[:,None])
        sums = np.sum(self.phis, axis = 1)
        for n in range(self.N):
            for k in range(self.K):
                if self.phis[n, k] != 0.0:
                    self.phis[n, k] /= sums[k]

    def _update_gammas(self):
        for k in range(self.K - 1):
             self.gammas[k,0] = 1 
             for n in range(self.N):
                 self.gammas[k,0] += self.phis[n,k]

        for k in range(self.K-1):
            self.gammas[k,1] = self.alpha
            for n in range(self.N):
                for j in range(k+1, self.K):
                    self.gammas[k,1] += self.phis[n,j]

    def _update_taus(self):
        for k in range(self.K):
            self.taus[k,:-1] = self.Lambda[:-1]
            for n in range(self.N):
                self.taus[k,:-1] += self.phis[n,k]*self.X[n]

        for k in range(self.K):
            self.taus[k,-1:] =  self.Lambda[-1:]
            for n in range(self.N):
                self.taus[k,-1:] += self.phis[n,k]

    def _initialize(self):
        
        #Initialize the variational parameters.
        #Shuffle the samples
        
        np.random.shuffle(self.X)
        self.gammas = np.empty((self.K-1, 2))
        from numpy.random import dirichlet
        self.phis = dirichlet(np.ones(self.K), size=self.N)
        self.taus = np.outer(np.ones(self.K), self.Lambda)
        self.gammas[:,0] = 1.
        self.gammas[:,1] = self.alpha

        

    def _update(self):
        self._update_phis()
        self._update_gammas()
        self._update_taus()


    def get_thetas(self):
        expected_vs = self.gammas[:,0] / self.gammas.sum(axis=1)

        thetas = np.empty(self.K) 
        stick_left = 1.0 
        #Iteratively break of the expected 
        #value of each theta form the stick
        for k, q in enumerate(expected_vs):
            thetas[k] = stick_left * q
            stick_left *= (1. - q) 
        thetas[-1] = stick_left
        return thetas

    def ELBO(self):
        expected_log_v, expected_log_ones_minus_v = self.v_expectations()

        #LL for Vs
        alphas, betas = np.split(self.gammas, 2, axis = 1)
        alphas = np.squeeze(alphas)
        betas = np.squeeze(betas)
        value = (alphas - 1)*expected_log_v + (betas - 1)*expected_log_ones_minus_v - betaln(alphas, betas)
        value[-1] = 0
        ELBO = np.sum(value)
        value = (self.alpha - 1)*expected_log_ones_minus_v - betaln(1, self.alpha)
        value[-1] = 0
        ELBO -= np.sum(value)

        #LL eta
        d2 = len(self.Lambda)
        Lambda = np.repeat(self.Lambda[np.newaxis,:], repeats = self.K, axis = 0)
        ELBO += self.prior.log_expect(Lambda)
        ELBO -= self.prior.log_expect(self.taus)
        ELBO = np.squeeze(ELBO)

        #LL Z
        for n in range(self.N):
            phi = self.phis[n]
            phi_cum_sum = np.flip(np.cumsum(np.flip(phi[:,np.newaxis])))
            phi_cum_sum = np.roll(phi_cum_sum, -1)
            phi_cum_sum[-1] = 0
            ELBO += np.sum(phi_cum_sum*np.append(expected_log_ones_minus_v, 0) + phi*np.append(expected_log_v, 0))

        #LL X
        expected_eta = self.prior.expected_eta(self.taus)
        expected_A = self.prior.expected_A(self.taus)
        for n, x in enumerate(self.X):
            for k in range(self.K):
                if self.phis[n,k] != 0.0:
                    ELBO += self.phis[n, k]*(self.dist.log_h(x) + np.squeeze(x.T@expected_eta[k].T) - expected_A[k] - np.log(self.phis[n, k]))

        return ELBO


    def predictive_likelihood(self, X):
        #Return the likelihood for new values based on the model
        mixing_proportions = self.get_thetas()
        log_predictive = np.array([self.prior.predictive_log_prob(X, self.taus[k]) for k in range(self.K)]).T
        return np.sum(np.exp(log_predictive) * mixing_proportions, axis = 1)

    def v_expectations(self):
        #Calculates expected values of log(V_i)
        #and log(1-V_i) 
        v = digamma(np.sum(self.gammas, axis = 1))
        expected_log_v = digamma(self.gammas[:,0]) - v
        expected_log_ones_minus_v = digamma(self.gammas[:,1]) - v
        return (expected_log_v, expected_log_ones_minus_v)

    def plot(self, ax):
        #Plot samples and their corresponding predictive distribution
        x, y = np.split(self.X, 2, axis = 1)
        ax.scatter(x, y)

        max_x, max_y = np.amax(self.X, axis = 0)
        min_x, min_y = np.amin(self.X, axis = 0)
        X = np.linspace(min_x, max_x, num = 50)
        Y = np.linspace(min_y, max_y, num = 50)
        Z = np.empty((len(X), len(Y)))


        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                Z[j, i] = self.predictive_likelihood(np.array([x, y]))

        ax.contour(X, Y, Z)
        return ax


    ######################################################################TEST#######################################################


def test_VI():
    N = 100
    dim = 2
    cov = np.diag([0.2,0.2])

    fig = plt.figure(constrained_layout=True)
    #fig.suptitle("Runs of VI Algorithm", fontsize = 30)
    axs = fig.subplots(1, 3)

    for ax in axs:
        samples = generate_simulated_data(BASE_MEAN=np.array([0, 0]),
                                                     BASE_COVARIANCE=np.diag([10, 10]),
                                                      COVARIANCE_MATRIX=cov,
                                                      alpha=5,
                                                      data_points=2*N,
                                                      truncation_level=20,
                                                      random_state=RANDOM_SEED)
        samples = np.squeeze(samples, axis = 1)
        samples, held_out = np.split(samples, 2, axis = 0)
        sample_mean = np.mean(samples, axis = 0)
        sample_cov = np.cov(samples.T)

        #Set Hyperparams
        alpha = 1
        Lambda = np.full(dim + 1, 0.01)
        #lambda_1, lambda_2 = MvNormalPrior(cov).eta((np.zeros(dim), np.eye(dim)/0.1))

        lambda_1, lambda_2 = MvNormalPrior(cov).eta((sample_mean, np.eye(dim)/0.1))
        Lambda[:-1] = lambda_1
        Lambda[-1] = lambda_2

        prior = MvNormalPrior(cov, Lambda)

        SBP_VI = VI_SBmodel(prior, alpha, Lambda)
        SBP_VI.fit(samples, 20, 500, 1e-10, 10, False)

        log_prob = 0.0

        for p in held_out:
            prob = SBP_VI.predictive_likelihood(p)
            log_prob += np.log(prob)
            
        print("avg held out LL:", log_prob/N)
        SBP_VI.plot(ax)

    plt.show()


def test_dimensions():
    all_dimensions = [5, 10, 20, 30, 40, 50]
    TRUNCATION_LEVEL = 20
    alpha = 1
    N = 100
    fig = plt.figure(constrained_layout=True)
    #fig.suptitle("Avg Log Likelihood", fontsize = 20)
    plt.xlabel('dimension', fontsize = 15)
    plt.ylabel('LL', fontsize = 15)
    LLs = []
    for dimension in all_dimensions:
        covariance_matrix = np.zeros((dimension, dimension))
        #Create autocorrelation matrix with rho = 0.9
        for i in range(dimension):
            for j in range(dimension):
                covariance_matrix[i, j] = (0.9)**(abs(i - j))

        
        COVARIANCE_MATRIX = covariance_matrix

        data = generate_simulated_data(BASE_MEAN=np.zeros(dimension),
                                                 BASE_COVARIANCE=np.eye(dimension),
                                                  COVARIANCE_MATRIX=COVARIANCE_MATRIX,
                                                  alpha=alpha,
                                                  data_points=2*N,
                                                  truncation_level=TRUNCATION_LEVEL,
                                                  random_state=RANDOM_SEED)

        data = np.squeeze(data, axis = 1)
        samples, held_out = np.split(data, 2, axis = 0)

        sample_mean = np.mean(samples, axis = 0)

        training_alpha = 1
        Lambda = np.full(dimension + 1, 0.01)
        #lambda_1, lambda_2 = MvNormalPrior(covariance_matrix).eta((np.zeros(dimension), np.eye(dimension)/0.1))
        lambda_1, lambda_2 = MvNormalPrior(covariance_matrix).eta((sample_mean, np.eye(dimension)/0.1))
        Lambda[:-1] = lambda_1
        Lambda[-1] = lambda_2

        prior = MvNormalPrior(COVARIANCE_MATRIX, Lambda)
        SBP_VI = VI_SBmodel(prior, training_alpha, Lambda)
        SBP_VI.fit(samples, 20, 500, 1e-10, 10)

        log_prob = 0.0

        for p in held_out:
            prob = SBP_VI.predictive_likelihood(p)
            log_prob += np.log(prob)

        LLs.append(log_prob)
        print("Dimension:", dimension, "held out LL:",log_prob)

    all_dimensions = np.array(all_dimensions)
    LLs = np.array(LLs)

    plt.scatter(all_dimensions, LLs)
    plt.show()


def test_robot_data():
    #Very slow
    #Found this dataset containing 4499 datapoints

    path = "DD2434FDD3434MachineLearningAdvancedCourse/1.5hANDIN/data/"
    X_train = []
    with open(path + "puma8NH.data") as Datafile:
        for j in range(4499):
            line = Datafile.readline().strip(' \n')
            values = line.split(' ')
            values = [float(values[i]) for i in range(len(values)-1)]
            assert len(values) == 8
            X_train.append(values)

    X_train = np.array(X_train)
    X_train, X_eval = np.split(X_train, [4499-250-1],axis = 0)

    sample_mean = np.mean(X_train, axis = 0)
  

    #Hyper Params
    dim = 8
    cov = np.cov(X_train.T)
    #alpha = 2
    #Lambda = np.zeros(dim + 1)
    #Lambda[:-1] = sample_mean
    #Lambda[-1] = 1 

    alpha = 1
    Lambda = np.full(dim + 1, 0.01)
    lambda_1, lambda_2 = MvNormalPrior(cov).eta((sample_mean, np.eye(dim)/0.1))
    Lambda[:-1] = lambda_1
    Lambda[-1] = lambda_2

    prior = MvNormalPrior(cov, Lambda)
    SBP_VI = VI_SBmodel(prior, alpha, Lambda)
    SBP_VI.fit(X_train, 20, 500, 1e-10, 1, False)

    log_prob = 0.0

    for p in X_eval:
        prob = SBP_VI.predictive_likelihood(p)
        #print(prob)
        log_prob += np.log(prob)
            

    print("held out LL:", log_prob/len(X_eval))



def gaussian_test():
    N = 4000
    
    weights = [0.125, 0.125, 0.125, 0.125,0.125, 0.125, 0.125, 0.125]
    mus = [
        [-5, 0],
        [5, 0],
        [-5, 5],
        [5, 5],
        [-5, 10],
        [5, 10],
        [-5, 15],
        [5, 15],
        ]

    cov = np.diag([1, 1])
    covs = [
        cov,
        cov,
        cov,
        cov,
        cov,
        cov,
        cov,
        cov
        ]
    
    GMM2 = GaussianMixtureModel(mus, covs, weights)
    X = GMM2.rvs(N*2)
    #x, y = np.split(X, 2, axis = 1)
    #plt.scatter(x, y)
    #plt.show()
    #likelihood = np.sum(GMM2.pdf(X))

    samples, held_out = np.split(X, 2, axis = 0)
    sample_mean = np.mean(samples, axis = 0)
  
    dim = 2
    alpha = 1
    Lambda = np.full(dim + 1, 0.01)
    lambda_1, lambda_2 = MvNormalPrior(cov).eta((np.zeros(dim), np.eye(dim)/0.1))
    Lambda[:-1] = lambda_1
    Lambda[-1] = lambda_2

    prior = MvNormalPrior(cov, Lambda)

    SBP_VI = VI_SBmodel(prior, alpha, Lambda)
    SBP_VI.fit(samples, 20, 500, 1e-10, 1, False)

    log_prob = 0.0

    for p in held_out:
        prob = SBP_VI.predictive_likelihood(p)
        log_prob += np.log(prob)
            
    print("avg held out LL:", log_prob/N)
    print("actuall avg LL", np.log(np.mean(GMM2.pdf(held_out))))
    SBP_VI.plot(plt)
    plt.show()

if __name__ == "__main__":
    test_VI()
    test_dimensions()
    gaussian_test()
    test_robot_data()
