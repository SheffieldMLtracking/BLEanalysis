from jax import grad
import optax
from jax.scipy.stats import norm
from jax.random import multivariate_normal as mvn
from jax.random import normal
from jax import random
key = random.key(0)
import jax.numpy as np
from BLEanalysis import kl_mvn 
from BLEanalysis import confidence_ellipse
import matplotlib.pyplot as plt
        
class Path:
    def __init__(self, observation_times, observations, kernel, noise_scale, inducing_points, ndims = 2):
        """
        Performs variational inference using the 'single angle only' observations to work out possible paths of the tag.
        
        This uses observations that consist of single angles, rather than probability distributions of angles.
        
        Parameters:
         observation_times : an array of the times the observations were made.
         observations : array of the angles they were (radians)
         kernel : an instantiation of a kernel class
         noise_scale : the standard deviation (TODO or variance?? TBC) of the Gasussian noise model
         inducing_points : either the number of inducing points, or an array of their values.
                  If the former, the class places inducing points, with some before and after the data.
         ndims : the numbers of spatial dimensions (default = 2).
        """
        self.observations = observations # Angle observations made
        self.observation_times = observation_times # Times observations were made at
        self.ndims = ndims
        self.noise_scale = noise_scale # Noise scale of likelihood
        self.kernel = kernel # Kernel function
        self.jitter = 0.01 # Jitter applied to covariance matrix during training

        if type(inducing_points)==int:
            self.Z = self.selectPoints(inducing_points) # Select inducing points
        else:
            self.Z = inducing_points
        self.loss_history = []


        self.nind = self.Z.shape[0]
        self.diag_jitter = np.eye(self.nind) * self.jitter
        #Precompute various matrices
        self.X = self.buildX(observation_times)
        self.y = self.observations
        self.n = len(self.y)
        
        self.prior_mean = np.zeros(self.nind)
        self.prior_covariance = self.kernel.K(self.Z, self.Z)
        self.prepare_likelihood()

    def buildX(self,observation_times):
        """Helper function to generate the X matrix:
         time1, 0
         time2, 0
         time3, 0
          :
         timeN, 0
         time1, 1
         time2, 1
        """
        return np.c_[np.tile(observation_times, self.ndims), np.repeat(np.arange(self.ndims),len(observation_times))]

    def precompute_matrices(self,X):
        """
        Computes matrices used later during inference. Note that this means the hyperparameters all remain fixed during inference.
        # GP(Kzx Kzz^-1 y, Kzz - Kzx Kxx^-1 Kxz)
        Also - once modified for testing, can't use for training!
        """
        self.Kzz = self.kernel.K(self.Z, self.Z) + np.eye(self.nind) * self.jitter
        self.Kxx = self.kernel.K(X, X) + np.eye(X.shape[0]) * self.jitter
        self.Kxz = self.kernel.K(X, self.Z)
        self.Kzx = self.Kxz.T
        self.KzzinvKzx = np.linalg.solve(self.Kzz, self.Kzx)
        self.KxzKzzinv = self.KzzinvKzx.T
        self.KxzKzzinvKzx = self.Kxz @ self.KzzinvKzx       

    #Likelihood specific methods
    def prepare_likelihood(self):
        """
        This optional method allows the child class to precompute expensive terms etc for likelihood calculation
        """
        #raise NotImplementedError
        pass
    

    def selectPoints(self, number, margin = 3):
        """Returns an array of one-dimensional (inducing) point locations, placed evenly over the domain of times
        in self.observation_times, with a margin added.

        TODO: Could make the margin depend on kernel lengthscale
        """
        
        max_time = np.max(self.observation_times) + margin
        min_time = np.min(self.observation_times) - margin       
        result = []        
        for vector_observed in range(int(self.ndims)):        
            result.extend(np.c_[np.linspace(min_time, max_time, number), np.full(number, vector_observed)])
        return np.array(result)

    def get_posterior_distribution_parameters(self,surrogate_mean,surrogate_cov_tril):
        tril = np.tril(surrogate_cov_tril)
        surrogate_cov = tril@tril.T + self.diag_jitter
        posterior_mean = (self.KxzKzzinv @ surrogate_mean)
        posterior_cov = self.Kxx - self.KxzKzzinvKzx + self.KxzKzzinv @ surrogate_cov @ self.KzzinvKzx        
        return posterior_mean, posterior_cov, surrogate_cov

    def compute_log_likelihood(self,samples):
        """
        Samples are possible locations of the bee. Return an array of the loglikelihood of each
        """
        raise NotImplementedError
        
        
    def calc_elbo(self,surrogate_mean,surrogate_cov_tril):
        posterior_mean, posterior_cov, surrogate_cov = self.get_posterior_distribution_parameters(surrogate_mean,surrogate_cov_tril)
        samples = mvn(key,posterior_mean,posterior_cov,self.number_samples) - self.y_pos #number_samples x number_obserations

        #compute ELBO: -(ll - KLdivergence)
        lls = self.compute_log_likelihood(samples) #log likelhioods of the samples
        elbo = -(np.mean(np.sum(lls,axis=1))- kl_mvn(self.prior_mean, self.prior_covariance, surrogate_mean, surrogate_cov))
        return elbo

    def run(self,iterations=500,number_samples=100,learning_rate=1,ndims=2):
        self.number_samples = number_samples
        self.iterations=iterations
        self.precompute_matrices(self.X)

        optimizer = optax.adam(learning_rate)
        surrogate_mean = normal(key,self.nind)*20
        surrogate_cov_tril = np.eye(self.nind)
        opt_state = optimizer.init((surrogate_mean,surrogate_cov_tril))

        print("Optimising mean...")
        for it in range(int(iterations*0.5)):
            grads = grad(self.calc_elbo,[0,1])(surrogate_mean,surrogate_cov_tril)
            updates, opt_state = optimizer.update(grads, opt_state)
            surrogate_mean,_ = optax.apply_updates((surrogate_mean,surrogate_cov_tril), updates)
            if it%50==0: 
                print("iteration %4d: %9.1f" % (it,self.calc_elbo(surrogate_mean,surrogate_cov_tril)))
        
        optimizer = optax.adam(0.3)
        opt_state = optimizer.init((surrogate_mean,surrogate_cov_tril))

        print("Optimising mean and covariance...")
        for it in range(int(iterations*0.5)):
            grads = grad(self.calc_elbo,[0,1])(surrogate_mean,surrogate_cov_tril)
            updates, opt_state = optimizer.update(grads, opt_state)
            surrogate_mean,surrogate_cov_tril = optax.apply_updates((surrogate_mean,surrogate_cov_tril), updates)
            if it%50==0: 
                print("iteration %4d: %9.1f" % (it,self.calc_elbo(surrogate_mean,surrogate_cov_tril)))
                #print(surrogate_mean)

        self.surrogate_mean = surrogate_mean
        self.surrogate_cov_tril = surrogate_cov_tril

    def get_predictions(self,test_times):
        testX = self.buildX(test_times)
        self.precompute_matrices(testX)
        posterior_mean, posterior_cov, _ = self.get_posterior_distribution_parameters(self.surrogate_mean,self.surrogate_cov_tril)
        return posterior_mean, posterior_cov
        
    def plot(self):
        n_test = 50
        posterior_mean, posterior_cov = self.get_predictions(np.linspace(-1,6,n_test))
        plt.plot(posterior_mean[:n_test],posterior_mean[n_test:],'-')
        ax = plt.gca()
        for i in np.linspace(0,n_test-1,7).astype(int):
            el = confidence_ellipse(posterior_mean[i::n_test],posterior_cov[i::n_test,i::n_test],ax)
            ax.add_patch(el)
        posterior_mean[0::n_test],posterior_cov[0::n_test,0::n_test]

class Path_VectorsToBee(Path):
    def __init__(self, observation_times, observations, kernel, noise_scale, inducing_points, ndims = 2):
        super().__init__(observation_times, observations, kernel, noise_scale, inducing_points, ndims)

    def prepare_likelihood(self):
        self.y_vects = np.c_[self.y[:,2:],np.zeros(self.n)] #vectors to bee extracted from 'y'
        self.y_pos = self.y[:,:2].T.reshape(self.y.shape[0]*self.ndims)

    def compute_log_likelihood(self,samples):
        """
        Samples are possible locations of the bee. 
        Return an array of the loglikelihood of each.
        Assumes observations are angles to the bees.
        """
        samples_with_z = np.c_[samples,np.zeros([self.number_samples,self.n])] #adds an extra axis
        s = samples_with_z.reshape(self.number_samples,3,self.n)
        s = np.swapaxes(s,1,2)        
        cross_vects = np.cross(self.y_vects,s)
        distances = np.linalg.norm(cross_vects,axis=2) #don't need to divide by np.linalg.norm(y[:,2:],axis=1), as this is set to one
        return norm.logpdf(distances,0,0.1)
        
    
