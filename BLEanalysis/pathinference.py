from jax import grad
import optax
from jax.scipy.stats import norm
from jax.random import multivariate_normal as mvn
from jax.random import normal
from jax import random
import jax
import jax.numpy as np
import pyproj

from BLEanalysis import kl_mvn 
from BLEanalysis.angleinference import AnglesUsePatternMeans
from BLEanalysis import confidence_ellipse
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

#key = random.key(0)

class Path:
    def __init__(self, observation_times, observations, kernel, inducing_points, ndims = 2, margin = 0.1, jitter = 0.1):
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
        self.key = random.PRNGKey(0)
        self.observations = observations # Angle observations made
        self.observation_times = observation_times # Times observations were made at
        self.ndims = ndims
        #self.noise_scale = noise_scale # Noise scale of likelihood --> likelihood specific.
        self.kernel = kernel # Kernel function
        self.jitter = jitter # Jitter applied to covariance matrix during training

        self.startime = 0
        self.endtime = 999

        if type(inducing_points)==int:
            self.Z = self.selectPoints(inducing_points, margin) # Select inducing points
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
        self.Kzz = self.kernel.K(self.Z, self.Z) + (np.eye(self.Z.shape[0], dtype=np.float32) * self.jitter)
        self.Kxx = self.kernel.K(X, X) + (np.eye(X.shape[0], dtype=np.float32) * self.jitter)
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
    

    def selectPoints(self, number, margin):
        """Returns an array of one-dimensional (inducing) point locations, placed evenly over the domain of times
        in self.observation_times, with a margin added.

        TODO: Could make the margin depend on kernel lengthscale
        """
        
        max_time = np.max(np.array(self.observation_times)) + margin
        min_time = np.min(np.array(self.observation_times)) - margin       
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
        self.key, subkey = random.split(self.key)
        posterior_mean, posterior_cov, surrogate_cov = self.get_posterior_distribution_parameters(surrogate_mean,surrogate_cov_tril)
        samples = mvn(subkey,posterior_mean,posterior_cov,(self.number_samples,)) #number_samples x number_obserations

        #compute ELBO: -(ll - KLdivergence)
        lls = self.compute_log_likelihood(samples) #log likelhioods of the samples
        elbo = -(np.mean(np.sum(lls,axis=1)) - kl_mvn(self.prior_mean, self.prior_covariance, surrogate_mean, surrogate_cov))
        return elbo

    def run(self,iterations=500,number_samples=100,just_optimise_means=False,learning_rate=1,ndims=2):
        self.number_samples = number_samples
        self.iterations=iterations
        self.precompute_matrices(self.X)
        self.key, subkey = random.split(self.key)
        optimizer = optax.adam(learning_rate)
        surrogate_mean = normal(subkey,(self.nind, ))*20
        surrogate_cov_tril = np.eye(self.nind)
        opt_state = optimizer.init((surrogate_mean,surrogate_cov_tril))

        print("Optimising mean...")
        for it in range(iterations):#int(iterations*0.5)):
            grads = grad(self.calc_elbo,[0,1])(surrogate_mean,surrogate_cov_tril)
            updates, opt_state = optimizer.update(grads, opt_state)
            surrogate_mean,_ = optax.apply_updates((surrogate_mean,surrogate_cov_tril), updates)
            if it%50==0: 
                print("iteration %4d: %9.1f" % (it,self.calc_elbo(surrogate_mean,surrogate_cov_tril)))

        if not just_optimise_means:
            optimizer = optax.adam(learning_rate)
            opt_state = optimizer.init((surrogate_mean,surrogate_cov_tril))

            print("Optimising mean and covariance...")
            for it in range(iterations):# int(iterations*0.5)):
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
        
    def plot(self, startTime = 0, endTime = 7, coordNormalisationFactor = [], transmitters = [], n_test=50, n_std=4, ellipseInterval = 1, 
             timeMultiplier = 10000, coordScaleFactor = [], GPSFile = "", syntheticPath = []):
        #plt.axis("equal")
        ax = plt.gca()
        plt.xlabel("Easting")
        plt.ylabel("Northing")

        if len(syntheticPath) > 0:
            plt.plot(syntheticPath[:,0], syntheticPath[:,1],'x-')
            n_test = len(syntheticPath)
            
        if not(GPSFile == ""):
            P = pyproj.Proj(proj='utm', zone=30, ellps='WGS84', preserve_units=True)
            
            # plot gps use number of GPS readings as inference points
            with open(GPSFile, 'r') as f:
                filedata = f.read()
            GPSData = filedata[299:].split('\n')
            n_test = len(GPSData)
            
            lat = []
            lon = []
            for coord in GPSData[:-1]:
                coord = coord.split(',')
                lat.append(float(coord[3]))
                lon.append(float(coord[4]))
            GPSxx, GPSyy = P(lon,lat)
            plt.plot(GPSxx, GPSyy, label="GNSS Ground Truth")
            
        times = np.linspace(startTime,endTime,n_test)
        posterior_mean, posterior_cov = self.get_predictions(times)
        plt.plot(posterior_mean[:n_test] + coordScaleFactor[1],posterior_mean[n_test:] + coordScaleFactor[0],'-', label="Inferred Path")
        for i in np.linspace(0,n_test-1,ellipseInterval).astype(int):
            el = confidence_ellipse([posterior_mean[i::n_test][0] + coordScaleFactor[1], posterior_mean[i::n_test][1] + coordScaleFactor[0]], 
                                    posterior_cov[i::n_test,i::n_test],ax,n_std=n_std)
            ax.add_patch(el)
            # Print start and end time
            if i == 0:
                plt.text(posterior_mean[i::n_test][0] + coordScaleFactor[1], posterior_mean[i::n_test][1] + coordScaleFactor[0],
                         "START")
            #if i == (n_test - 1):
                #plt.text(posterior_mean[i::n_test][0] + coordScaleFactor[1], posterior_mean[i::n_test][1] + coordScaleFactor[0],
                #         "t = " + str(int((times[i] * timeMultiplier)/1000) - int(times[0] * timeMultiplier/1000)))
        for transmitter in transmitters:
            plt.scatter(transmitters[transmitter][0][1],transmitters[transmitter][0][0])
            plt.text(transmitters[transmitter][0][1],transmitters[transmitter][0][0], transmitter)

        scalebar = AnchoredSizeBar(ax.transData,
                           50,          # length in data units (100 meters here)
                           '50 m',      # label
                           'upper right',# location
                           pad=0.4,
                           color='black',
                           frameon=False,
                           size_vertical=2)
        ax.add_artist(scalebar)
        ax.ticklabel_format(useOffset=False, style='plain')

        MAE = 0
        CDF = []
        if not(GPSFile == ""):
            # calculate the MAE point for point between the GPS path and the inferred path
            for point in range(n_test-1):
                #plt.plot((GPSxx[point], posterior_mean[:n_test][point] + coordScaleFactor[1]),
                #         (GPSyy[point], posterior_mean[n_test:][point] + coordScaleFactor[0]))
                CDF.append(np.round((np.abs(np.sqrt(np.square(GPSxx[point] - (posterior_mean[:n_test][point] + coordScaleFactor[1])) 
                                      + np.square(GPSyy[point] - (posterior_mean[n_test:][point] + coordScaleFactor[0]))))),2))
                MAE += np.round((np.abs(np.sqrt(np.square(GPSxx[point] - (posterior_mean[:n_test][point] + coordScaleFactor[1])) 
                                      + np.square(GPSyy[point] - (posterior_mean[n_test:][point] + coordScaleFactor[0]))))),2)
            plt.text(.02, .98, 'MAE: ' + str(round(MAE/n_test, 2)) + 'm', ha='left', va='top', transform=ax.transAxes)

        if len(syntheticPath) > 0:
            for point in range(n_test-1):
                MAE += np.round((np.abs(np.sqrt(np.square(syntheticPath[point][0] - (posterior_mean[:n_test][point] + coordScaleFactor[1]))
                                       + np.square(syntheticPath[point][1] - (posterior_mean[n_test:][point] + coordScaleFactor[0]))))),2)
            plt.text(.02, .98, 'MAE: ' + str(round(MAE/n_test, 2)) + 'm', ha='left', va='top', transform=ax.transAxes)

        plt.legend(loc="lower left")
        
        return MAE/n_test, CDF

class Path_VectorsToBee(Path):
    def __init__(self, observation_times, observations, kernel, inducing_points, noise_scale=0.1, ndims = 2, margin = 0.1, jitter = 0.1):
        """
        noise_scale = number of distance-units (e.g. metres) the standard deviation of the noise is (default 0.1m)
        """
        self.noise_scale = noise_scale
        super().__init__(observation_times, observations, kernel, inducing_points, ndims, margin, jitter)
        
    def prepare_likelihood(self):
        self.y_vects = np.c_[self.y[:,2:],np.zeros(self.n)] #vectors to bee extracted from 'y'
        self.y_pos = self.y[:,:2].T.reshape(self.y.shape[0]*self.ndims)

    def compute_log_likelihood(self,samples):
        """
        Samples are possible locations of the bee. 
        Return an array of the loglikelihood of each.
        Assumes observations are angles to the bees.
        """
        samples_with_z = np.c_[samples- self.y_pos,np.zeros([self.number_samples,self.n])] #adds an extra axis
        s = samples_with_z.reshape(self.number_samples,3,self.n)
        s = np.swapaxes(s,1,2)        
        cross_vects = np.cross(self.y_vects,s)
        distances = np.linalg.norm(cross_vects,axis=2) #don't need to divide by np.linalg.norm(y[:,2:],axis=1), as this is set to one
        return norm.logpdf(distances,0,self.noise_scale)
   
class Path_ProbabilityDensitiesToBee(Path):
    def __init__(self, observation_times, burst_observations, kernel, noise_scale, inducing_points, ndims = 2, margin = 0.1, jitter = 0.1):
        """Pass each burst as an observation"""
        self.angles = AnglesUsePatternMeans(noisevar=noise_scale**2)
        super().__init__(observation_times, burst_observations, kernel, inducing_points, ndims, margin, jitter)

    def prepare_likelihood(self):
        all_lpdfs = []
        all_ypos = []
        all_argmax_angles = []
    
        for obs in self.y:
            lpdf, _, _, _ = self.angles.infer(obs['rssis'], obs['angles'])
            all_lpdfs.append(lpdf)
            all_ypos.append([
                obs['transmitter_position'][0][1],
                obs['transmitter_position'][0][0]
            ])
            all_argmax_angles.append(np.deg2rad(np.argmax(lpdf)))
    
        self.y_vects = jax.device_put(np.array(all_lpdfs))
        self.y_angles = jax.device_put(np.array(all_argmax_angles))
        self.y_pos = jax.device_put(
            np.array(all_ypos).T.reshape(len(self.y) * self.ndims)
        )
    
    #def prepare_likelihood(self):
    #    all_pdfs = []
    #    all_lpdfs = []
    #    all_ypos = []
    #    all_argmax_angles = []
    #    for i,obs in enumerate(self.y):
    #        lpdf,_,_,_ = self.angles.infer(obs['rssis'],obs['angles'])
    #        all_lpdfs.append(lpdf)
    #        all_ypos.append([obs['transmitter_position'][0][1],obs['transmitter_position'][0][0]])
    #        all_argmax_angles.append(np.deg2rad(np.argmax(lpdf)))
    #    self.y_vects = np.array(all_lpdfs)
    #    self.y_angles = np.array(all_argmax_angles)

    #    self.y_pos = np.array(all_ypos).T.reshape(len(self.y)*self.ndims)
        
    def compute_log_likelihood(self,samples):
        #We need to get what the probability of each sample is
        #1) Subtract the location of each transmitter
        s = (samples - self.y_pos)
        
        #2) Compute the angle from each transmitter to the sample
        
        sample_angles = (np.atan2(s[:,:self.n],s[:,self.n:]))%(np.pi*2)
        #    sample_angles = (np.atan2(s[:,:self.n], s[:,self.n:])) % (2*np.pi)


        
        #Detour: If we want to use an alternative approach
        #err = (self.y_angles-sample_angles)%(2*np.pi)
        #circularises the subtraction
        #err = err[:,:,None]
        #err = np.c_[np.abs(err),np.abs(err-np.pi*2)]
        #err = np.min(err,2)
        #print(np.mean(err))
        #distances = np.sqrt(s[:,self.n:]**2 + s[:,:self.n]**2)
        #err = err/distances
        
        #return norm.logpdf(err,0,0.05)
        
        #3) Find the floating point "index" into the probability density data (stored as a list of densities) -- currently there are 360 of them THIS COULD CHANGE!
        bin_float_index = 360 * sample_angles.T/(2*np.pi)
        
        #4) Find the actual index:
        bin = np.trunc(bin_float_index).astype(int) #we happen to use 360 bins in the inference, but that isn't required!
        assert np.all(bin>=0)
        assert np.all(bin<360)
        
        #5) and the proportional amount (so we can linearly interpolate)
        bin_ratio = bin_float_index%1
        
        #6) compute the linear interpolated prob. densities:
        logps = np.take_along_axis(self.y_vects,bin,axis=1)*(1-bin_ratio) + np.take_along_axis(self.y_vects,(1+bin)%360,axis=1)*bin_ratio
        #print(np.min(np.log(ps)))
        return logps

