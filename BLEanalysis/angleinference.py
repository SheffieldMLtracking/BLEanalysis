import hashlib
import numpy as np
import pickle
from BLEanalysis.signals import Signals
from scipy.stats import norm

def normalise_logs_to_ps(logp):
    p = np.exp(logp - np.max(logp))
    p/= np.sum(p)
    return p


class Angles:
    def __init__(self):
        raise NotImplementedError
        
    def infer(self,time_intervals,obs):
        raise NotImplementedError
        
class AnglesUsePatternMeans(Angles):
    def __init__(self,sigs=None,noisevar = 10**2):
        """...
        
        Parameters:
         sigs : a Signal object containing the raw training data from one of the transmitters that
                is used to build the estimate of the pattern. sigs.data will have a table containing:
                  [RSSI, ID(ord of a character id), Angle(radians), Time(milliseconds since transmitter turned on)]
         noisevar : the noise variance in the observations at test time (might be in dB^2?)
         """
        if sigs is None:
            sigs = Signals("noamploc2long.log",'d',angleOffset = 38) #TODO Figure out how to make this always available
        self.avgRSSIs,_ = sigs.averageRSSIsAtAngle(detrend=True,smooth=True)
        self.noisevar = noisevar
        
    def infer(self,obs,obs_angles):
        """
        Returns the [unnormalised] log probabilities of a list of angles, given the observed signal strengths, and
        the angles recorded.
        
        Currently returns: logp,errs,avgAtAngles,keptObs
        """ 
        #obs and obs_angle can contain NaNs for missing observations.
        keep = ~np.isnan(obs_angles)
        
        #the data in the avg is organised into one degree per item, so this gives the indices:
        obs_angle_indices = np.rad2deg(obs_angles[keep]).astype(int)
        
        #we build an array N_obs x 360, of the indicies of the avgRSSIs array we should look in, for offsets in 1 degree steps
        obs_angle_indices = (obs_angle_indices[:,None]+np.arange(360))%360
        
        #compute the difference for 1 degree step, for the N_obserations we have.
        avgAtAngles = self.avgRSSIs[obs_angle_indices,1]
        keptObs = obs[keep]
        avgAtAngles = avgAtAngles - np.mean(avgAtAngles,0)
        keptObs-=np.mean(keptObs)
        errs = avgAtAngles.T - keptObs # self.avgRSSIs[obs_angle_indices,1].T-obs[keep]
        
        #compute the SSE for each of these 1 degree steps, and divide by the noise-variance. This assumes a Gaussian noise model
        #for our noise.
        logp = -np.sum(errs**2,1)/self.noisevar #sse
        ##logp = -np.sum(np.abs(errs),1)/np.sqrt(self.noisevar) #exponential-dist. noise
        
        #if about 20% are missing, we should include this: 
        #
        #    log(p(y_missing_items|theta) * p(y_not_missing_items|theta))
        #    Nmissing * log(p(missing|theta)) + logp (computed above)  
        logp += np.sum(~keep)*np.log(1/5) 
        #To compute the probability:
        #p = np.exp(logp - np.max(logp))
        #p/= np.sum(p)
        #plt.plot(p)
        return logp,errs,avgAtAngles,keptObs

