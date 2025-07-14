import hashlib
import numpy as np
import pickle
from BLEanalysis.signals import Signals
from scipy.stats import norm

def getSample(data, time, time_intervals,raw=False,exclude_missing=10):        
    """This method is for sampling a whole bunch of times at once, as part of generating the training data.
    Specifically, given a numpy array 'data', containing columns [RSSI, ID, Angle(radians), Time(milliseconds)]
    and the time we want to sample near (time) and the intervals (time_intervals) returns a single instance of this
    #TODO Return None (or raise exception) if missing
    #TODO Allow a list of times to be past, to make this faster    
    
    Parameters:
     - time: the time that we are sampling.
     - time_intervals: separation between samples
     
     - raw: If set to true, returns the actual (rather than relative RSSIs) (default False).
     - exclude_missing: if a number, sets to nan if no samples are available within that time
                   (default=10ms, which is equivalent to 1.8 degrees). Set to None to disable
    Note: If the first element of the returned RSSIs is set to NaN, and raw is False, then all the elements will be NaN
     ( as the first one is subtracted from the rest ).
    """
    times = np.array(time_intervals) + time
    # find index of the packets in our data set, closest to these times.
    
    index = np.argmin(np.abs(data[:, -1:] - times[None,:]), 0)       
    rssis = data[index,0]
    if exclude_missing is not None:
        rssis[(np.abs(data[:, -1:] - times[None,:]))[index,range(len(index))]>=exclude_missing]=np.NaN
        #print((np.abs(data[:, -1:] - times[None,:]))[index,range(len(index))]>=exclude_missing)
    if not raw:       
        rssis-=rssis[0] #NOTE: I've switched to making the first time the angle index time as we could have an unknown number of time_intervals.        
    return rssis, data[index[0],2]

def OLDgetSample(data, time, time_intervals):        
    """This method is for sampling a whole bunch of times at once, as part of generating the training data.
    Specifically, given a numpy array 'data', containing columns [RSSI, ID, Angle(radians), Time(milliseconds)]
    and the time we want to sample near (time) and the intervals (time_intervals) returns a single instance of this
    #TODO Return None (or raise exception) if missing
    #TODO Allow a list of times to be past, to make this faster    
    
    Parameters:
     - time: the time that we are sampling.
     - time_intervals: separation between samples
    """
    times = np.array(time_intervals) + time
    # find index of the packets in our data set, closest to these times.
          
    index = np.argmin(np.abs(data[:, -1:] - times[None,:]), 0)
    rssis = data[index,0]
    rssis-=rssis[0] #NOTE: I've switched to making the first time the angle index time as we could have an unknown number of time_intervals.        
    return rssis, data[index[0],2]

class Angles:
    def __init__(self):
        raise NotImplementedError
        
    def infer(self,time_intervals,obs):
        raise NotImplementedError


class AnglesUsePatternMeansIntegrateOutAttenuation(Angles):
    def __init__(self,sigs,angs = np.linspace(0,2*np.pi,180)):
        """...
        
        Parameters:
         sigs : a Signal object containing tthe raw training data from one of the transmitters that
                is used to build the estimate of the pattern. sigs.data will have a table containing:
                  [RSSI, ID(ord of a character id), Angle(radians), Time(milliseconds since transmitter turned on)]
         angs : search space (e.g. np.linspace(0,2*np.pi,180)).
         """
        self.avgRSSIs,_ = sigs.averageRSSIsAtAngle(detrend=True,smooth=True)
        self.angs = angs
        #I want to get the average angle change over 100ms, for this data
        #intervals = np.array([-200,-100,0,100,200])        
        #s = []
        #for i in range(1000):
        #    obs, obs_angles = getSample(sigs.data,747685+np.random.rand()*732482,intervals,raw=True,exclude_missing=5)
        #    if np.any(np.isnan(obs)): continue #skip if ALL have no data
        #    s.append(obs_angles)
        #np.median(np.diff(np.array(s))%(np.pi*2))
        
    def infer(self,time_intervals,obs):
#            #need to infer the observed angles of any missing packets (from the angles sent in the other packets)
#        diff = np.median(((obs_angles-(np.pi*intervals/1000))%(2*np.pi))[~np.isnan(obs)])
#        obs_angles[np.isnan(obs)] = (np.pi*intervals/1000+diff)[np.isnan(obs)]        
    
        log_p_y_given_Theta = []
        for i,theta in enumerate(self.angs):
            print("%d/%d" % (i,len(self.angs)),end="\r")
            p = 0
            #TODO Switch to using the actual ANGLES (or at least times)
            #get index of avgRSSI for each observation
            idxs = np.argmin(np.abs(self.avgRSSIs[:,0:1]-(((np.pi*time_intervals/1000)+theta)%(2*np.pi))),0)

            #the 'unattenuated' prediction (i.e. the training set - we pretend is unattenuated!)
            unatt_pred = self.avgRSSIs[idxs,1]
            for a in np.linspace(0,50,100):        
                pred = unatt_pred-a
                p+=np.nanprod(norm(pred,3).pdf(obs))
            log_p_y_given_Theta.append(np.log(p))

        p_y_given_Theta = np.exp(log_p_y_given_Theta-np.max(log_p_y_given_Theta))
        p_y_given_Theta /= np.sum(p_y_given_Theta)
        return p_y_given_Theta
        
class AnglesUsePatternMeans(Angles):
    def __init__(self,sigs,angs = np.linspace(0,2*np.pi,180)):
        """...
        
        Parameters:
         sigs : a Signal object containing the raw training data from one of the transmitters that
                is used to build the estimate of the pattern. sigs.data will have a table containing:
                  [RSSI, ID(ord of a character id), Angle(radians), Time(milliseconds since transmitter turned on)]
         angs : search space (e.g. np.linspace(0,2*np.pi,180)).
         """
        self.avgRSSIs,_ = sigs.averageRSSIsAtAngle(detrend=True,smooth=True)
        self.angs = angs
        #I want to get the average angle change over 100ms, for this data
        #intervals = np.array([-200,-100,0,100,200])        
        #s = []
        #for i in range(1000):
        #    obs, obs_angles = getSample(sigs.data,747685+np.random.rand()*732482,intervals,raw=True,exclude_missing=5)
        #    if np.any(np.isnan(obs)): continue #skip if ALL have no data
        #    s.append(obs_angles)
        #np.median(np.diff(np.array(s))%(np.pi*2))
        
    def infer(self,time_intervals,obs):
#            #need to infer the observed angles of any missing packets (from the angles sent in the other packets)
#        diff = np.median(((obs_angles-(np.pi*intervals/1000))%(2*np.pi))[~np.isnan(obs)])
#        obs_angles[np.isnan(obs)] = (np.pi*intervals/1000+diff)[np.isnan(obs)]        
    
        log_p_y_given_Theta = []
        for i,theta in enumerate(self.angs):
            print("%d/%d" % (i,len(self.angs)),end="\r")
            p = 0
            #TODO Switch to using the actual ANGLES (or at least times)
            #get index of avgRSSI for each observation
            idxs = np.argmin(np.abs(self.avgRSSIs[:,0:1]-(((np.pi*time_intervals/1000)+theta)%(2*np.pi))),0)

            #the 'unattenuated' prediction (i.e. the training set - we pretend is unattenuated!)
            unatt_pred = self.avgRSSIs[idxs,1]       
            pred = unatt_pred-np.mean(unatt_pred[~np.isnan(obs)])
            p=np.nanprod(norm(pred,3).pdf(obs-np.nanmean(obs)))
            log_p_y_given_Theta.append(np.log(p))

        p_y_given_Theta = np.exp(log_p_y_given_Theta-np.max(log_p_y_given_Theta))
        p_y_given_Theta /= np.sum(p_y_given_Theta)
        return p_y_given_Theta        

class AnglesRejectionSampling(Angles):
    def __init__(self,sigs,usecache=True):
        """We use rejection sampling to pick angles that are most likely to have generated the observed signal strengths.
        
        Each time we ask for such a distribution of angles, we might either be using the same spacing of times, or
        we might be using a different spacing. If the latter, we need to build a new training set to sample from.
        
        Parameters:
         sigs : a Signal object containing tthe raw training data from one of the transmitters that
                is used to build the rejection tables. sigs.data will have a table containing:
                  [RSSI, ID(ord of a character id), Angle(radians), Time(milliseconds since transmitter turned on)]
         usecache : angle training data cache. Set to False to erase and reconstruct the cache
        """

        #we first need to generate a simulated list of observations... this is how we do it with
        #real empirically collected data, and later sample from it... we could instead skip this
        #step and just call getRSSI later when we want to getSample. TODO Decide if we should rewrite...

        data = sigs.data
        self.data = data

        #this is a dictionary, it is a cache/store of tables of training sets, each with a different set of time_intervals.
        self.training_data = {}
        
        if usecache:
            try:
                self.training_data = pickle.load(open('angle_inference_cache.pkl','rb'))
            except:
                print("Failed to open angle inference cache")
        
     

    def buildTable(self,time_intervals):
        """
        We need to build some training data for later rejection sampling.
        """        
        training_data = []
        while len(training_data)<10000: #Switched to while loop, as we often reject incomplete examples
#        for sampling in range(1000): #TODO we should use all of them.
            rssis, angle = getSample(self.data,np.random.uniform(self.data[0,-1]+1,self.data[-1,-1]-1), time_intervals=time_intervals)
            #print(rssis.shape)
            if np.any(np.isnan(rssis)):
                continue #Don't add samples with missing data!
            training_data.append(np.r_[rssis,angle]) #TODO! This will arbitrarily return angle of the first non-NaN element...fix!
        training_data = np.array(training_data)
        return training_data

    def infer(self,time_intervals,obs,rejection_threshold=4):
        """
        At the time_intervals specified, we have a series of observed signal strengths
        (in 'obs').

        Parameters:
        - time_intervals = the separation of times when we took the samples, e.g. [0, 0.1, 0.2 ,0.3]
        - obs = the signal strengths observed.

        Returns a list of angles that match the rejection sampling.       
        """
        #assert time_intervals[0] == 0, "We assume that the first value in time interval is now at 0."        
        time_intervals-=time_intervals[0]
        hash = hashlib.sha1(time_intervals).hexdigest()
        if hash in self.training_data:
            training_data = self.training_data[hash]
        else:
            training_data = self.buildTable(time_intervals)
            self.training_data[hash] = training_data
            pickle.dump(self.training_data,open('angle_inference_cache.pkl','wb'))

        obs = obs - obs[0]
        #matchingrowindices = np.all(np.abs(training_data[:,:-1]-obs)<rejection_threshold,1)
        matchingrowindices = np.mean((training_data[:,:-1]-obs)**2,1)<rejection_threshold
        return training_data[matchingrowindices,-1]
