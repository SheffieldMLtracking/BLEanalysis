import hashlib
import numpy as np
import pickle
from BLEanalysis.signals import Signals
from scipy.stats import norm, circmean
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def normalise_logs_to_ps(logp):
    p = np.exp(logp - np.max(logp))
    p/= np.sum(p)
    return p

def normalize_radians(angle):
    """
    Makes an angle clockwise and between 0-2pi radians
    """
    normalized = angle % (2 * np.pi)
    return 2*np.pi - (normalized) if normalized >= 0 else 2*np.pi - (normalized + 2 * np.pi)

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
            sigs = Signals("../bluetooth_experiments/no rf amp experiments/noamploc2long.log",'d',angleOffset = 38)
       
        """
        - an array of averages. Each row contains:
            - angle in radians (in 1 degree steps, from 0 to 359).
            - the mean signal strength
            - the number of records used to compute the mean
            - the standard error on the mean estimate, i.e. np.std(matching_data)/np.sqrt(len(matching_data)).
        """
        self.avgRSSIs,_ = sigs.averageRSSIsAtAngle(detrend=True,smooth=True)
        self.noisevar = noisevar
        
    def infer(self,obs,obs_angles):
        """
        Returns the [unnormalised] log probabilities of a list of angles, given the observed signal strengths, and
        the angles recorded.

        obs: rssi values in the burst
        obs_angles: the corresponding angles where these rssis were measured (angle is from the packet)
        [these will both be NaN if a observation is missing]

        Note: self.avgRSSIs contains a list of 360 RSSIs that are provided by training data.
        
        Currently returns: logp,errs,avgAtAngles,keptObs
        """ 
        
        
        #obs and obs_angle can contain NaNs for missing observations.
        keep = ~np.isnan(obs_angles)
        
        #the data in the avg is organised into one degree per item, so this gives the indices:
        #if we want a higher resolution we would just multiply the number of degrees before calling 'astype'.
        #it's 'astype to int' to allow it to be used as an index.
        obs_angle_indices = np.rad2deg(obs_angles[keep]).astype(int)
        #obs_angle_indices shape: just a 1d array with one value per non-missing observation.
        
        #we build an array N_obs x 360, of the indicies of the avgRSSIs array we should look in, for offsets in 1 degree steps
        obs_angle_indices = (obs_angle_indices[:,None]+np.arange(360))%360
        #this will look like:
        #[...100, 101, 102...]
        #[...104, 105, 106...]

        #each of these indices is used to index the avgRSSIs array (each row is one degree, 
        #and the 1th column is the avg signal strenth at that angle). The result is a array of
        #the size as obs_angle_indices (i.e. N_obs x 360) but with the average signal strength
        #at that angle. So now: Each column of this array is the list of signal strengths we 
        #would have expected if we had been at that angle from the transmitter (i.e. the 34th
        #column would give us the expected signal strength if we were at 34 degrees from 'north'
        #wrt the transmitter). We can then simple test how close each of these possible columns
        #of signal strengths are to what we observed. [note that we need to 'normalise' by subtracting
        #the mean -- as the absolute values of these signal strengths are irrelevant, and we're
        #just interested in the relative signal strengths.
        #
        #we do the same subtraction on both (using their respective means)
        #TODO! This isn't perfect as this depends strongly on e.g. what happens if we miss a packet
        #at the peak of the pattern --> the mean will be artificially reduced... so might need work!
        
        #compute the difference for 1 degree step, for the N_observations we have.
        avgAtAngles = self.avgRSSIs[obs_angle_indices,1]
        keptObs = obs[keep]
        avgAtAngles = avgAtAngles - np.mean(avgAtAngles,0) #
        keptObs-=np.mean(keptObs)
        errs = avgAtAngles.T - keptObs # self.avgRSSIs[obs_angle_indices,1].T-obs[keep]
        
        #compute the SSE for each of these 1 degree steps, and divide by the noise-variance. This assumes a Gaussian noise model
        #for our noise.
        logp = -np.sum(errs**2,1)/self.noisevar #sse # TODO: replace gaussian with gaussian + some noise
        ##logp = -np.sum(np.abs(errs),1)/np.sqrt(self.noisevar) #exponential-dist. noise
        
        #if about 20% are missing, we should include this: 
        #
        #    log(p(y_missing_items|theta) * p(y_not_missing_items|theta))
        #    Nmissing * log(p(missing|theta)) + logp (computed above)  
        logp += np.sum(~keep)*np.log(1/5) 
        #logp = np.log(np.exp(logp)+1e-1000)

        #To compute the probability:
        #p = np.exp(logp - np.max(logp))
        #p/= np.sum(p)
        #plt.plot(p)
        return logp,errs,avgAtAngles,keptObs

class AnglesUsePeaks(Angles):
    def __init__(self, varThreshold = 5, windowLength = 5):
        self.varThreshold = varThreshold # TODO: Threshold to filter out bursts in which there is no obvious peak
        self.windowLength = windowLength
    
    def infer(self,obs,obs_angles):
        observations = []
        for rssi in range(len(obs)):
            if np.isnan(obs[rssi]):
                obs[rssi] = np.nanmean(obs)
        
        if np.std(observations) < self.varThreshold:
            return np.nan
            
        else:
            if len(obs) < 3:
                return np.nan
            smoothed = savgol_filter(obs, window_length=self.windowLength, polyorder=1) # TODO Better args?
            maxValueIndex = np.argmax(smoothed)
            return normalize_radians(obs_angles[maxValueIndex])


class AnglesUseRejectionSampling(Angles):
    def __init__(self, rejectionTableStep = 50, trainingData = None):
        self.rejectionTableStep = rejectionTableStep
        if trainingData == None:
            self.trainingData = Signals("../bluetooth_experiments/no rf amp experiments/noamploc2long.log",'d',angleOffset = 38).data
        self.rejectionTable = self.createRejectionTable(self.trainingData)

    def GetSample(self, data, time):
        times = np.array([time - 300, time - 100, time + 100, time + 300, time])
        idxs = np.argmin(np.abs(data[:, -1:] - times[None, :]), 0)
        sigs = data[idxs, 0]
        signaldiffs = sigs[:-1] - sigs[-1]
        # returns signal differences, middle selected packet's angle, raw signal strengths
        return signaldiffs , data[idxs, 2][-1], data[idxs, 0]

    def GetSampleFromBurst(self, burst, time):
        times = np.array([time - 300, time - 100, time + 100, time + 300, time])
        data = []
        for i in range(len(burst['rssis'])):
            data.append([burst['rssis'][i], burst['angles'][i], burst['times'][i]])
        data = np.array(data)
        idxs = np.argmin(np.abs(data[:, -1:] - times[None, :]), 0)
        sigs = data[idxs, 0]
        signaldiffs = sigs[:-1] - sigs[-1]
        return signaldiffs , data[idxs, 1][-1], data[idxs, 0]
        
    def createRejectionTable(self, trainingdata):
        """
        Parameters:
            trainingdata : A file of packets processed with Signals class consisting of one transmitter's packets
                           offset such that the peak signal strength occurs at 0 degrees.
        """
        record = []
        mint, maxt = np.min(trainingdata[:, -1]), np.max(trainingdata[:, -1])
        for i, t in enumerate(np.arange(mint + 100e3, maxt, self.rejectionTableStep)):
            signalstrengths, angle, rawss = self.GetSample(trainingdata, t)
            record.append([t] + list(signalstrengths) + [angle])
            if (i % 1000 == 0):
                print(str(int(t - mint)) + "ms out of " + str(int(maxt - mint)) + "ms added to table.")
        print("Rejection table built of " + str(int(maxt-mint)) + "ms of antenna profile.")
        return np.array(record)

    def inferFromBurst(self, burst, angleOffset = 0, sampleInterval = 10, filterStd = 1, filterLen = 5, filterMean = 10, filter = True, plot = False):
        """
        Does rejection sampling on a burst generated from signals.parseburst instead of a full file
        """
        mint, maxt = np.min(burst['times']), np.max(burst['times'])
        predictions = []
        avgss = [1000] * 20
        if plot:
            plt.figure(figsize=[5, 5])
        for i, t in enumerate(np.arange(mint, maxt, sampleInterval)):    
            ss, ang, rawss = self.GetSampleFromBurst(burst, t)
            del avgss[:1]
            avgss.append(rawss[-1])

            select = np.all(np.abs(ss - self.rejectionTable[:, 1:-1]) < 6, 1) #all 5 measurements must be <6 dB out
            predangle = self.rejectionTable[select, -1] 
            predangle -= (ang + angleOffset)
            predangle[predangle < 0] += np.pi * 2
            predangle[predangle > 2 * np.pi] -= np.pi * 2
            predangle = (2 * np.pi - predangle)
            
            if filter:
                if np.mean(rawss[-1]) < np.mean(avgss) + filterMean: continue
                #if len(predangle) < filterLen: continue
                if (np.std(predangle) > filterStd): continue
                if plot:
                    plt.plot(np.repeat(t, len(predangle)), predangle, '.k', markersize = 40, alpha = 0.01)
            else:
                if plot:
                    plt.plot(np.repeat(t, len(predangle)), predangle, '.k', markersize = 3, alpha = 0.01)
            predictions.append([t, (2 * np.pi - circmean(predangle)) % (2 * np.pi)])
            
        return [x for x in predictions if not np.isnan(x[1])]

    
    def infer(self, testdata, angleOffset = 0, sampleInterval = 10, filterStd = 0.15, filterLen = 5, filterMean = 5, filter = True, plot = False):
        """
        Parameters:
            testdata : all packets recieved by ONE reciever over time, as processed by signals class
        """
        mint, maxt = np.min(testdata[:, -1]), np.max(testdata[:, -1])
        predictions = []
        avgss = [1000] * 20
        if plot:
            plt.figure(figsize=[90, 8])
        for i, t in enumerate(np.arange(mint, maxt, sampleInterval)):    
            ss, ang, rawss = self.GetSample(testdata, t)
            del avgss[:1]
            avgss.append(rawss[-1])

            select = np.all(np.abs(ss - self.rejectionTable[:, 1:-1]) < 6, 1) #all 5 measurements must be <6 dB out
            predangle = self.rejectionTable[select, -1] 
            predangle -= (ang + angleOffset)
            predangle[predangle < 0] += np.pi * 2
            predangle[predangle > 2 * np.pi] -= np.pi * 2
            predangle = (2 * np.pi - predangle)
            
            if filter:
                if np.mean(rawss[-1]) < np.mean(avgss) + filterMean: continue
                if len(predangle) < filterLen: continue
                if (np.std(predangle) > filterStd): continue
                if plot:
                    plt.plot(np.repeat(t, len(predangle)), predangle, '.k', markersize = 40, alpha = 0.01)
            else:
                if plot:
                    plt.plot(np.repeat(t, len(predangle)), predangle, '.k', markersize = 3, alpha = 0.01)
            predictions.append([t, (2 * np.pi - circmean(predangle)) % (2 * np.pi)])
            
        return predictions

            

