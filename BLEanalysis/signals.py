import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as filter
import pyproj

class Signals:
    def __init__(self,filename=None,transmitterIDs=None,filedata=None,filetype=None,angleOffset=0,timeOffset=0):
        """
        Class to load, parse and store the raw data from the receiver.

        See 'data' attribute for numpy array of the raw data, stores with follow columns:
        RSSI, ID(ord of a character id), Angle(radians), Time(milliseconds since transmitter turned on)
        
        Load in a file from either a .pcap file (off a tag) or a .log file
        (produced by active_scanner.c running on a DA14531 dev board).

        arguments:
         filename : filename of data file (default None)
         transmitterIDs : if specified, restricts to these transmitters.
         fieldata : instead of a file, can provide string of data (previously loaded from a file) (default None)
         filetype : either 'pcap' or 'log' (or leave as None, for type to be determined from filename).
         angleOffset : TODO 
         timeOffset : TODO
        """

        self.angleOffset = angleOffset
        self.timeOffset = timeOffset
        
        if filetype is None:
            if filename[-3:]=='log': filetype='log'
            if filename[-4:]=='pcap': filetype='pcap'
        if filetype is None:
            raise Exception("File datatype unknown or unspecified.") 

        self.filedata = filedata
        if filename is not None:
            if filedata is not None:
                raise Exception('Cannot specify both filename and filedata!')
            with open(filename, 'r') as f:
                self.filedata = f.read()

        if self.filedata is None:
            raise Exception("No data to parse.") 

        self.transmitterIDs = transmitterIDs
        self.transmitterLocations = {}
        self.normalizedTransmitterLocations = {}
        self.coordNormalisationFactor = [0,0]
        
        if filetype == 'log':
            self.data = self.parseDataFromDevBoard(self.filedata,self.transmitterIDs)
        if filetype == 'pcap':
            self.data = self.parseDataFromPcapPaste(self.filedata,self.transmitterIDs)    

        self.data = self.standardizeAnglesAndTimes(self.data,angleOffset,timeOffset)
            
    def summarise(self, showNormalised = True):
        print("Transmitter       Number of records")
        for transmitter_id in set(self.data[:,1]):
            print("%9s            %9d" % (chr(int(transmitter_id)),sum(self.data[:,1]==transmitter_id)))
        plt.axis("equal")
        if showNormalised:
            for ID in self.transmitterIDs:
                plt.scatter(self.normalizedTransmitterLocations[ID][0][1],self.normalizedTransmitterLocations[ID][0][0])
                plt.text(self.normalizedTransmitterLocations[ID][0][1],self.normalizedTransmitterLocations[ID][0][0],str(ID))
        else:
            for ID in self.transmitterIDs:
                plt.scatter(self.transmitterLocations[ID][0][1],self.transmitterLocations[ID][0][0])
                plt.text(self.transmitterLocations[ID][0][1],self.transmitterLocations[ID][0][0],str(ID))


    def parseBursts(self, burstInterval= 5000, offset = 0):
        """
        Function to take the data parsed by this object using parseDataFromDevBoard or parseDataFromPcapPaste,
        and turn it into a struct with the burst scans separated out.
            transmitterLocations : 2D array comprising of [transmitter ID, transmitter lat, transmitter lon]
            burstInterval : the space between bursts of scanning, if not specified separates data into 2 second bursts (used for data in which
                            scanning is constant and not in bursts)
        """
        bursts = []
        times = []
        
        for transmitterID in self.transmitterIDs:
            # Get only data from one transmitter at a time 
            currentTransmitterData = []
            for packet in self.data:
                if packet[1] == float(ord(transmitterID)):
                    currentTransmitterData.append(packet)

            currentTransmitterData = np.array(currentTransmitterData)

            if burstInterval > 0:
                
                time_diffs = np.diff(currentTransmitterData[:, 3])
                split_indices = np.where(time_diffs > burstInterval)[0] + 1

                # Split into clusters
                clusters = np.split(currentTransmitterData, split_indices)
            else:
                packettimes = currentTransmitterData[:, 3]
    
                # Ensure data is sorted by time (optional but recommended)
                sort_idx = np.argsort(packettimes)
                currentTransmitterData = currentTransmitterData[sort_idx]
                packettimes = packettimes[sort_idx]
    
                # Compute bin edges from min to max time
                start_time = packettimes[0]
                end_time = packettimes[-1]
                bins = np.arange(start_time, end_time + 2000, 2000)
    
                # Digitize each timestamp to a bin
                bin_indices = np.digitize(packettimes, bins)  # bin index starts from 1
    
                # Group rows by bin index
                clusters = [currentTransmitterData[bin_indices == i] for i in np.unique(bin_indices)]

            for cluster in clusters:
                currentCluster = {}
                currentCluster['transmitter_position'] = np.array(self.normalizedTransmitterLocations[transmitterID])
                currentCluster['rssis'] = cluster[:, 0]
                currentCluster['angles'] = cluster[:, 2]
                currentCluster['times'] = cluster[:, 3]
                bursts.append(currentCluster)
                times.append(int(np.mean(cluster[:, 3])))
        
        # returns the burst object and the times at which these bursts occur
        return bursts,times
        

    def parseDataFromDevBoard(self, filedata, transmitterIDs=None):
        """Split data from .log file, generated from the dev board active_scanner.c into packets

        Parameters:
         filedata : string from .log file
         transmitterIDs : the ids of the transmitters in a list to use, e.g. ['a','b','c','d'], or None for all of them.
        """
          
        splitFile = filedata[54:].split('\n')
        splitFile = splitFile[1:len(splitFile)-1]
        data = []
        for i in range(len(splitFile)):
            if(len(splitFile[i]) == 29) and (transmitterIDs is None or splitFile[i][27:28] in transmitterIDs):
                dataPoint = []
                dataPoint.append(-int(splitFile[i][7:9])) # RSSI
                dataPoint.append(ord(splitFile[i][27:28])) # ID
                dataPoint.append(int(int(splitFile[i][23:25] + splitFile[i][26:27], 16))) # Angle (need to standardize between 0 and 359 degrees)
                dataPoint.append(int(splitFile[i][14:16] + splitFile[i][17:19] + splitFile[i][20:22], 16)) # Relative time 
                data.append(dataPoint)
        return np.array(data).astype(float)

    def standardizeAnglesAndTimes(self,data,angleOffset,timeOffset):
        print("Standardising angles and times (shifting by %0.2f degrees)" % angleOffset)
        # Standardize the angles
        data[:,2] = np.deg2rad((data[:,2] + angleOffset) % 360)
        # Standardize the times
        data[:,3]-= timeOffset
        data = data[np.argsort(data[:, -1]), :]
        return data

    def configureTransmitter(self, transmitterID, timeOffset, angleOffset, transmitterLat, transmitterLon):
        # Express transmitter position in northing and easting
        P = pyproj.Proj(proj='utm', zone=30, ellps='WGS84', preserve_units=True)
        transmitterLon, transmitterLat = P(transmitterLon, transmitterLat)
        self.transmitterLocations[transmitterID] = [[transmitterLat, transmitterLon]]
        
        # Apply time and angle offset to all packets from this transmitter
        for packet in range(len(self.data)):
            if self.data[packet][1] == ord(transmitterID):
                self.data[packet][-1] -= timeOffset
                self.data[packet][2] -= angleOffset

    def noramliseTransmitterLocs(self):
        # Get average of coords to normalise
        coords = [val[0] for val in self.transmitterLocations.values()]  # get [x,y] pairs
        xs = [x for x, y in coords]
        ys = [y for x, y in coords]
        self.coordNormalisationFactor[0] = sum(xs) / len(xs)
        self.coordNormalisationFactor[1] = sum(ys) / len(ys)

        for ID in self.transmitterIDs:
            currentLat = self.transmitterLocations[ID][0][0]
            currentLon = self.transmitterLocations[ID][0][1]
            # Update & store normalised coordinates of transmitter positions
            self.normalizedTransmitterLocations[ID] = [[currentLat - self.coordNormalisationFactor[0],
                                                                   currentLon - self.coordNormalisationFactor[1]]]
    
    def subsampleBurst(self, dictionary, n, burstLength, method="random"):

        # Get indexes of only times that fall within the burstLength window (regarded as starting the beginning of bursts)
        dictionary['times'] = [x-np.min(dictionary['times']) for x in dictionary['times']]
        packetIndexes = np.where(np.array(dictionary['times']) < burstLength)

        # Select these indexes from the burst
        dictionary['rssis'] = np.array(dictionary['rssis'])[packetIndexes]
        dictionary['angles'] = np.array(dictionary['angles'])[packetIndexes]
        dictionary['times'] = np.array(dictionary['times'])[packetIndexes]
        
        if n >= len(dictionary['rssis']):
            return dictionary
        elif method == "random":
            samples = {}
            samples['transmitter_position'] = dictionary['transmitter_position']
            samples['rssis'] = []
            samples['angles'] = []
            samples['times'] = []
            
            # Randomly select unique indices
            rand_indices = np.random.choice(len(dictionary['rssis']), size=n, replace=False)
            
            # Collect samples at those indices
            samples['rssis'] = np.array([dictionary['rssis'][i] for i in rand_indices])
            samples['angles'] = np.array([dictionary['angles'][i] for i in rand_indices])
            samples['times'] = np.array([dictionary['times'][i] for i in rand_indices])
            return samples
            
        elif method == "linspaceTimes":
            samples = {}
            samples['transmitter_position'] = dictionary['transmitter_position']
            samples['rssis'] = []
            samples['angles'] = []
            samples['times'] = []
            
            # Times must be sorted, this happens in the parse burst method
            times = np.linspace(dictionary['times'][0], dictionary['times'][-1], n)
            indices = np.searchsorted(dictionary['times'], times)
            indices = np.clip(indices, 0, len(times)-1)

            samples['rssis'] = np.array([dictionary['rssis'][i] for i in indices])
            samples['angles'] = np.array([dictionary['angles'][i] for i in indices])
            samples['times'] = np.array([dictionary['times'][i] for i in indices])
            return samples
        elif method == "linspaceIdxs":
            samples = {}
            samples['transmitter_position'] = dictionary['transmitter_position']
            samples['rssis'] = []
            samples['angles'] = []
            samples['times'] = []

            idxs = np.linspace(0, len(dictionary['rssis'])-1, n)

            # Collect samples at those indices
            samples['rssis'] = np.array([dictionary['rssis'][int(i)] for i in idxs])
            samples['angles'] = np.array([dictionary['angles'][int(i)] for i in idxs])
            samples['times'] = np.array([dictionary['times'][int(i)] for i in idxs])
            return samples
            
    def parseDataFromPcapPaste(self, filedata, transmitterIDs):
        """
        TODO: Untested
        """
        splitFile = filedata.split('\n')
        rawPackets = []
        stuff = []
        for packet in splitFile:
            #print(".",end="",flush=True)
            packet = packet.split('\t')
            if( packet[2] == 'DialogSemico_70:00:04' ):
                stuff.append(packet[3])
            if( packet[2] == 'DialogSemico_70:00:04' and len(packet[3]) == 17 and -int(packet[3][0:2], 16) >= -95
            and -int(packet[3][0:2], 16) <= -25 and int(packet[3][-5:-3] + packet[3][-2], 16) <= 360):
                rawPackets.append(packet[3])

        rawUniquePackets = set(rawPackets)
        
        data = []
        for packet in rawUniquePackets:
            if transmitterIDs is None or packet[-1] in transmitterIDs:
                dataPoint = []
                dataPoint.append(-int(packet[0:2],16)) # RSSI
                dataPoint.append(ord(packet[-1])) # ID
                dataPoint.append(int(packet[-5:-3] + packet[-2], 16)) # ANGLE
                dataPoint.append(int(packet[3:5] + packet[6:8] + packet[9:11],16))
                data.append(dataPoint)

        return np.array(data).astype(float)

    def averageRSSIsAtAngle(self,transmitterId=None,detrend=False,smooth=False,smoothwindow=np.deg2rad(2)):
        """
        Finds the average RSSI for each DEGREE (but returns in radians), using this received dataset. Note that the transmitterId might
        already have been selected in the constructor.
        
        detrend = whether to subtract a smoothed rolling average from the dataset
        smooth = whether to smooth over a short period the dataset
        smoothwindow = if smooth set to True the smoothwindow controls the width of the window
        
        Returns a tuple:
         - an array of averages. Each row contains:
            - angle in radians (in 1 degree steps, from 0 to 359).
            - the mean signal strength
            - the number of records used to compute the mean
            - the standard error on the mean estimate, i.e. np.std(matching_data)/np.sqrt(len(matching_data)).
         - an array of all the records used at each angle to compute the array.
        """
        if transmitterId is not None:
            data = self.data[self.data[:,1]==ord(transmitterId),:].copy()
        else:
            data = self.data.copy()
        
        if detrend:
            f = filter(data[:,0],200,2)        
            data[:,0] = data[:,0]-f+np.mean(f)

        uniqueAngles = np.deg2rad(np.arange(360))

        #uniqueAngles = np.array(sorted(set(data[:, 2])))
        avgRSSIatAngle = []
        raw_data_atAngle = []
        

        for angle in uniqueAngles:
            if smooth:
                matching_data = data[(data[:,2]>=angle-smoothwindow) & (data[:,2]<=angle+smoothwindow),0]
            else:
                matching_data = data[data[:,2]==angle,0]
            raw_data_atAngle.append(matching_data)
            avgRSSIatAngle.append([angle,np.mean(matching_data),len(matching_data),np.std(matching_data)/np.sqrt(len(matching_data))])
        return np.array(avgRSSIatAngle), raw_data_atAngle
        
        
    def getSample(self, burst_length, sample_interval, target_time=None, target_angle=None, accept_missing = 0,exclude_missing=None,raw=True):
        """Generates one burst with required parameters
        This method is for sampling a whole bunch of times at once, as part of generating the training data.
        Specifically, using the object's 'data' array, containing columns [RSSI, ID, Angle(radians), Time(milliseconds)]
        and the time we want to sample near (time) and the intervals (time_intervals) returns a single instance of this.
        
        burst_length = how many milliseconds the burst takes
        sample_interval = how long between samples during the burst
        target_time = time you want to sample at [optional]
        target_angle = angle you want to sample at [optional]
        accept_missing = maximum number of samples without an observation (doesn't work with target_time).
        exclude_missing = if a number, sets to nan if no samples are available within that time
                       (default=10ms, which is equivalent to 1.8 degrees). Set to None to disable
        Note: If the first element of the returned RSSIs is set to NaN, and raw is False, then all the elements will be NaN
         ( as the first one is subtracted from the rest )."""
        
        if target_time is not None and target_angle is not None:
            raise Exception("Need to select EITHER target_time OR target_angle but not both.")
        time_intervals = np.arange(0,burst_length,sample_interval)
        
        data_starttime = np.min(self.data[:,3])
        data_endtime = np.max(self.data[:,3])
        data_length = data_endtime - data_starttime
        while True:
            if target_time is not None:
                time = target_time
            else:
                if target_angle is not None:            
                    index = np.random.choice(np.where(np.abs(self.data[:,2]-target_angle)<np.deg2rad(1.2))[0])
                    time = self.data[index,3]
                else:
                    time = data_starttime+np.random.rand()*data_length

            times = np.array(time_intervals) + time
            # find index of the packets in our data set, closest to these times.
            index = np.argmin(np.abs(self.data[:, -1:] - times[None,:]), 0)       
            rssis = self.data[index,0]
            angles = self.data[index,2]
            if exclude_missing is not None:
                rssis[(np.abs(self.data[:, -1:] - times[None,:]))[index,range(len(index))]>=exclude_missing]=np.NaN
                angles[(np.abs(self.data[:, -1:] - times[None,:]))[index,range(len(index))]>=exclude_missing]=np.NaN
            if not raw:       
                rssis-=rssis[0] #NOTE: I've switched to making the first time the angle index time as we could have an unknown number of time_intervals. 
            if target_time is not None: break
            if np.sum(np.isnan(rssis))>accept_missing: continue   
            break
            
        return rssis, angles

    def getDataFromOneTransmitter(self, transmitterID):
        packets = []
        for packet in range(len(self.data)):
            if self.data[packet][1] == ord(transmitterID):
                packets.append(self.data[packet])
        return np.array(packets)
        
