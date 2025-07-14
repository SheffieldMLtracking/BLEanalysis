import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as filter


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
        
        if filetype == 'log':
            self.data = self.parseDataFromDevBoard(self.filedata,self.transmitterIDs)
        if filetype == 'pcap':
            self.data = self.parseDataFromPcapPaste(self.filedata,self.transmitterIDs)    

        self.data = self.standardizeAnglesAndTimes(self.data,angleOffset,timeOffset)
            
    def summarise(self):
        print("Transmitter       Number of records")
        for transmitter_id in set(self.data[:,1]):
            print("%9s            %9d" % (chr(int(transmitter_id)),sum(self.data[:,1]==transmitter_id)))
        

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
        return data
    
    def parseDataFromPcapPaste(filedata, transmitterIDs):
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
        print(len(set(stuff)))
        print(len(rawUniquePackets))
        
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
        
        
    def getSample(self, burst_length, sample_interval, target_time=None, target_angle=None, accept_missing = 2,exclude_missing=10,raw=True):
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
        time_intervals = np.arange(-burst_length/2,burst_length,sample_interval)
        
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
        
