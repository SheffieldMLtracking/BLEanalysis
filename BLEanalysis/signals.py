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

    def averageRSSIsAtAngle(self,transmitterId,detrend=False,smooth=False,smoothwindow=np.deg2rad(2)):
        """
        Returns a numpy array. Each row contains:
            angle
        """
        data = self.data[self.data[:,1]==ord(transmitterId),:]
        
        if detrend:
            f = filter(data[:,0],200,2)        
            data[:,0] = data[:,0]-f+np.mean(f)

        uniqueAngles = np.array(sorted(set(data[:, 2])))
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

