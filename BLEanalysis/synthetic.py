import numpy as np
import matplotlib.pyplot as plt
#from matplotlib_scalebar.scalebar import ScaleBar

class SimpleDemo:
    def __init__(self):
        """Create simple straight line, synthetic data"""
        
        # Location of transmitters
        a = [28,40,0]
        e = [63,15,0]
        self.stationlocations = np.array([a,e])
        # Time for each observation in order that observations are stored in observations[]
        self.obstimes = np.linspace(0,5,30)

        # Populate observations[] with [x, y, z] vectors
        self.observations = []
        self.trueLocations = []
        for i,t in enumerate(self.obstimes):
            location = np.array([(1*t)*10+10 + np.random.randint(1,5),(1*t)*10 + np.random.randint(1,5),0])
            vect = location-self.stationlocations
            vect/= np.linalg.norm(vect,axis=1)[:,None]
            possibleobs = np.c_[self.stationlocations,vect]
            obs = possibleobs[i%2,:]
            self.observations.append(obs)
            self.trueLocations.append(location)
        # observations[observer_x, observer_y, observer_z (unused), observation_x, observation_y, observation_z (unused)]
        # NOTE: Need to convert bearing in radians to directional vector
        # V.x = cos(B)
        # V.y = sin(B)
        self.observations = np.array(self.observations)
        # Tuples of x,y,z(unused) coords in order
        self.trueLocations = np.array(self.trueLocations)

    def plot(self):
        """Plot synthetic data
        
        TODO Switch to using axis object
        """
        plt.axis('equal')
        plt.plot(self.trueLocations[:,0],self.trueLocations[:,1],'x-')
        plt.axis('equal')
        for obs in self.observations:
            plt.plot([obs[0],obs[0]+obs[3]*30],[obs[1],obs[1]+obs[4]*30],color='grey',alpha=0.1)
        plt.title("Synthetic Path")
        plt.xlabel("x coords")
        plt.ylabel("y coords")
