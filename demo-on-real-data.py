import numpy as np
import random
from datetime import datetime, timedelta
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator

np.random.seed(1990)
random.seed(1990)

start_time = datetime.now().replace(microsecond=0)

# Generate transition model
# i.e. fk(xk|xk-1)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])

time_max = 180  # timestamps the simulation is observed over
timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]

xdirection = 1
ydirection = 1

ground_truth_simulator = SingleTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GroundTruthState([-50, 1, -50, 1], timestamp=timesteps[0]),
    number_steps=time_max,
    timestep=timedelta(seconds=1),
)

ground_truth = list(ground_truth_simulator)


# Load Stone Soup materials
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix

from stonesoup.platform.base import FixedPlatform 
from stonesoup.sensor.radar.radar import RadarRotatingBearing 

from stonesoup.types.state import State

# setup the sensors

# setup the transmitter in same place as the norfolk park field experiment

transmitter_xx = [ -44.56495493,   17.54448279, -132.44165643,  127.58969322]
transmitter_yy = [-106.13383274,  207.6112732,   207.36738564,  -76.98800717]

from typing import List
sensor_platforms: List[FixedPlatform] = []

for i in range(len(transmitter_xx)):
    platform = FixedPlatform(
        State([transmitter_xx[i], transmitter_yy[i]]),  
        position_mapping=[0, 1],
    )

    # Configure the radar noise, since we are using just a single dimension we need to specify only the
    # noise associated with the bearing dimension, we assume a bearing accuracy of +/- 0.025 degrees for
    # each measurement
    noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(0.025) ** 2])))

    # instantiate the radar
    radar = RadarRotatingBearing(
        position_mapping=(0, 1),
        noise_covar=noise_covar,
        #position=[[0], [0]],
        rpm=30,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([np.pi]),
        #clutter_model=None,
        #resolution=np.radians(res)  # resolution of sensor equal to distance between search cells
    )

    radar.timestamp = start_time
    platform.add_sensor(radar)

    sensor_platforms.append(platform)


from stonesoup.plotter import AnimatedPlotterly

import copy
from collections import defaultdict

plotter = AnimatedPlotterly(timesteps, tail_length=0.1)


# "Dictionary mapping timestamps to sensor states" -- but we are just copying the sensor object? not sure about this tbh
sensor_history = defaultdict(dict)
sensor_set = [platform.sensors[0] for platform in sensor_platforms]
for ts in timesteps:

    # for every sensor
    for platform in sensor_platforms:
        sensor = platform.sensors[0]
        sensor_history[ts][sensor] = copy.copy(sensor)


plotter.fig.data = []   # resets the graph

plotter.plot_ground_truths(ground_truth[0][1], [0,2])
plotter.plot_sensors(sensor_platforms)
plotter.plot_sensor_fov(sensor_set, sensor_history=sensor_history)
plotter.fig.show()

# Load the platform detection simulator - Let's use a simulator for each track
# Instantiate the simulators

from stonesoup.simulator.platform import PlatformDetectionSimulator

radar_detection_simulator = PlatformDetectionSimulator(
    groundtruth=ground_truth_simulator,
    platforms=sensor_platforms)



# plotting the radar bearing measurements to sanity check model and visualise how it is working 

timestamps = []
bearings = []
radar_pos = []

for time, detections in radar_detection_simulator:
    for detection in detections:
        
        radar_pos.append(detection.groundtruth_path.position)
        bearings.append(detection.state_vector[0])
        timestamps.append(detection.timestamp)



