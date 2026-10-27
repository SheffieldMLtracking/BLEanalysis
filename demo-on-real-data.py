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

time_max = 1000  # timestamps the simulation is observed over
timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]


# initial state of bee
locx = -50
locy = -50
x_velocity = 0.01
y_velocity = 0.01 


# Generate transition model
from stonesoup.types.groundtruth import GroundTruthPath

truth = GroundTruthPath([GroundTruthState([locx, x_velocity, locy, y_velocity], timestamp=timesteps[0])])

for k in range(1, time_max):
    truth.append(
        GroundTruthState(transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
                         timestamp=timesteps[k])
    )


# Load Stone Soup materials
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix

from stonesoup.platform.base import FixedPlatform 
from stonesoup.sensor.radar.radar import RadarRotatingBearing 

from stonesoup.types.state import State

from stonesoup.functions import mod_bearing

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
        id=str(i)
    )

    # offset the starting orientation
    offset = mod_bearing(i * (np.pi/3))


    # Configure the radar noise, since we are using just a single dimension we need to specify only the
    # noise associated with the bearing dimension(, we assume a bearing accuracy of +/- 0.025 degrees for
    # each measurement
    noise_covar = CovarianceMatrix(np.array(np.diag([np.deg2rad(0.025) ** 2])))

    # instantiate the radar
    radar = RadarRotatingBearing(
        position_mapping=(0, 1),
        noise_covar=noise_covar,
        #position=[[0], [0]],
        rpm=1,
        max_range=1600,
        fov_angle=np.radians(30),
        dwell_centre=StateVector([offset]),
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
        sensor.act(ts)
        sensor_history[ts][sensor] = copy.copy(sensor)


plotter.fig.data = []   # resets the graph

plotter.plot_ground_truths(truth, [0,2])
plotter.plot_sensors(sensor_platforms)
plotter.plot_sensor_fov(sensor_set, sensor_history=sensor_history)
#plotter.fig.show()

# Load the filter components
from stonesoup.types.hypothesis import SingleHypothesis

from stonesoup.types.state import GaussianState

from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.types.track import Track


# initialise the filter to exactly where the ground truth is  

prior = (GaussianState([[locx], [x_velocity], [locy], [y_velocity]],
                                np.diag([0.5, 0.5, 0.5, 0.5]),
                                timestamp=start_time))


# set the track so it begins with the prior
track = Track([prior])

predictor = ExtendedKalmanPredictor(transition_model)

updater = ExtendedKalmanUpdater(measurement_model=None)

# "Dictionary mapping timestamps to sensor states"
sensor_history = defaultdict(dict)
sensor_set = [platform.sensors[0] for platform in sensor_platforms]

sensor_measurements = []

count = 0
for ts in timesteps[1:]:

    measurements = set()  # measurements at timestep 

    # check for measurements for each sensor
    for platform in sensor_platforms:
        sensor: RadarRotatingBearing = platform.sensors[0]

        sensor.act(ts)
        measurements |= sensor.measure([truth[ts]])

        sensor_history[ts][sensor] = copy.copy(sensor)

    # evovle filter 
    prediction = predictor.predict(track[-1], timestamp=ts)

    if measurements:
        for detection in measurements:
            # if there are detections update
            hypothesis = SingleHypothesis(
                prediction,
                detection
            )
            prediction = updater.update(hypothesis)
            pass
        track.append(prediction)
    else:
        # no detection prediction only
        track.append(prediction)


    # plotting the detections at timestep if any
    if measurements:
        for detection in measurements:
            
            sensor_measurements.append((ts, detection.state_vector)) 

    if (count % 20) == 0:
        print("\n x: " + str(track[-1].state_vector[0]))
        print("y: " + str(track[-1].state_vector[2]))
        print("x_vel: " + str(track[-1].state_vector[1]))
        print("y_vel: " + str(track[-1].state_vector[3]))
        print("truth: " + str(truth[ts].state_vector[0]) + ", " + str(truth[ts].state_vector[2]) )

    count = count + 1

