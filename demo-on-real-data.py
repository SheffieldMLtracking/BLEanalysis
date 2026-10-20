from stonesoup.sensor.radar.radar import RadarRotatingBearingRange

# https://stonesoup.readthedocs.io/en/v1.9.1/auto_tutorials/sensormanagement/01_SingleSensorManagement.html#create-sensors

import numpy as np
import random
from ordered_set import OrderedSet
from datetime import datetime, timedelta
from stonesoup.plotter import AnimatedPlotterly

start_time = datetime.now().replace(microsecond=0)

from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState


np.random.seed(1990)
random.seed(1990)

# Generate transition model
# i.e. fk(xk|xk-1)
transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),
                                                          ConstantVelocity(0.005)])




#yps = range(0, 100, 10)  # y value for prior state
#truths = OrderedSet() 
#ntruths = 3  # number of ground truths in simulation
time_max = 180  # timestamps the simulation is observed over
timesteps = [start_time + timedelta(seconds=k) for k in range(time_max)]

xdirection = 1
ydirection = 1

# truth = GroundTruthPath([GroundTruthState([-50, 1, -50, 1], timestamp=timesteps[0])])

# for k in range(1, time_max):
#     truth.append(
#         GroundTruthState(transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
#                          timestamp=timesteps[k])
#     )

from stonesoup.simulator.simple import SingleTargetGroundTruthSimulator
ground_truth_simulator = SingleTargetGroundTruthSimulator(
    transition_model=transition_model,
    initial_state=GroundTruthState([-50, 1, -50, 1], timestamp=timesteps[0]),
    number_steps=time_max,
    timestep=timedelta(seconds=1),
)

# # viewing the ground truth path
# plotter.fig.data = []
# plotter.plot_ground_truths(ground_truth[0][1], [0,2])
# plotter.fig
ground_truth = list(ground_truth_simulator)


plotter = AnimatedPlotterly(timesteps, tail_length=0.1)
# plotter.plot_ground_truths(truth, [0, 2])
# plotter.fig


plotter.fig.data = []

# Load Stone Soup materials
from stonesoup.types.state import State, GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.models.transition.linear import (CombinedLinearGaussianTransitionModel, ConstantVelocity)
from stonesoup.models.measurement.nonlinear import Cartesian2DToBearing

# Load the filter components
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.tracker.simple import SingleTargetTracker

from stonesoup.platform.base import FixedPlatform 
from stonesoup.sensor.radar.radar import RadarRotatingBearing 

from stonesoup.types.state import State

# setup the sensors

# setup the transmitter in same place as the norfolk park field experiment

transmitter_xx = [ -44.56495493,   17.54448279, -132.44165643,  127.58969322]
transmitter_yy = [-106.13383274,  207.6112732,   207.36738564,  -76.98800717]

sensor_platforms = []

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

plotter.plot_ground_truths(ground_truth[0][1], [0,2])
plotter.plot_sensors(sensor_platforms)
plotter.fig

# Load the platform detection simulator - Let's use a simulator for each track
# Instantiate the simulators

from stonesoup.simulator.platform import PlatformDetectionSimulator

radar_detection_simulator = PlatformDetectionSimulator(
    groundtruth=ground_truth_simulator,
    platforms=sensor_platforms)



# plotting the radar bearing measurements to sanity check model and visualise how it is working 
# for time, detections in radar_detection_simulator:
#     for detection in detections:
#         print(detection)
#         print("Measurement:", detection.state_vector)
#         print("Timestamp:", detection.timestamp)
#         print("Measurement model:", detection.measurement_model)

for time, detections in radar_detection_simulator:
    for detection in detections:
        radar_pos.append(detection.groundtruth_path.position)
        bearings.append(detection.state_vector[0])
        timestamps.append(detection.timestamp)



import plotly.graph_objects as go

# Your existing Stone Soup Plotterly instance:
# plotter = Plotterly(...)

ray_length = 100  # Length to display each bearing ray

# Each entry: (sensor_x, sensor_y, bearing_radians, timestamp)
bearing_data = [
    (-50, -50, 0.7, ),
    (50, -50, 2.1, ),
]

for sx, sy, bearing, timestamp in bearing_data:
    ex = sx + ray_length * np.cos(bearing)
    ey = sy + ray_length * np.sin(bearing)

    plotter.fig.add_trace(go.Scatter(
        x=[sx, ex],
        y=[sy, ey],
        mode="lines",
        name="Bearing detection",
        line=dict(dash="dash"),
        text=[str(timestamp), str(timestamp)],
        hovertemplate="Timestamp: %{text}<extra></extra>",
    ))

plotter.fig.show()