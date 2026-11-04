"""
trying to encapsulate the code written in EKF_demo_v1

Things that need to be passed to be used for a scheduler algorithm:
- the transition model
- the tracker
- the sensors & their position
- the current time
"""

import numpy as np
import random
from datetime import datetime, timedelta
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthState
from stonesoup.types.state import State
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.platform.base import FixedPlatform 
from stonesoup.sensor.radar.radar import RadarRotatingBearing 
from stonesoup.types.state import State
from stonesoup.functions import mod_bearing
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.predictor.kalman import ExtendedKalmanPredictor
from stonesoup.types.track import Track
from stonesoup.plotter import AnimatedPlotterly
from stonesoup.types.groundtruth import GroundTruthPath

from collections import defaultdict   
import copy
from typing import List

class bee_simulation:
    """
    encapsulates all the code needed for bee simulation
    """

    def __init__(self):
        """
        sets up the ground truth path, radar positions, filter, 
        """

        np.random.seed(1990)
        random.seed(1990)

        # initialise ground_truth

        self.start_time = datetime.now().replace(microsecond=0)

        # Generate transition model
        # i.e. fk(xk|xk-1)
        velocity=0.01
        self.transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(velocity),
                                                                ConstantVelocity(velocity)])
        
        # timestamps the simulation is observed over
        self.time_max = 180 
        self.timesteps = [self.start_time + timedelta(seconds=k) for k in range(self.time_max)]

        # initial state of bee
        locx = -50
        locy = -50
        x_velocity = velocity
        y_velocity = velocity 

        # Generate transition model

        self.truth = GroundTruthPath([GroundTruthState([locx, x_velocity, locy, y_velocity], timestamp=self.timesteps[0])])

        for k in range(1, self.time_max):
            self.truth.append(
                GroundTruthState(self.transition_model.function(self.truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
                                timestamp=self.timesteps[k])
            )

        # setup the sensors

        transmitter_xx = [ -44.56495493,   17.54448279, -132.44165643,  127.58969322]
        transmitter_yy = [-106.13383274,  207.6112732,   207.36738564,  -76.98800717]

        self.sensor_platforms: List[FixedPlatform] = []

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
                position_mapping=(0, 2),
                noise_covar=noise_covar,
                #position=[[0], [0]],
                rpm=1,
                max_range=1600,
                fov_angle=np.radians(20),
                dwell_centre=StateVector([offset]),
                #clutter_model=None,
                #resolution=np.radians(res)  # resolution of sensor equal to distance between search cells
            )

            radar.timestamp = self.start_time
            platform.add_sensor(radar)

            self.sensor_platforms.append(platform)



        # initialise the filter to exactly where the ground truth is  

        prior = (GaussianState([[locx], [x_velocity], [locy], [y_velocity]],
                                        np.diag([0.5, 0.5, 0.5, 0.5]),
                                        timestamp=self.start_time))


        # set the track so it begins with the prior
        self.track = Track([prior])

        self.predictor = ExtendedKalmanPredictor(self.transition_model)

        self.updater = ExtendedKalmanUpdater(measurement_model=None)

        # "Dictionary mapping timestamps to sensor states"
        self.sensor_history = defaultdict(dict)
        self.sensor_set = [platform.sensors[0] for platform in self.sensor_platforms]

        self.sensor_measurements = []



    def run_at_regular_intervals(self):
        """
        runs the simulation in regular timesteps
        """

        for ts in self.timesteps[1:]:
            self.evolve_simulation_by_time(ts)


    def run_using_scheduler(self):
        """
        testing that the code works the same if the simulation is sampling at irregular periods
        """

        # TODO 
        pass


    def evolve_simulation_by_time(self, ts):
        """
        ts: timestamp that represents the new point in simulation to update and 
        """
        measurements = set()  # measurements at timestep 

        # check for measurements for each sensor
        for platform in self.sensor_platforms:
            sensor: RadarRotatingBearing = platform.sensors[0]

            # TODO check that this is right place to be updating sensor
            sensor.act(ts)

            measurements |= sensor.measure([self.truth[ts]])

            self.sensor_history[ts][sensor] = copy.copy(sensor)

        # evovle filter 
        prediction = self.predictor.predict(self.track[-1], timestamp=ts)

        if measurements:
            for detection in measurements:
                # if there are detections update
                hypothesis = SingleHypothesis(
                    prediction,
                    detection
                )
                prediction = self.updater.update(hypothesis)
            self.track.append(prediction)
        else:
            # no detection prediction only
            self.track.append(prediction)


    # METHODS FOR SENSOR SCHEDULING:
    def select_best_timestep(self, future_timesteps):
        """
        looks ahead k steps
        num_lookahead, number of timesteps ahead of current time to evaluate
        returns the best timestep
        """

        # imagine this works, and we return a next timestep to sample at
        return future_timesteps[-1] 





    # METHODS FOR VISUALISING:

    def plot_everything_using_plotterly(self):
        plotter = AnimatedPlotterly(self.timesteps, tail_length=0.1)

        plotter.plot_ground_truths(self.truth, [0,2])
        plotter.plot_sensors(self.sensor_platforms)
        plotter.plot_sensor_fov(self.sensor_set, sensor_history=self.sensor_history)
        plotter.plot_tracks(self.track, [0,2], uncertainty=True, plot_history=False, )
        plotter.fig.show()


sim = bee_simulation()
sim.run_at_regular_intervals()
sim.plot_everything_using_plotterly()