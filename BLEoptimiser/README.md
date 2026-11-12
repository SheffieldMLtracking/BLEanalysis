# BLEoptimiser

Contains code to simulate the bees flight and schedule when to sample. 

`bee_simulation_discrete_time.py` is a simulation that expects discrete time intervals. In this case we can use the Stone Soup visualisation methods built on top of plotterly.

`bee_simulation.py` handles arbitary time intervals. 

# CalculatesGain 

Packages the example from the Modern Bayesian Experimental Design (that Mike coded for the toy example.)

### For dummy version
The bee is at an unknown location, $\theta \in [0,10]^2$.

We can make an observation, $y$ in direction $\xi$. This observation is a triplet of signal strengths at three angles around the actual angle (and transmitter) requested, $\xi$.

We need: $p(y|\theta, \xi)$:

I'll model that here with:
