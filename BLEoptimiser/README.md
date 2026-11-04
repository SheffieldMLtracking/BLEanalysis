# Sensor Scheduler Algorithms

TODO explain repo. 

# EKF_demo_based_on_bee
- attempts to model the bee 

# EKF_demo_slower
- simple demo that is working on a slower time rate

# CalculatesGain 

Packages the example from the Modern Bayesian Experimental Design (that Mike coded for the toy example.)

### For dummy version
The bee is at an unknown location, $\theta \in [0,10]^2$.

We can make an observation, $y$ in direction $\xi$. This observation is a triplet of signal strengths at three angles around the actual angle (and transmitter) requested, $\xi$.

We need: $p(y|\theta, \xi)$:

I'll model that here with:
