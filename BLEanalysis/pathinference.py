import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib
matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as sci
import pyproj
#from matplotlib_scalebar.scalebar import ScaleBar

# TODO Work our which of these imports are necessary for this file

def crossProduct(a, b):
    """Computes cross product of vectos in two tensors.
    
    Parameters:
    a and b: The tensors containing the vectors to compute over.
    
    TODO: This might be in tensorflow now.
    TODO: Explain the shape that a and b can be.
    """
    size = a.shape[0]
    A = tf.Variable([[tf.zeros(size), -a[:, 2], a[:, 1]], [a[:, 2], tf.zeros(size), -a[:, 0]], [-a[:, 1], a[:, 0], tf.zeros(size)]])
    A = tf.transpose(A, [2, 0, 1])
    A = A[None, :, :, :]
    b = b[:, :, :, None]
    return (A@b)[:,:,:,0]

class Path:
    def __init__(self, observationTimes, observations, kernel, noiseScale, numberOfInducingPoints):
        """
        Performs variational inference using the 'single angle only' observations to work out possible paths of the tag.
        
        This uses observations that consist of single angles, rather than probability distributions of angles.
        
        Parameters:
         observationTimes : an array of the times the observations were made.
         observations : array of the angles they were (radians)
         kernel : an instantiation of a kernel class
         noiseScale : the standard deviation (TODO or variance?? TBC) of the Gasussian noise model
         numberOfInducingPoints : number of inducing points
         
         NOTE: I've removed inducingPointRange -- this is just set now inside the code. 
         the class places inducing points, with some before and after the data, this specifies the percentage
                              time before and after ( TODO: I think it makes more sense that this should depend on the lengthscale
                              or just be a fixed time? ).
        """
        self.observations = observations # Angle observations made
        self.observationTimes = observationTimes # Times observations were made at
        self.noiseScale = noiseScale # Noise scale of likelihood
        self.kernel = kernel # Kernel function
        self.jitter = 0 # Jitter applied to covariance matrix during training
        self.mean = [] # Posterior mean
        self.covariance = [] # Posterior covariance\
        self.Z = self.selectPoints(numberOfInducingPoints) # Select inducing points
        self.lossHistory = []


    def selectPoints(self, numInducingPoints, margin = 10):
        """Returns a tf array of one-dimensional (inducing) point locations, placed evenly over the domain of times in self.observationTimes, with a margin
        added.
        
        NOTE: Previously +/- a percentage of the time observations were made at, using the values specified by observationTimes and inducingPointRange.
        NOTE: This has been changed from "selectInducingPoints" as it is used by 'predict' (formally inference) to place test points.
        
        TODO: Could make margin depend on kernel lengthscale?
        """
        
        max_time = np.max(self.observationTimes) + margin
        min_time = np.min(self.observationTimes) - margin
        
        inputMatrix = []
        # Evenly spaced inducing points
        for vectorObserved in range(int(self.observations.shape[1] / 2)):
            inputMatrixEntry = np.c_[np.linspace(min_time, max_time, numInducingPoints), np.full(numInducingPoints, vectorObserved)]
            inputMatrix.extend(inputMatrixEntry)
        return tf.Variable(np.array(inputMatrix), dtype=tf.float32)   
    
    def predict(self, numOfPredictions=100, Xs=None):
        """Make predictions
        
        Parameters:
         Xs : The times to make the predictions, or None (default).
         numOfPredictions: If Xs set to None, place at evenly spaced times (spaced over the same domain as the training data)
         
        Returns:
         mean and covariance of these points."""
        # GP(Kzx Kzz^-1 y, Kzz - Kzx Kxx^-1 Kxz)
        
        if Xs is None:
            Xs = self.selectPoints(numOfPredictions)
            
        Kzz = self.kernel.K(self.Z, self.Z) + (np.eye(self.Z.shape[0], dtype=np.float32) * self.jitter)
        Kxx = self.kernel.K(Xs, Xs) + (np.eye(Xs.shape[0], dtype=np.float32) * self.jitter)
        Kxz = self.kernel.K(Xs, self.Z)
        Kzx = tf.transpose(Kxz)
        KzzinvKzx = tf.linalg.solve(Kzz, Kzx)
        KxzKzzinv = tf.transpose(KzzinvKzx)
        KxzKzzinvKzx = Kxz @ KzzinvKzx

        #TODO What is this????
        numInputs = int(Xs.shape[0] / int(self.observations.shape[1] / 2))
        mean = tf.transpose(tf.reshape((KxzKzzinv @ self.mean)[:, 0], [int(self.observations.shape[1] / 2), numInputs]), [1, 0])
        covariance = tf.transpose(tf.concat([(Kxx - KxzKzzinvKzx + KxzKzzinv @ (self.covariance @ tf.transpose(self.covariance)) @ KzzinvKzx)
                                      [i::numInputs, i::numInputs][:, :, None] for i in range(numInputs)], axis=2), [2, 0, 1])
        
        return mean, covariance

    def train(self, iterations=500, learningRate=0.15, numOfSamples = 100):
        """Perform VI - iteratively optimise a surrogate GP to most closely resemble the intractable true posterior distribution using
        the gradient of the ELBO at each step.
        
        Parameters:
         interations: default 500.
         learningRate: default 0.15.
         numOfSamples: default 100.
         
        Returns: true if successful, false if 
        """
        X = tf.Variable(np.c_[np.tile(self.observationTimes, int(self.observations.shape[1] / 2))[:, None],
                        np.repeat(np.arange(int(self.observations.shape[1] / 2)), len(self.observationTimes), axis=0)], dtype=tf.float32)
        y = tf.Variable(self.observations, dtype = tf.float32)
        
        optimiser = tf.keras.optimizers.Adam(learning_rate = learningRate)

        # Number of inducing points & inputs
        numInducingPoints = self.Z.shape[0]
        numInputs = int(X.shape[0] / int(self.observations.shape[1] / 2))
        
        # Mean of "surrogate" posterior
        surrogateMean = tf.Variable(tf.random.normal([numInducingPoints, 1]))
        # Use diagonal of covariance matrix for LU decomposition during iterative optimisation
        surrogateLowerDiagonal = tf.Variable(np.tril(0.01 * np.random.randn(numInducingPoints,numInducingPoints) + 1 * 
                                                     np.eye(numInducingPoints)), dtype=tf.float32)       

        # Prior distribution using K
        priorMean = tf.zeros([1, numInducingPoints])
        priorCovariance = tf.Variable(self.kernel.K(self.Z, self.Z))
        prior = tfd.MultivariateNormalFullCovariance(priorMean, priorCovariance + (np.eye(priorCovariance.shape[0]) * self.jitter))

        # GP(Kzx Kzz^-1 y, Kzz - Kzx Kxx^-1 Kxz)
        Kzz = self.kernel.K(self.Z, self.Z) + (np.eye(self.Z.shape[0], dtype=np.float32) * self.jitter)
        Kxx = self.kernel.K(X, X) + (np.eye(X.shape[0], dtype=np.float32) * self.jitter)
        Kxz = self.kernel.K(X, self.Z)
        Kzx = tf.transpose(Kxz)
        KzzinvKzx = tf.linalg.solve(Kzz, Kzx)
        KxzKzzinv = tf.transpose(KzzinvKzx)
        KxzKzzinvKzx = Kxz @ KzzinvKzx

        # Scaling factor for jitter hyperparameter, adjusted if cholesky decomp fails
        jitterScale = tf.eye(numInducingPoints) * 0.00001

        # Iteratively optimise surrogate distribution
        for iteration in range(iterations):
            with tf.GradientTape() as tape:

                # Form surrogate posterior
                surrogatePosterior = tfd.MultivariateNormalTriL(surrogateMean[:, 0], surrogateLowerDiagonal + jitterScale)
                # If it fails, break so that the jitter can be adjusted
                if np.any(np.isnan(surrogatePosterior.mean())):
                    return False # TODO This might be better handled with an exception.
                    
                # Calculate parameters of surrogate posterior
                posteriorSurrogateMean = (KxzKzzinv @ surrogateMean)[:,0]
                # TODO: use offdiagonal matrix or multiple diagonals in LL^(T)
                posteriorSurrogateCovariance = Kxx - KxzKzzinvKzx + KxzKzzinv @ ((surrogateLowerDiagonal + jitterScale) 
                                                                                 @ tf.transpose(surrogateLowerDiagonal+jitterScale)) @ KzzinvKzx
                covariance = tf.transpose(tf.concat([posteriorSurrogateCovariance[i::numInputs, i::numInputs] [:, :, None] 
                                                     for i in range(numInputs)], axis=2),[2, 0, 1])
                mean = tf.transpose(tf.reshape(posteriorSurrogateMean, [int(self.observations.shape[1] / 2),numInputs]), [1, 0])

                # Sample from surrogate posterior...
                samples = tfd.MultivariateNormalTriL(mean, tf.linalg.cholesky(covariance + tf.eye(int(self.observations.shape[1] / 2)) 
                                                                            * self.jitter)).sample(numOfSamples)
                        
                # Calculate distance between surrogate posterior samples and observations
                distance = tf.norm(crossProduct(y[:, 3:], samples - y[:, :3]), axis=2) / tf.norm(y[:, 3:], axis=1)
                # Calculate the ELBO using the log likelihood of each distance
                ELBO = -(tf.reduce_mean(tf.reduce_sum(tfd.Normal(0, self.noiseScale).log_prob(distance), 1)) 
                         - tfd.kl_divergence(surrogatePosterior, prior))
            
            # Use tf gradients to optimise ELBO
            gradients = tape.gradient(ELBO, [surrogateMean, surrogateLowerDiagonal])
            optimiser.apply_gradients(zip(gradients, [surrogateMean, surrogateLowerDiagonal]))

            self.lossHistory.append(ELBO.numpy())
            # Print progress
            if iteration % 50 == 0:
                print("At iteration: %6d, loss is: %9.0f" % (iteration, ELBO.numpy()[0]))
                
        # Following completion of optimisation, store final variational parameters
        self.mean = surrogateMean
        self.covariance = surrogateLowerDiagonal
        return True

    def run(self, iterations=500, learningRate=0.15, numOfSamples = 100, jitterStart=0.000001):  
        """
        Wrapper function for training variational parameters.
        
        If training fails due to inability to invert Kzz, jitter is increased
        """

        self.jitter = jitterStart
        for i in range(10):
            if self.train(iterations, learningRate, numOfSamples):
                print("Training successful!")
                return
            else:
                self.lossHistory = [] # Clear loss history
                self.jitter *= 10
                print("Increasing jitter to %0.5f" % self.jitter)
