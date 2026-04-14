import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.math as tfmath
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib
matplotlib.rcParams["axes.formatter.limits"] = (-99, 99)
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import scipy.stats as sci
import pyproj
from matplotlib_scalebar.scalebar import ScaleBar
import math

# Cross product of a & b
def crossProduct(a, b):
    size = a.shape[0]
    A = tf.Variable([[tf.zeros(size), -a[:, 2], a[:, 1]], [a[:, 2], tf.zeros(size), -a[:, 0]], [-a[:, 1], a[:, 0], tf.zeros(size)]])
    A = tf.transpose(A, [2, 0, 1])
    A = A[None, :, :, :]
    b = b[:, :, :, None]
    return (A@b)[:,:,:,0]

class Kernel:
    def __init__(self):
        """Base kernel class"""
        raise NotImplementedError
        
    def K(self, X, Xprime):
        raise NotImplementedError
        
class ExponentiatedQuadraticKernel(Kernel):
    def __init__(self, lengthscale, scalefactor):
        """Computes the EQ kernel, with the second column a dimension index.
        
        the covariance method returns: scalefactor^2 * exp((-(x-x')^2) / (2*ls^2))
        if in same dimension; otherwise returns zero.
        
        Parameters:
         lengthscale: lengthscale of the kernel
         scalefactor: the premultiplier.

        """
        self.lengthscale = lengthscale
        self.scalefactor = scalefactor
        
    def K(self, X, Xprime):
        """EQ kernel: scale^2 * exp((-(x-x')^2) / (2*ls^2))
        
        Parameters:
        X and Xprime:
        
        Returns the covariacne between points in X and Xprime.
        
        TODO Computing more than we need (as 2/3rds gets set to zero.
        """
        covariance = (self.scalefactor ** 2) * np.exp(-(X[:,0:1]-Xprime[:,0:1].T)**2 / (2*self.lengthscale**2))
        axsel = X[:,1:2]==Xprime[:,1:2].T
        covariance *= axsel
        return covariance

class TensorFlowExponentiatedQuadraticKernel(Kernel):
    def __init__(self, lengthScale, scaleFactor):
        self.lengthScale = lengthScale
        self.scaleFactor = scaleFactor
        
    def K(self, X, Xprime):
        # EQ kernel: scale^2 * exp((-(x-x')^2) / (2*ls^2))
        covariance = (self.scaleFactor ** 2) * np.exp(-np.sum((np.subtract(X[:, None], Xprime[None, :])) ** 2 / (2 * self.lengthScale ** 2), 2))
        print(covariance)
        axsel = tf.cast((X[:,1][:,None]==Xprime[:,1][None,:]), dtype=tf.float32)
        covariance *= axsel
        return covariance

class Path:
    def __init__(self, observationTimes, observations, kernel, noiseScale, numberOfInducingPoints, inducingPointRange):
        self.observations = observations # Angle observations made
        self.observationTimes = observationTimes # Times observations were made at
        self.noiseScale = noiseScale # Noise scale of likelihood
        self.kernel = kernel # Kernel function
        self.inducingPointRange = inducingPointRange # Range outside the time series that inducing points are selected from/to
        self.jitter = 0 # Jitter applied to covariance matrix during training
        self.mean = [] # Posterior mean
        self.covariance = [] # Posterior covariance\
        self.Z = self.SelectInducingPoints(numberOfInducingPoints) # Select inducing points
        self.lossHistory = []
        self.posteriorCovarianceForInference = 0
        self.posteriorMeanForInference = 0
        self.Kzz = []
        self.Kxx = []
        self.Kxz = []
        self.Kzx = []
        self.precomputeFlag = False

    # Get inducing points between +/- a percentage of the time observations were made at
    def SelectInducingPoints(self, numInducingPoints):
        max_time = np.max(self.observationTimes) + ((np.max(self.observationTimes) - np.min(self.observationTimes))
                                                    * self.inducingPointRange)
        min_time = np.min(self.observationTimes) - ((np.max(self.observationTimes) - np.min(self.observationTimes))
                                                    * self.inducingPointRange)
        
        inputMatrix = []
        # Evenly spaced inducing points
        for vectorObserved in range(int(self.observations.shape[1] / 2)):
            inputMatrixEntry = np.c_[np.linspace(min_time, max_time, numInducingPoints), np.full(numInducingPoints, vectorObserved)]
            inputMatrix.extend(inputMatrixEntry)
        return tf.Variable(np.array(inputMatrix), dtype=tf.float32)   
    
    # Make predictions at evenly spaced times
    def Inference(self, numOfPredictions):
        # GP(Kzx Kzz^-1 y, Kzz - Kzx Kxx^-1 Kxz)
        Xs = self.SelectInducingPoints(numOfPredictions)
        Kzz = self.kernel.K(self.Z, self.Z) + (np.eye(self.Z.shape[0], dtype=np.float32) * self.jitter)
        Kxx = self.kernel.K(Xs, Xs) + (np.eye(Xs.shape[0], dtype=np.float32) * self.jitter)
        Kxz = self.kernel.K(Xs, self.Z)
        Kzx = tf.transpose(Kxz)
        KzzinvKzx = tf.linalg.solve(Kzz, Kzx)
        KxzKzzinv = tf.transpose(KzzinvKzx)
        KxzKzzinvKzx = Kxz @ KzzinvKzx

        numInputs = int(Xs.shape[0] / int(self.observations.shape[1] / 2))
        mean = tf.transpose(tf.reshape((KxzKzzinv @ self.mean)[:, 0], [int(self.observations.shape[1] / 2), numInputs]), [1, 0])
        covariance = tf.transpose(tf.concat([(Kxx - KxzKzzinvKzx + KxzKzzinv @ (self.covariance @ tf.transpose(self.covariance)) @ KzzinvKzx)
                                      [i::numInputs, i::numInputs][:, :, None] for i in range(numInputs)], axis=2), [2, 0, 1])
        
        return mean, covariance

    # Perform VI - iteratively optimise a surrogate GP to most closely resemble the intractable true posterior distribution using
    # the gradient of the ELBO at each step.
    def Train(self, iterations=500, learningRate=0.15, numOfSamples = 100):
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
        if self.precomputeFlag == False:
            self.Kzz = self.kernel.K(self.Z, self.Z)
            self.Kxx = self.kernel.K(X, X)
            self.Kxz = self.kernel.K(X, self.Z)
            self.Kzx = tf.transpose(self.Kxz)
            self.precomputeFlag = True
        print(X.shape)
        print(self.Z.shape)
        Kzz = self.Kzz + (np.eye(self.Z.shape[0], dtype=np.float32) * self.jitter)
        Kxx = self.Kxx + (np.eye(X.shape[0], dtype=np.float32) * self.jitter)
        Kxz = self.Kxz
        Kzx = self.Kzx
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
                    return False
                    
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
                print("At iteration:", iteration, " loss is:", ELBO.numpy())  
                
        # Following completion of optimisation, store final variational parameters
        self.mean = surrogateMean
        self.covariance = surrogateLowerDiagonal
        return True

    # Wrapper function for training variational parameters
    def Run(self, iterations=500, learningRate=0.15, numOfSamples = 100, jitterStart=0.000001):  
        # If training fails due to inability to invert Kzz, jitter is increased
        self.jitter = jitterStart
        for i in range(10):
            if self.Train(iterations, learningRate, numOfSamples):
                print("Training successful!")
                return
            else:
                self.lossHistory = [] # Clear loss history
                self.jitter *= 10
                print("Increasing jitter to %0.5f" % self.jitter)