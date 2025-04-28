import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np


# TODO Work our which of these imports are necessary for this file


class Kernel:
    def __init__(self):
        """Base kernel class"""
        raise NotImplementedError
        
    def K(self, X, Xprime):
        raise NotImplementedError
        
class ExponentiatedQuadraticKernel(Kernel):
    def __init__(self, lengthscale, scalefactor):
        """Computes the EQ kernel.
        
        the covariance method returns: scale^2 * exp((-(x-x')^2) / (2*ls^2))
        
        Parameters:
         lengthscale: lengthscale of the kernel
         scalefactor: the premultiplier.

        """
        self.lengthscale = lengthscale
        self.scalefactor = scalefactor
        
    def K(self, X, Xprime):
        """EQ kernel: scale^2 * exp((-(x-x')^2) / (2*ls^2))
        
        Parameters:
        X and Xprime: TODO What shape can these be?
        
        Returns the covariacne between points in X and Xprime.
        
        TODO You could just use - instead of np.subtract. Are you sure that's 'tf compatible'?
        TODO What is the axsel / tf.cast for?
        """
        covariance = (self.scalefactor ** 2) * np.exp(-np.sum((np.subtract(X[:, None], Xprime[None, :])) ** 2 / (2 * self.lengthscale ** 2), 2))
        axsel = tf.cast((X[:,1][:,None]==Xprime[:,1][None,:]), dtype=tf.float32)
        covariance *= axsel
        return covariance
