import jax.numpy as np

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

