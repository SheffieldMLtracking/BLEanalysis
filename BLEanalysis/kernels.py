import jax.numpy as np
import jax.numpy as jnp
from jax import vmap, jit
from jax.lax import select, gt
import jax.scipy as jsp

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

class EQIntegralKernel:
    def __init__(self, lengthscale, scalefactor):
        self.lengthscale = lengthscale
        self.scalefactor = scalefactor

    def g(self, z):
        return z * jnp.sqrt(jnp.pi) * jsp.special.erf(z) + jnp.exp(-z**2)

    def k_xx(self, x, xprime):
        l = self.lengthscale
        return 0.5 * (l ** 2) * (
            self.g(x / l) 
            - self.g((x - xprime) / l) 
            + self.g(xprime / l) 
            - 1.0
        )

    def K(self, X, Xprime):
        """Compute covariance matrix between X and Xprime.
        
        Both X and Xprime are arrays of shape [N, 2], where
        - X[:, 0] are the inputs
        - X[:, 1] are the class/label/group indices (for masking)

        Returns:
            Covariance matrix of shape [N, M].
        """
        # Extract coordinates and labels
        x1, l1 = X[:, 0], X[:, 1]
        x2, l2 = Xprime[:, 0], Xprime[:, 1]

        # Compute full pairwise kernel matrix
        k_fn = vmap(lambda xi: vmap(lambda xj: self.k_xx(xi, xj))(x2))(x1)
        cov = k_fn * (self.scalefactor ** 2)

        # Mask out entries where labels are equal
        mask = (l1[:, None] == l2[None, :])
        return cov * mask.astype(jnp.float32)