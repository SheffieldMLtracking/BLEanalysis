import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import jax.numpy as jnp

def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )

    copied from https://stackoverflow.com/a/55688087
    """
    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = jnp.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = jnp.trace(iS1 @ S0)
    det_term  = jnp.log(jnp.linalg.det(S1)/jnp.linalg.det(S0))
    quad_term = diff.T @ jnp.linalg.inv(S1) @ diff
    return .5 * (tr_term + det_term + quad_term - N) 

def confidence_ellipse(mean, cov, ax, n_std=3.0, fill=False, opacity=1, **kwargs):
    """Draws an ellipse to illustrate a Gaussian with given mean and covariance.
    
    Parameters:
     mean: the location of the Gaussian
     cov: the covariance of the Gaussian
     ax: what axis to draw on
     n_std: Which contour to draw (standard deviations from the mean), default 3.
     fill: boolean, whether to fill the ellipose (default: false)
     opacity: how opaque to make the ellipse (default 1.0).
     **kwargs: pass other keywords to the Ellipse constructor.
    
    Returns the ellipse drawing object, can be drawn with ax.add_patch(el)
    TODO: Convert to using a passed axis object (usually 'ax', but confusingly this is already a parameter).
    """
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, fill=fill, alpha=opacity, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    return ellipse
    


