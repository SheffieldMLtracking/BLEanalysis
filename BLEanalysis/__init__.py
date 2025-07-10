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
    ax.add_patch(el)
    return ellipse
    


