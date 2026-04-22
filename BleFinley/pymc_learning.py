import scipy
from IPython.display import display
import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, cauchy, laplace

def learn_cauchy(rss_diffs :list[int]):
    """
    Learns the parameter beta of the cauchy distribution
    and then plots it overtop of the histogram
    """
    with pm.Model():
        alpha = 0
        beta = pm.HalfNormal("beta", sigma=2)

        obs = pm.Cauchy("obs", alpha=alpha, beta=beta, observed=rss_diffs)

        trace = pm.sample(2000, tune=1000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs))
    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True)
    plt.plot(x_vals, cauchy.pdf(x_vals, scale=summary.loc["beta", "mean"]))

    plt.plot(x_vals, norm.pdf(x_vals, 0, 3.5))

    plt.show()


def learn_laplace(rss_diffs :list[int]):
    with pm.Model():
        mu = 0
        std = pm.HalfNormal("std", sigma=2)

        obs = pm.Laplace("obs", mu=mu, b=std, observed=rss_diffs)

        trace = pm.sample(2000, tune=1000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs))
    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True)
    plt.plot(x_vals, laplace.pdf(x_vals, loc=0, scale=summary.loc["std", "mean"]))

    plt.show()


def learn_laplace_norm_mix(rss_diffs :list[int]):
    with pm.Model():
        w = pm.Dirichlet("w", a=np.array([1, 1]))
        sigma = pm.Normal("sigma", mu=3, sigma=2)
        b = pm.Uniform("b", lower=0, upper=5)

        normal = pm.Normal.dist(mu=0, sigma=sigma)
        laplace = pm.Laplace.dist(mu=0, b=b)

        obs = pm.Mixture("obs", w=w, comp_dists=[normal, laplace], observed=rss_diffs)

        trace = pm.sample(3000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs))
    l = scipy.stats.laplace.pdf(x_vals, 0, summary.loc["b", "mean"])
    n = norm.pdf(x_vals, 0, summary.loc["sigma", "mean"])

    mix = summary.loc["w[0]", "mean"] * n + summary.loc["w[1]", "mean"] * l

    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True)
    plt.plot(x_vals, mix)
    plt.show()

def learn_cauchy_norm_mix(rss_diffs :list[int]):
    with pm.Model():
        w = pm.Dirichlet("w", a=np.array([1, 1]))
        sigma = pm.HalfNormal("sigma", sigma=4)
        beta = pm.HalfNormal("beta", sigma=4)

        normal_comp = pm.Normal.dist(mu=0, sigma=sigma)
        cauchy_comp = pm.Cauchy.dist(alpha=0, beta=beta)

        obs = pm.Mixture(
            "obs",
            w=w,  # [w, 1 - w],
            comp_dists=[normal_comp, cauchy_comp],
            observed=rss_diffs
        )

        trace = pm.sample(1000, tune=2000, target_accept=0.9, return_inferencedata=True)

    display(az.summary(trace))