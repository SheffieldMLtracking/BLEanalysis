import scipy
from IPython.display import display
import pymc as pm
import arviz as az
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, cauchy, laplace

X_LABEL = "Δy (dBm)"
Y_LABEL = "RSS (dBm)"

def learn_normal(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        sigma = pm.HalfNormal("sigma", sigma=2)
        obs = pm.Normal("obs", mu=mu, sigma=sigma, observed=rss_diffs)
        trace = pm.sample(2000, tune=1000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, norm.pdf(x_vals, mu, summary.loc['sigma', 'mean']), label="Normal PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()


def learn_cauchy(rss_diffs :list[int]):
    """
    Learns the parameter beta of the cauchy distribution
    and then plots it overtop of the histogram
    """
    mu = np.mean(rss_diffs)

    with pm.Model():
        beta = pm.HalfNormal("beta", sigma=2)

        obs = pm.Cauchy("obs", alpha=mu, beta=beta, observed=rss_diffs)

        trace = pm.sample(2000, tune=1000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, cauchy.pdf(x_vals, scale=summary.loc["beta", "mean"]), label="Cauchy PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()


def learn_laplace(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        std = pm.HalfNormal("std", sigma=2)

        obs = pm.Laplace("obs", mu=mu, b=std, observed=rss_diffs)

        trace = pm.sample(2000, tune=1000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, laplace.pdf(x_vals, loc=mu, scale=summary.loc["std", "mean"]), label="Laplace PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()


def learn_norm_norm_mix(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        w = pm.Dirichlet("w", a=np.array([1, 1]))
        sigma1 = pm.HalfNormal("sigma1", sigma=2)
        sigma2 = pm.HalfNormal("sigma2", sigma=2)

        normal1 = pm.Normal.dist(mu=mu, sigma=sigma1)
        normal2 = pm.Normal.dist(mu=mu, sigma=sigma2)

        obs = pm.Mixture("obs", w=w, comp_dists=[normal1, normal2], observed=rss_diffs)
        trace = pm.sample(3000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    n1 = norm.pdf(x_vals, mu, summary.loc["sigma1", "mean"])
    n2 = norm.pdf(x_vals, mu, summary.loc["sigma2", "mean"])

    mix = summary.loc["w[0]", "mean"] * n1 + summary.loc["w[1]", "mean"] * n2

    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, mix, label="Normal-Normal Mixture PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()


def learn_laplace_norm_mix(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        w = pm.Dirichlet("w", a=np.array([1, 1]))
        sigma = pm.Normal("sigma", mu=3, sigma=2)
        b = pm.Uniform("b", lower=0, upper=5)

        normal_dist = pm.Normal.dist(mu=mu, sigma=sigma)
        laplace_dist = pm.Laplace.dist(mu=mu, b=b)

        obs = pm.Mixture("obs", w=w, comp_dists=[normal_dist, laplace_dist], observed=rss_diffs)

        trace = pm.sample(3000)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    l = scipy.stats.laplace.pdf(x_vals, mu, summary.loc["b", "mean"])
    n = norm.pdf(x_vals, mu, summary.loc["sigma", "mean"])

    mix = summary.loc["w[0]", "mean"] * n + summary.loc["w[1]", "mean"] * l

    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, mix, label="Laplace-Normal Mixture PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()

def learn_cauchy_norm_mix(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        w = pm.Dirichlet("w", a=np.array([1, 1]))
        sigma = pm.HalfNormal("sigma", sigma=4)
        beta = pm.HalfNormal("beta", sigma=4)

        normal_comp = pm.Normal.dist(mu=mu, sigma=sigma)
        cauchy_comp = pm.Cauchy.dist(alpha=mu, beta=beta)

        obs = pm.Mixture(
            "obs",
            w=w,
            comp_dists=[normal_comp, cauchy_comp],
            observed=rss_diffs
        )

        trace = pm.sample(1000, tune=2000, target_accept=0.9, return_inferencedata=True)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    n = norm.pdf(x_vals, mu, summary.loc["sigma", "mean"])
    c = cauchy.pdf(x_vals, scale=summary.loc["beta", "mean"])

    mix = summary.loc["w[0]", "mean"] * n + summary.loc["w[1]", "mean"] * c

    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, mix, label="Cauchy-Normal Mixture PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()

def learn_triple_norm_mix(rss_diffs :list[int]):
    mu = np.mean(rss_diffs)

    with pm.Model():
        w = pm.Dirichlet("w", a=np.ones(3))
        sigma = pm.Uniform("sigma", lower=1, upper=7, shape=3)

        components = pm.Normal.dist(mu=mu, sigma=sigma, shape=3)

        obs = pm.Mixture("obs", w=w, comp_dists=components, observed=rss_diffs)
        trace = pm.sample(2000, tune=1000, return_inferencedata=True)

    summary = az.summary(trace)
    display(az.summary(trace))

    # plot PDF over histogram
    x_vals = np.linspace(min(rss_diffs), max(rss_diffs)).round().astype(int)
    n1 = norm.pdf(x_vals, mu, summary.loc["sigma[0]", "mean"])
    n2 = norm.pdf(x_vals, mu, summary.loc["sigma[1]", "mean"])
    n3 = norm.pdf(x_vals, mu, summary.loc["sigma[2]", "mean"])

    mix = summary.loc["w[0]", "mean"] * n1 + summary.loc["w[1]", "mean"] * n2 + summary.loc["w[2]", "mean"] * n3

    plt.hist(rss_diffs, bins=x_vals, alpha=0.6, density=True, label="Δy histogram")
    plt.plot(x_vals, mix, label="Triple-Normal Mixture PDF")

    plt.ylabel(Y_LABEL)
    plt.xlabel(X_LABEL)
    plt.legend()
    plt.show()