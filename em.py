"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
from scipy.stats import multivariate_normal as norm

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    Cu = X > 0
    likelihoods = np.empty([n, K])
    for i in range(K):
        for j in range(n):
            likelihoods[j,i] = norm.pdf(
                X[j][Cu[j]], 
                mixture.mu[i,Cu[j]], 
                mixture.var[i]*np.identity(Cu[j].sum())
                )
            print(d, n, Cu[j].sum(), j)
    log_likelihoods = np.log(likelihoods)
    f = log_likelihoods + np.log(mixture.p)
    log_post = f.T - np.log(np.exp(f).sum(axis=1))
    return np.exp(log_post).T, f.sum()

def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]
    Cu = X > 0
    mu = np.empty([K, d])
    var = np.empty(K)
    p = post.mean(axis=0)
    for i in range(K):
        Cu_P = Cu[:,i]*post[:,i]
        if Cu_P.sum() >= 1:
            mu[i,:] = (Cu_P*X.T).sum(axis=1)/(Cu[:,i]*post[:,i]).sum()
        else:
            mu[i,:] = mixture.mu[i,:]
        s1, s2 = 0, 0
        for j in range(n):
            s1 += p[j,i]*((X[j,Cu[j]] - mu[i,Cu[j]])**2).sum()
            s2 += Cu[j].sum()*p[j,i]
        var[i] = max(min_variance, s1/s2)
    
    return GaussianMixture(mu, var, p)
    


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    mix = mixture
    new_log_likelihood = 0
    old_log_likelihood = -np.inf
    while abs((new_log_likelihood - old_log_likelihood)) > 1e-6*abs(new_log_likelihood):
        old_log_likelihood = new_log_likelihood
        post, new_log_likelihood = estep(X, mix)
        mix = mstep(X, post, mix)
    return mix, post, new_log_likelihood


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    post, _ = estep(X, mixture)
    n = X.shape[0]
    for i in range(n):
        X[n][X[n] == 0] = mixture.mu[np.argmax(post[i]), X[n]==0]
    return X 
