vb_linear
=========

MATLAB code to perform Variational Bayesian linear regression. Two variants are available:

*   Variational Bayesian linear regression with Automatic Relevance Determination (ARD).

    In this case, the prior on the weight vector is a zero-mean multivariance Gaussian with a diagonal matrix in which each diagonal element is modeled separately by an inverse-Gamma hyper-prior.
    
*   Variational Bayesian linear regression without ARD.

    The model is the same as for the ARD variant, only that all the elements of the diagonal covariance matrix are modeled in combination by the same inverse-Gamma hyper-prior.

The code is licensed under the New BSD License.

Documentation
-------------

Find more information on the [homepage of Jan Drugowitsch](http://www.lnc.ens.fr/~jdrugowi/code_vb.html)
