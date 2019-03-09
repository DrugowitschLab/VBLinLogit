VBLinLogit
==========

MATLAB code to perform Variational Bayesian linear and logistic regression. Linear regression is available in following two variants:

*   Variational Bayesian linear regression with Automatic Relevance Determination (ARD).

    In this case, the prior on the weight vector is a zero-mean multivariance Gaussian with a diagonal matrix in which each diagonal element is modeled separately by an inverse-Gamma hyper-prior.
    
*   Variational Bayesian linear regression without ARD.

Logistic regression is available in the following two variants:

*   Variational Bayesian logistic regression with Automatic Relevance Determination (ARD).

    In this case, the prior on the weight vector is a zero-mean multivariance Gaussian with a diagonal matrix in which each diagonal element is modeled separately by an inverse-Gamma hyper-prior.

*   Variational Bayesian logistic regression without ARD.
   
    The model is the same as for the ARD variant, only that all the elements of the diagonal covariance matrix are modeled in combination by the same inverse-Gamma hyper-prior.

The code is licensed under the New BSD License.

Documentation
-------------

Find more information about derivation and use in *Variational Bayesian
inference for linear and logistic regression*, [arxiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML]
