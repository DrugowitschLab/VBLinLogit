# VBLinLogit

This library provides stand-alone MATLAB/Octave code to perform variational Bayesian linear and logistic regression. In contrast to standard linear and logistic regression, the library assumes priors over the parameters whose parameters are tuned by variational Bayesian inference, to avoid overfitting. Specifically, it supports a fully Bayesian version of automatic relevance determination (ARD), which is a sparsity-promoting prior that prunes regression coefficients that are deemed irrelevant. 

Linear regression is available in the following two variants:

*   Variational Bayesian linear regression with ARD: assumes a zero-mean multivariate Gaussian prior on the weight vector, for which each element along the diagonal of the covariance matrix is modeled separately by an inverse-Gamma hyper-prior.
    
*   Variational Bayesian linear regression without ARD.

Logistic regression is available in the following two variants:

*   Variational Bayesian logistic regression with ARD: assumes a zero-mean multivariate Gaussian prior on the weight vector, for which each element along the diagonal of the covariance matrix is modeled separately by an inverse-Gamma hyper-prior.

*   Variational Bayesian logistic regression without ARD: assumes the same model as for the ARD variant, only that all elements of the diagonal covariance are modeled jointly by the same inverse-Gamma hyper-prior.

The code is licensed under the New BSD License.

## Installation

Download [v0.2](https://github.com/DrugowitschLab/VBLinLogit/archive/v0.2.zip) or [the latest version](https://github.com/DrugowitschLab/VBLinLogit/archive/master.zip) of VBLinLogit and extract the downloaded file to a folder of your choice, or clone the repository. To use within MATLAB/Octave, add the folder to the search path, either using the GUI in MATLAB or by calling
```Matlab
>> addpath('/path/to/VBLinLogit/src')
```
at the MATLAB/Octave command line. See the MATLAB/Octave documentation for how to save this search path for use in future MATLAB/Octave sessions.

The installation can be checked by running the tests in the [`test`](test) folder.

## Requirements

All scripts have been tested in MATLAB R2018a and Octave v5.1.0, but should work with earlier MATLAB/Octave versions. In particular, they should be compatible with MATLAB starting R2007a. Please file an issue if you identify incompatibility with earlier MATLAB/Octave versions.

Some linear regression example scripts rely on the MATLAB Statistics and Machine Learning Toolbox to estimate the regression coefficient confidence intervals. These function won't plot confidence intervals if this toolbox isn't installed.

## Usage and documentation

The library source code resides in the [`src`](src) folder. The below provides a brief description of the API for the different functions. The header of each function file provides a more extended description of the function it performs. For a more extended discussion of the derivations and the use, please consult *Variational Bayesian
inference for linear and logistic regression*, [arxiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML].

See the [`examples`](examples) folder for example use of the different scripts in the `src` folder.

In all of the below, `D` is the dimensionality of the input, the output is one-dimensional, and `N` is the number of data points in the training set. For both linear and logistic regression, the training set is specified by the `N x D` matrix `X`, and the `N`-element column vector `y`. The `n`th row in `X` specifies one `D`-element input vector that corresponds to the output given by the `n`th element of `y`. For linear regression, these outputs are expected to be scalars. For logistic regression, they are `-1` or `1`.

### Variational Bayesian linear regression

#### Model fitting

```Matlab
[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(X, y)
[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit(X, y, a0, b0, c0, d0)
```
fits variational Bayesian linear regression without ARD to the training data given by `X` and `y`. The optional scalars `a0`, `b0`, `c0`, and `d0` specify the prior and hyper-prior parameters. The function returns the posterior weight mean vector `w` and covariance matrix `V`, as well as its inverse `invV` and scalar log-determinant `logdetV`. It furthermore returns the scalar posterior precision parameters, `an` and `bn`, the hyper-posterior mean `E_a`, as well as the variational bound `L`.

```Matlab
[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit_ard(X, y)
[w, V, invV, logdetV, an, bn, E_a, L] = vb_linear_fit_ard(X, y, a0, b0, c0, d0)
```
is similar to `vb_linear_fit(.)`, but uses an ARD prior. Thus, it returns the hyper-posterior mean vector, `E_a`, rather than a scalar.

#### Model predictions

```Matlab
[mu, lambda, nu] = vb_linear_pred(X, w, V, an, bn)
```
for a fitted variational Bayesian linear regression model, predicts the outputs for the given `K x D` input matrix `X`, with one input vector per row. The additional arguments `w`, `V`, `an`, and `bn` are those returned by `vb_linear_fit[_ard]`. The function returns the posterior predictive means `mu`, precisions `lambda`, and degrees of freedom `nu`. `mu` and `lambda` are `K`-element vectors, and `nu` is a scalar that is shared by all outputs.

### Variational Bayesian logistic regression

#### Model fitting

```Matlab
[w, V, invV, logdetV, E_a, L] = vb_logit_fit(X, y)
[w, V, invV, logdetV, E_a, L] = vb_logit_fit(X, y, a0, b0)
```
fits variational Bayesian logistic regression without ARD, but a global shrinkage prior, to the training data given by `X` and `y`. The optional scalars `a0` and `b0` specify the parameters of the shrinkage prior. The function returns the posterior weight mean vector `w` and covariance matrix `V`, as well as its inverse `invV` and scalar log-determinant `logdetV`. It furthermore returns the scalar posterior shrinkage mean, `E_a`, as well as the variational bound `L`.

```Matlab
[w, V, invV, logdetV] = vb_logit_fit_iter(X, y)
```
is similar to `vb_logit_fit(.)`, but uses only a weak pre-specified shrinkage prior. Thus, it does not support specifying `a0` and `b0`, and doesn't return `E_a`. Furthermore, iterates over the inputs separately rather than processing them all at once, and is therefore slower, but also computationally more stable as it avoids computing the inverse of possibly close-to-singular matrices.

```Matlab
[w, V, invV, logdetV, E_a, L] = vb_logit_fit_ard(X, y)
[w, V, invV, logdetV, E_a, L] = vb_logit_fit_ard(X, y, a0, b0)
```
is similar to `vb_logit_fit(.)`, but uses an ARD prior. Thus, it returns the posterior shrinkage mean vector, `E_a`, rather than a scalar.

#### Model predictions

Please note that all prediction functions return the probabilities `p(y=1 | x, ...)` rather than the most likely `y`'s for the given inputs. How to turn these probabilities into predicted `y` depends on the loss function. For a standard `0-1` loss, the rational choice would be to predict `y=1` if `p(y=1 | x, ...) > 0.5`, and `y=-1` otherwise.

```Matlab
out = vb_logit_pred(X, w, V, invV)
```
for a fitted variational Bayesian logistic regression model, predicts `p(y=1 | x)` for the given `K x D` input matrix `X`, with one input vector `x` per row. The additional arguments `w`, `V`, `invV`, are those returned by `vb_linear_fit[_*]`. The returned `K`-element vector contains the posterior predictive probabilities `p(y=1 | x)`, one element for each row in `X`.

```Matlab
out = vb_logit_pred_incr(X, w, V, invV)
```
is similar to `vb_logit_pred`, but rather than computing all predictions simultaneously, it does so for each row of `X` separately by iterating over the rows of `X`.

## Contributing

For contributions and bug reports, please see the [contribution guidelines](CONTRIBUTING.md).
