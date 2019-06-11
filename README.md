VBLinLogit
==========

This library provides stand-alone MATLAB/Octave code to perform variational Bayesian linear and logistic regression. In contrast to standard linear and logistic regression, the library assumes priors over the parameters whose parameters are tuned by variational Bayesian inference, to avoid overfitting. Specifically, it supports a fully Bayesian version of automatic relevance determination (ARD), which is a sparsity-promoting prior that prunes regression coefficients that are deemed irrelevant. 

Linear regression is available in the following two variants:

*   Variational Bayesian linear regression with ARD.

    In this case, the prior on the weight vector is a zero-mean multivariance Gaussian with a diagonal matrix in which each diagonal element is modeled separately by an inverse-Gamma hyper-prior.
    
*   Variational Bayesian linear regression without ARD.

Logistic regression is available in the following two variants:

*   Variational Bayesian logistic regression with ARD.

    In this case, the prior on the weight vector is a zero-mean multivariance Gaussian with a diagonal matrix in which each diagonal element is modeled separately by an inverse-Gamma hyper-prior.

*   Variational Bayesian logistic regression without ARD.
   
    The model is the same as for the ARD variant, only that all the elements of the diagonal covariance matrix are modeled in combination by the same inverse-Gamma hyper-prior.

The code is licensed under the New BSD License.

Installation
------------

Download [v0.2](https://github.com/DrugowitschLab/VBLinLogit/archive/v0.2.zip) or [the latest version](https://github.com/DrugowitschLab/VBLinLogit/archive/master.zip) of VBLinLogit and extract the downloaded file to a folder of your choice, or clone the repository. To use within MATLAB/Octave, add the folder to the search path, either using the GUI in MATLAB or by calling
```Matlab
>> addpath('/path/to/VBLinLogit/src')
```
at the MATLAB/Octave command line. See the MATLAB/Octave documentation for how to save this search path for use in future MATLAB/Octave sessions.

Requirements
------------

All scripts have been tested in MATLAB R2018a and Octave v5.1.0, but should work with earlier MATLAB/Octave versions. In particular, they should be compatible with MATLAB starting R2007a. Please file an issue if you identify incompatibility with earlier MATLAB/Octave versions.

Some linear regression example scripts rely on the MATLAB Statistics and Machine Learning Toolbox to estimate the regression coefficient confidence intervals. These function won't plot confidence intervals if this toolbox isn't installed.

Use and documentation
---------------------

The library source code resides in the [`src`](src) folder. The header of each script in that folder provides a description of the function it performs.

See the [`examples`](examples) folder for example use of the different scripts in the `src` folder.

Find more information about derivation and use in *Variational Bayesian
inference for linear and logistic regression*, [arxiv:1310.5438](http://arxiv.org/abs/1310.5438) [stat.ML]

Contributing
------------

For contributions and bug reports, please see the [contribution guidelines](CONTRIBUTING.md).
