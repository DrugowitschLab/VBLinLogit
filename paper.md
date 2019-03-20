---
title: 'VBLinLogit: Variational Bayesian linear and logistic regression'
tags:
  - MATLAB
  - linear regression
  - logistic regression
  - Variational Bayes
authors:
 - name: Jan Drugowitsch
   orcid: 0000-0002-7846-0408
   affiliation: 1
affiliations:
 - name: Harvard Medical School
   index: 1
date: 10 March 2019
bibliography: paper.bib
---

# Summary

Linear and logistic regression are essential workhorses of statistical analysis, whose Bayesian treatment has received much recent attention [@Gelman:2013; @Bishop:2006; @Murphy:2012; @Hastie:2011]. Using Bayesian statistics for linear and logistic regression allows specifying prior beliefs over certain model parameters, which makes it particularly useful for small and/or high-dimensional datasets. Bayesian regression furthermore provides an estimate of its uncertainty in the estimated regression coefficient, as well as uncertainty when making predictions arising from the regression. Both are again particularly important for small and/or high-dimensional datasets, as these might result in highly uncertain predictions, and allow the user to be explicit about this uncertainty.

``VBLinLogit`` is a MATLAB library that provides an variational Bayesian implementation of Bayesian models for both linear and logistic regression. It uses variational Bayesian inference [@Beal:2003; @Bishop:2006; @Murphy:2012] --- a method for approximating Bayesian computations --- as the Bayesian computations underlying the used models would otherwise be intractable. It is significantly faster than Markov Chain Monte Carlo (MCMC) methods [@Gilks:1995] --- another form of approximate Bayesian inference --- which makes it applicable to high-dimensional problems for which standard MCMC might be too slow.

A specific regression variant implemented by this library is automatic relevance determination (ARD), which uses a model that automatically determines which data dimensions are relevant for the regression, discarding the others [@Wipf:2007]. It does so without a separate "validation set", as would be required by alternative methods, like the Lasso [@Tibshirani:1996]. Therefore, it can be used when the small size of the dataset makes creating such a separate "validation set" prohibitive.

The scripts encompassing the library were written to be light-weight and thus do not depend on external libraries. They include variants with and without ARD. Their use is deliberately kept simple. The core arguments to the regression scripts are a matrix of predictors, as well as a vector of response variable. Additional parameters specifying the prior and hyper-prior parameters are usually optional. Details about the specifics of the used models, the variational Bayes derivation, and detailed use of the scripts included in the library can be found in Drugowitsch [-@Drugowitsch:2013].

# Novelty and relation to other work

The models underlying variational Bayesian linear and logistic regression implemented in this library are Bayesian hierarchical models with priors on the regression coefficient, as well as hyper-priors on the prior parameters. For the ARD variants, the hyper-priors are assigned to each of the regressors separately, which supports pruning eventually irrelevant coefficients [@Wipf:2007]. This happens without the need for a separate validation set, unlike comparable sparsity-inducing methods like the Lasso [@Tibshirani:1996]. Bishop [-@Bishop:2006] describes ARD only in the  context of type-II maximum likelihood [@MacKay:1992; @Neal:1996; @Tipping:2001], where it (hyper-)parameters are tuned by maximizing the marginal likelihood (or model evidence). The library instead provides implementations for the full Bayesian treatment, that finds the ARD hyper-posteriors by variational Bayesian inference.

Since release ``R2017a``, MATLAB also provides some functions for Bayesian linear regression, but none for Bayesian logistic regression. For linear regression, it provides variants with and without variable selection. Neither of the variants without variable selection use and infer hyper-priors. Therefore, they do not support inferring prior parameters in hierarchical models, unlike this library. The variants with variable selection either use a straight-foward Bayesian formulation of Lasso, or Stochastic Search Variable Selection (SSVS) [@George:1993]. Both are different from ARD, and the advantages of either methods remain to be clarified.

Some of the scripts provided by this library have been included in the ``pmtk3`` software accompanying Murphy [-@Murphy:2012]. It has furthermore been used for various scientific publications across multiple domains [e.g., @Kanitscheider:2015; @Oh:2016; @Wang:2017; @Ruxanda:2018].

# References
