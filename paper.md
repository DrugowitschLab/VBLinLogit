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

Linear and logistic regression are essential workhorses of statistical analysis, whose Bayesian treatment has received much recent attention [@Gelman:2013; @Bishop:2006; @Murphy:2012; @Hastie:2011]. These allow specifying the a-priori uncertainty and infer a-posteriori uncertainty about regression coefficients explicitly and hierarchically, by, for example, specifying how uncertain we are a-priori that these coefficients are small. However, Bayesian inference in such hierarchical models quickly becomes intractable, such that recent effort has focused on approximate inference, like Markov Chain Monte Carlo methods [@Gilks:1995], or variational Bayesian approximation [@Beal:2003; @Bishop:2006; @Murphy:2012).

``VBLinLogit`` is a MATLAB library that provides an variational Bayesian implementation of Bayesian hierarchical models for both linear and logistic regression. Both models include a variant with automatic relevance determination (ARD), which consists of assigning an individual hyper-prior to each regression coefficient separately. These  hyper-priors are adjusted to eventually prune irrelevant coefficients [@Wipf:2007] without the need for a separate validation set, unlike comparable sparsity-inducing methods like the Lasso [@Tibshirani:1996]. [-@Bishop:2006] describes ARD only in the  context of type-II maximum likelihood [@MacKay:1992; @Neal:1996; @Tipping:2001], where it (hyper-)parameters are tuned by maximizing the marginal likelihood (or model evidence). Here, instead, we provide implementations for the full Bayesian treatment, that finds the ARD hyper-posteriors by variational Bayesian inference.

The scripts encompassing the library were written to be light-weight and thus do not depend on external libraries. Details about the variational Bayes derivation and use of the library can be found in [-@Drugowitsch:2013].

# References
