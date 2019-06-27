---
title: 'VBLinLogit: Variational Bayesian linear and logistic regression'
tags:
  - MATLAB
  - Octave
  - linear regression
  - logistic regression
  - Variational Bayes
authors:
 - name: Jan Drugowitsch
   orcid: 0000-0002-7846-0408
   affiliation: 1
affiliations:
 - name: Department of Neurobiology, Harvard Medical School, Boston, MA 02115, USA
   index: 1
date: 10 March 2019
bibliography: paper.bib
---

# Summary

Linear and logistic regression are essential workhorses of statistical analysis, whose Bayesian treatment has received much recent attention [@Gelman:2013; @Bishop:2006; @Murphy:2012; @Hastie:2011]. Using Bayesian statistics for linear and logistic regression allows specifying prior beliefs over certain model parameters, which makes it particularly useful for small and/or high-dimensional datasets. Bayesian regression furthermore provides an estimate of the uncertainty about estimated regression coefficients, as well as uncertainty about predictions arising from the regression. Both are again particularly important for small and/or high-dimensional datasets, as the use of such data might result in highly uncertain predictions, and allow the user to be explicit about this uncertainty.

``VBLinLogit`` is a MATLAB/Octave library that provides a variational Bayesian implementation of Bayesian models for both linear and logistic regression. It uses variational Bayesian inference [@Beal:2003; @Bishop:2006; @Murphy:2012] as a method for approximating Bayesian computations, as these computations would otherwise be intractable for the used regression models. It is significantly faster than Markov Chain Monte Carlo (MCMC) methods [@Gilks:1995], another form of approximate Bayesian inference, which makes it applicable to high-dimensional problems for which standard MCMC might be too slow.

A specific regression variant implemented by this library is automatic relevance determination (ARD), which uses a model that automatically determines which data dimensions are relevant for the regression, discarding the others [@Wipf:2007]. It does so without a separate "validation set", as would be required by alternative methods, like the Lasso [@Tibshirani:1996]. Therefore, it can be used when the small size of the dataset makes the use of such a separate "validation set" prohibitive.

The scripts encompassing the library were written to be light-weight and thus do not depend on external libraries. They include variants with and without ARD. Their use is deliberately kept simple. The core arguments to the regression scripts are a matrix of predictors, as well as a vector of response variable. Additional parameters specifying the prior and hyper-prior parameters are usually optional. Details about the specifics of the used models, the variational Bayes derivation, and detailed use of the scripts included in the library can be found in Drugowitsch [-@Drugowitsch:2013].

# Additional details, novelty, and relation to other approaches

Use of the library is particularly beneficial if the data is sparse. Sparsity can occur either if few training examples are available, or if the dimensionality of the input (i.e., the number of dependent variables) is large. Data sparsity becomes particularly challenging if the input dimensionality exceeds the number of training examples. In this case, the regression is underdetermined: multiple solutions exist that fit the training set equally well. However, only some of them yield good predictions on a separate test set.

A common approach to handle underdetermination is to make additional assumptions about potential solutions. Specifically, it is commonly assumed that the regression weights (i.e., the regression coefficients) that form the solution and describe how the output (i.e., the independent variable) varies with the inputs, take small values. This is known as _regularization_. Regularization isn't only beneficial if the regression is underdetermined, but also if the data is noisy. The different methods discussed here differ in how exactly they introduce regularization by making different a-priori assumptions about the regression weights.

The Bayesian methods provided in this library implement two different sets of assumptions. They either assume that all regression coefficients are equally small and tunes how small they are overall (that's the variant without ARD), or they adjust "smallness" of each regression coefficient individually (that's the variant with ARD). The latter is particularly beneficial if the data includes some spurious input dimensions that don't determine the outputs. In this case, the ARD variant might be able to set the associated regression weights to zero, effectively ignoring these input dimensions. As mentioned further above, another benefit of Bayesian regularization is that it doesn't need a separate "validation set" to tune its parameters. How well this actually works depends on how close the data matches the assumptions underlying the different methods. Thus, different regularization approaches might work better or worse on different datasets. There currently exists no single best method that works best for all datasets.

More specifically, the models underlying variational Bayesian linear and logistic regression implemented in this library are Bayesian hierarchical models with priors on the regression coefficient, as well as hyper-priors on the prior parameters. For the ARD variants, the hyper-priors are assigned to each of the regressors separately, which supports pruning eventually irrelevant coefficients [@Wipf:2007]. This happens without the need for a separate validation set, unlike comparable sparsity-inducing methods like the Lasso [@Tibshirani:1996]. Bishop [-@Bishop:2006] describes ARD only in the  context of type-II maximum likelihood [@MacKay:1992; @Neal:1996; @Tipping:2001], in which case the (hyper-)parameters are tuned by maximizing the marginal likelihood (or model evidence). The library instead provides implementations for the full Bayesian treatment, that finds the ARD hyper-posteriors by variational Bayesian inference.

Since release ``R2017a``, MATLAB also provides some functions for Bayesian linear regression, but none for Bayesian logistic regression. For linear regression, it provides variants with and without variable selection. Neither of the variants without variable selection use or infer hyper-priors. Therefore, they do not support inferring prior parameters in hierarchical models, unlike this library. The variants with variable selection either use a straight-foward Bayesian formulation of Lasso, or Stochastic Search Variable Selection (SSVS) [@George:1993]. Both are different from ARD, and the advantages of either method remain to be clarified.

Some of the scripts provided by this library have been included in the ``pmtk3`` software accompanying Murphy [-@Murphy:2012]. Part of the library has furthermore been used for various scientific publications across multiple domains [e.g., @Kanitscheider:2015; @Oh:2016; @Wang:2017; @Ruxanda:2018].

# References
